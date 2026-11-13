"""UI adapter for the actual vendored source SAM2CameraPredictor.

Implementation provenance: D:/segment-anything-2-real-time-main/demo/demo.py
and sam2/sam2_camera_predictor.py (see vendor/source_camera/SOURCE_MANIFEST.json).
No old SAMURAI recovery, color matching or additional AI model is invoked.
"""
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from video_profile import timed

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor/source_camera'


def load_source_builder():
    # Do not silently mix two incompatible packages named sam2 in one process.
    for name, module in tuple(sys.modules.items()):
        if name == 'sam2' or name.startswith('sam2.'):
            path = getattr(module, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(VENDOR):
                raise RuntimeError(f'Conflicting SAM2 already imported: {path}. Start camera.py in a fresh process.')
    sys.path.insert(0, str(VENDOR)) if str(VENDOR) not in sys.path else None
    from sam2.build_sam import build_sam2_camera_predictor
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(VENDOR):
        raise RuntimeError(f'Wrong SAM2 package: {sam2.__file__}')
    return build_sam2_camera_predictor


class SourceCameraTracker:
    target_mode = 'SAM_ONLY'

    def __init__(self, checkpoint=None, image_size=512, model_config=None, precision='bf16'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Source SAM2CameraPredictor uses CUDA internally; CPU fallback is not supported by this port.')
        self.device = torch.device('cuda')
        self.dtype = torch.bfloat16 if precision == 'bf16' and torch.cuda.is_bf16_supported() else torch.float16
        checkpoint = Path(checkpoint or ROOT/'sam2/checkpoints/sam2.1_hiera_tiny.pt')
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
        config = model_config or ('configs/sam2.1/sam2.1_hiera_t_512.yaml' if image_size == 512
                                  else 'configs/sam2.1/sam2.1_hiera_t.yaml')
        build = load_source_builder()
        overrides = [f'++model.compile_{part}=false' for part in
                     ('image_encoder','memory_encoder','memory_attention','prompt_encoder','mask_decoder')]
        self.predictor = build(config, str(checkpoint), device='cuda', hydra_overrides_extra=overrides)
        if self.predictor.image_size != image_size:
            raise ValueError(f'Config image size {self.predictor.image_size} differs from --image-size {image_size}')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.correct = self.wrong = self.reacquired = self.attempt = 0
        self.reset()
        print(f'Source tracker: {type(self.predictor).__module__}.{type(self.predictor).__name__}\n'
              f'Package: {VENDOR}\nConfig: {config}\nCheckpoint: {checkpoint}\n'
              f'Device: {torch.cuda.get_device_name()} | torch {torch.__version__} | {self.dtype}\n'
              'Source tracking replaces legacy recovery/crop/probe policies; no compiled CUDA graph.', flush=True)

    def reset(self):
        # Source load_first_frame resets dictionaries but not frame_idx.
        self.predictor.condition_state = {}
        self.predictor.frame_idx = 0
        self.active = False
        self.state = 'WAIT_SELECTION'
        self.shape = None

    def warmup(self):
        self.warming = True
        frame = np.zeros((128,128,3), np.uint8)
        self.step(frame, 0., [32,32,96,96])
        self.step(frame, .033)
        torch.cuda.synchronize()
        self.warming = False
        self.correct = self.wrong = self.reacquired = self.attempt = 0
        self.reset()

    @torch.inference_mode()
    @timed('sam')
    def step(self, frame, stamp, box=None, mask_prompt=None):
        start = time.perf_counter()
        before = self.state
        if box is None and not self.active:
            return dict(status='WAIT_SELECTION', mask=None, box=None, dx=None, dy=None,
                        valid=False, inference_ms=0., predictor_called=False, target_mode=self.target_mode)
        if box is None and frame.shape[:2] != self.shape:
            self.reset()
            return dict(status='WAIT_SELECTION', mask=None, box=None, dx=None, dy=None,
                        valid=False, inference_ms=0., predictor_called=False,
                        transition_reason='resolution_changed_reselect', target_mode=self.target_mode)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # identical to source demo
        with torch.autocast('cuda', dtype=self.dtype):
            if box is not None:
                self.reset()
                self.shape = frame.shape[:2]
                self.predictor.load_first_frame(rgb)
                x1,y1,x2,y2 = map(float,box)
                if mask_prompt is not None:
                    _, ids, logits = self.predictor.add_new_mask(frame_idx=0, obj_id=1, mask=mask_prompt)
                elif x1 == x2 and y1 == y2:
                    _, ids, logits = self.predictor.add_new_prompt(frame_idx=0, obj_id=1,
                        points=np.array([[x1,y1]],np.float32), labels=np.array([1],np.int32))
                else:
                    _, ids, logits = self.predictor.add_new_prompt(frame_idx=0, obj_id=1,
                        bbox=np.array([[x1,y1],[x2,y2]],np.float32))
                self.active = True
                self.attempt += 1
            else:
                ids, logits = self.predictor.track(rgb)
        scores=logits[ids.index(1),0]
        self.last_mask_stability=float(((scores>.5).sum()/(scores>-.5).sum().clamp_min(1)).item())
        mask = (scores > 0).cpu().numpy()
        ys,xs = np.nonzero(mask)
        valid = bool(len(xs))
        bounds = [int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if valid else None
        self.state = 'TRACKING' if valid else 'SEARCHING'
        if before == 'SEARCHING' and valid and box is None:
            self.reacquired += 1
            self.attempt += 1
        reason = 'source_prompt' if box is not None else ('source_mask' if valid else 'source_empty_mask')
        h,w = frame.shape[:2]
        ms = (time.perf_counter()-start)*1000
        outputs = self.predictor.condition_state['output_dict']
        return dict(status=self.state, transition=f'{before}->{self.state}', transition_reason=reason,
            mask=mask if valid else None, box=bounds, valid=valid,
            dx=(bounds[0]+bounds[2])/2-w/2 if valid else None,
            dy=(bounds[1]+bounds[3])/2-h/2 if valid else None,
            control_gain=1. if valid else 0., target_mode=self.target_mode, engine='source_camera',
            inference_ms=ms, pipeline_ms=ms, sam_core_wall_ms=ms, predictor_called=not getattr(self,'warming',False),
            mask_stability=self.last_mask_stability, captured=stamp, reacquired=self.reacquired, attempt_id=self.attempt,
            confirmed_correct=self.correct, confirmed_wrong=self.wrong,
            gallery_size=sum(len(v) for v in outputs.values()),
            pinned_memory=len(outputs['cond_frame_outputs']), search_masks=[],
            sam_frame_idx=self.predictor.frame_idx, conditioning_count=len(outputs['cond_frame_outputs']),
            recent_memory_count=len(outputs['non_cond_frame_outputs']), recovery_policy='source_camera',
            target_pixels=[bounds[2]-bounds[0]+1,bounds[3]-bounds[1]+1] if valid else None,
            model_target_pixels=[(bounds[2]-bounds[0]+1)*self.predictor.image_size/w,
                                 (bounds[3]-bounds[1]+1)*self.predictor.image_size/h] if valid else None)

    def label(self, correct, stamp, attempt_id=0):
        # Human feedback remains logged; it does not invent an identity verifier.
        if correct:
            self.correct += 1
        else:
            self.wrong += 1
        return True

    def close(self):
        self.reset()
