"""Bounded, single-target SAMURAI streaming adapter (no video accumulation)."""
import sys
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sam2'))
from sam2.build_sam import build_sam2


class LiveTracker:
    def __init__(self, checkpoint, memory=24, min_iou=0.5, max_gap=1.0, image_size=1024, cuda_graph=False):
        # Avoid CPU thread oversubscription between OpenCV capture and PyTorch.
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; check the selected .venv and NVIDIA driver')
        if memory < 16:
            raise ValueError('memory must be >=16 to retain the SAMURAI pointer window')
        if image_size not in (512, 768, 1024):
            raise ValueError('image_size must be 512, 768 or 1024')
        # Same checkpoint and SAMURAI memory; change the actual encoder/token grid,
        # not merely the camera preview resolution. Prompt encoder follows config.
        self.model = build_sam2('configs/samurai/sam2.1_hiera_t.yaml', str(checkpoint), device='cuda',
                               hydra_overrides_extra=[f'++model.image_size={image_size}'])
        if sys.platform == 'win32' and not torch.backends.cuda.is_flash_attention_available():
            # This wheel cannot run FlashAttention. Let SDPA choose available
            # kernels directly instead of throwing once and then falling back.
            from sam2.modeling.sam import transformer
            transformer.ALLOW_ALL_KERNELS = True
        assert self.model.samurai_mode
        self.model.fast_mask_boxes = True
        self.model.cache_tracking_prompts = True
        self.cache_memory_scores = True
        self.memory, self.min_iou, self.max_gap = memory, min_iou, max_gap
        self.mean = torch.tensor([.485, .456, .406], device='cuda')[:, None, None]
        self.std = torch.tensor([.229, .224, .225], device='cuda')[:, None, None]
        self.reset()
        self.backbone = self.model.forward_image
        self.cuda_graph_enabled = False
        if cuda_graph:
            from cuda_backbone import CudaBackbone
            try:
                self.backbone = CudaBackbone(self.model)
                self.cuda_graph_enabled = True
                print('CUDA graph: image encoder enabled; SAMURAI tracking unchanged.', flush=True)
            except RuntimeError as exc:
                print(f'CUDA graph unavailable; using standard encoder: {exc}', flush=True)
            if self.cuda_graph_enabled and image_size == 512:
                from cuda_backbone import CudaMemoryAttention
                try:
                    self.model.memory_attention = CudaMemoryAttention(self.model.memory_attention, self.model)
                    print('CUDA graph: full memory attention enabled; other shapes use standard execution.', flush=True)
                except RuntimeError as exc:
                    print(f'Memory graph unavailable; using standard attention: {exc}', flush=True)
                from cuda_backbone import CudaMaskDecoder
                try:
                    decoder = CudaMaskDecoder(self.model)
                    self.model.sam_mask_decoder.forward = decoder
                    print('CUDA graph: tracking mask decoder enabled; box prompts use standard execution.', flush=True)
                except RuntimeError as exc:
                    print(f'Decoder graph unavailable; using standard decoder: {exc}', flush=True)

    def warmup(self):
        # Pay lazy CUDA/kernel startup costs before selecting a live target.
        self.step(np.zeros((480, 640, 3), dtype=np.uint8), 0., [160, 120, 480, 360])
        self.reset()

    def reset(self):
        self.outputs = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        self.index = 0
        self.last_time = None
        self.lost = False
        self.last_box = None
        self.shape = None
        self.model.kf_mean = self.model.kf_covariance = None
        self.model.stable_frames = self.model.frame_cnt = 0
        self.model.history = {}

    @torch.inference_mode()
    def step(self, frame, timestamp, box=None):
        start = time.perf_counter()
        if box is not None:
            self.reset()
        if self.shape is not None and self.shape != frame.shape:
            self.lost = True
        self.shape = frame.shape
        dt = None if self.last_time is None else timestamp - self.last_time
        if dt is not None and (dt <= 0 or dt > self.max_gap):
            self.lost = True
        if self.lost:
            return dict(status='LOST - press R', mask=None, box=None, dx=None, dy=None,
                        inference_ms=0., memory=len(self.outputs['non_cond_frame_outputs']))
        if self.index == 0 and box is None:
            raise ValueError('Select a box before tracking')
        self.model.kf.dt = max(dt * 30., .01) if dt is not None else 1.
        h, w = frame.shape[:2]
        size = self.model.image_size
        rgb = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb).to('cuda').permute(2, 0, 1).float() / 255.
        image = ((image - self.mean) / self.std)[None]
        points = None
        if box is not None:
            coords = torch.tensor(box, device='cuda', dtype=torch.float32).reshape(1, 2, 2)
            coords *= torch.tensor([size / w, size / h], device='cuda')
            points = {'point_coords': coords, 'point_labels': torch.tensor([[2, 3]], device='cuda')}
        with torch.autocast('cuda', dtype=torch.float16):
            backbone = self.backbone(image)
            _, feats, pos, sizes = self.model._prepare_backbone_features(backbone)
            out = self.model.track_step(self.index, self.index == 0, feats, pos, sizes,
                                        points, None, self.outputs, max(2, self.index + 1))
            mask = (F.interpolate(out['pred_masks'].float(), (h, w), mode='bilinear',
                                  align_corners=False)[0, 0] > 0).cpu().numpy()
        score = out['best_iou_score'].item()
        obj = out['object_score_logits'].item()
        ys, xs = np.nonzero(mask)
        bbox = None if len(xs) < 16 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        valid = bbox is not None and score >= self.min_iou and obj > 0
        # Reject implausible changes before committing memory. Identity is never
        # recovered automatically after rejection: the user must confirm a box.
        if valid and self.last_box is not None:
            a, b = np.array(self.last_box), np.array(bbox)
            area_ratio = np.prod(b[2:] - b[:2] + 1) / np.prod(a[2:] - a[:2] + 1)
            displacement = np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2)
            valid = .25 <= area_ratio <= 4 and displacement <= max(w, h) * (.08 + .5 * dt)
        if valid:
            # Only tensors actually consumed by memory attention are retained.
            compact = {key: out[key] for key in ('maskmem_features', 'maskmem_pos_enc',
                       'obj_ptr', 'best_iou_score', 'object_score_logits')}
            compact['kf_score'] = out['kf_ious']
            if self.cache_memory_scores:
                # These scalar scores are only inspected by Python .item() in
                # memory selection. Retain exact values on CPU once, instead of
                # synchronizing CUDA for every past score on every new frame.
                compact['best_iou_score'] = torch.tensor(score, dtype=out['best_iou_score'].dtype)
                compact['object_score_logits'] = torch.tensor(obj, dtype=out['object_score_logits'].dtype)
                if compact['kf_score'] is not None:
                    compact['kf_score'] = compact['kf_score'].detach().cpu()
            dest = 'cond_frame_outputs' if self.index == 0 else 'non_cond_frame_outputs'
            self.outputs[dest][self.index] = compact
            bank = self.outputs['non_cond_frame_outputs']
            while len(bank) > self.memory:
                del bank[min(bank)]
            self.last_box = bbox
        else:
            self.lost = True
            mask, bbox = None, None
        self.last_time = timestamp
        self.index += 1
        torch.cuda.synchronize()
        return dict(status='TRACKING' if valid else 'LOST - press R', mask=mask, box=bbox,
                    dx=None if bbox is None else (bbox[0] + bbox[2]) / 2 - w / 2,
                    dy=None if bbox is None else (bbox[1] + bbox[3]) / 2 - h / 2,
                    iou=score, object_logit=obj, inference_ms=(time.perf_counter() - start) * 1000,
                    memory=len(self.outputs['non_cond_frame_outputs']))
