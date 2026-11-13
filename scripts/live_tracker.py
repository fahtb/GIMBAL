"""Bounded, single-target SAMURAI streaming adapter (no video accumulation)."""
import sys
import copy
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from video_profile import timed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sam2'))
from sam2.build_sam import build_sam2


class LiveTracker:
    def __init__(self, checkpoint, memory=24, min_iou=0.5, max_gap=1.0, image_size=1024, cuda_graph=False, amp_dtype='fp16'):
        # Avoid CPU thread oversubscription between OpenCV capture and PyTorch.
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; check the selected .venv and NVIDIA driver')
        if amp_dtype=='bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 is unavailable on this device')
        self.amp_dtype=torch.bfloat16 if amp_dtype=='bf16' else torch.float16
        if amp_dtype=='bf16' and cuda_graph:
            print('BF16 experiment: existing FP16 CUDA graphs disabled, eager execution.',flush=True)
            cuda_graph=False
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
        self.policy = 'person'  # OBJECT policy is applied by ObjectTracker.
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
        self.loss_reason = ''
        self.last_box = None
        self.shape = None
        self.cached_features = None
        self.pending_memory = None
        self.pinned_quality = {}
        self.active_prompt_index = None
        self.durable_indices=set()
        self.model.kf_mean = self.model.kf_covariance = None
        self.model.stable_frames = self.model.frame_cnt = 0
        self.model.history = {}
        self.motion_reference_stamp = None

    @torch.inference_mode()
    @timed('sam')
    def step(self, frame, timestamp, box=None, continue_lost=False, mask_prompt=None, reuse_features=False,
             preserve_memory=False):
        start = time.perf_counter()
        measure=getattr(self,'measure_stages',False)
        stage_metrics={}
        if box is not None and not preserve_memory:
            self.reset()
        if self.shape is not None and self.shape != frame.shape:
            self.lost = True
            self.loss_reason = 'frame_shape_changed'
        self.shape = frame.shape
        dt = None if self.last_time is None else timestamp - self.last_time
        if dt is not None and (dt <= 0 or dt > self.max_gap):
            self.lost = True
            self.loss_reason = f'timestamp_gap:{dt:.6f}'
        if self.lost and not continue_lost:
            return dict(status='LOST - press R', mask=None, box=None, dx=None, dy=None,
                        inference_ms=0., predictor_called=False, sam_reason='latched:'+self.loss_reason,
                        memory=len(self.outputs['non_cond_frame_outputs']))
        if self.index == 0 and box is None and mask_prompt is None:
            raise ValueError('Select a box before tracking')
        motion_dt = timestamp-self.motion_reference_stamp if self.motion_reference_stamp is not None else dt
        self.model.kf.dt = max(motion_dt * 30., .01) if motion_dt is not None else 1.
        h, w = frame.shape[:2]
        size = self.model.image_size
        from sam_preprocess import prepare_rgb
        rgb = prepare_rgb(frame,size,getattr(self,'resize_mode','linear'),getattr(self,'jpeg_quality',0))
        image = torch.from_numpy(rgb).to('cuda').permute(2, 0, 1).float() / 255.
        image = ((image - self.mean) / self.std)[None]
        points = None
        if box is not None:
            single = box[0] == box[2] and box[1] == box[3]
            coords = torch.tensor(box[:2] if single else box, device='cuda', dtype=torch.float32).reshape(1, -1, 2)
            coords *= torch.tensor([size / w, size / h], device='cuda')
            points = {'point_coords': coords, 'point_labels': torch.tensor([[1]] if single else [[2, 3]], device='cuda')}
        if measure:
            torch.cuda.current_stream().synchronize()
            stage_metrics['preprocess_ms']=(time.perf_counter()-start)*1000
            gpu_start,gpu_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            gpu_start.record()
        with torch.autocast('cuda', dtype=self.amp_dtype):
            if reuse_features:
                feats, pos, sizes = self.cached_features
            else:
                backbone = self.backbone(image)
                _, feats, pos, sizes = self.model._prepare_backbone_features(backbone)
            if self.policy == 'sam_memory':
                self.cached_features = (feats, pos, sizes)
            prompt = None if mask_prompt is None else F.interpolate(
                torch.as_tensor(mask_prompt, device='cuda', dtype=torch.float32)[None, None],
                (size, size), mode='nearest')
            out = self.model.track_step(self.index, self.index == 0 or prompt is not None or box is not None, feats, pos, sizes,
                                        points, prompt, self.outputs, max(2, self.index + 1))
            if measure:
                gpu_end.record();gpu_end.synchronize()
                stage_metrics['gpu_core_ms']=gpu_start.elapsed_time(gpu_end)
                post_start=time.perf_counter()
            mask = (F.interpolate(out['pred_masks'].float(), (h, w), mode='bilinear',
                                  align_corners=False)[0, 0] > 0).cpu().numpy()
        score = float(out['best_iou_score'])
        obj = float(out['object_score_logits'])
        ys, xs = np.nonzero(mask)
        bbox = None if len(xs) < 16 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        valid = bbox is not None and score >= self.min_iou and obj > 0
        reasons = []
        if bbox is None:
            reasons.append('mask_area_below_16')
        if score < self.min_iou:
            reasons.append('mask_iou_below_min')
        if obj <= 0:
            reasons.append('object_logit_nonpositive')
        area_ratio = displacement = displacement_limit = None
        # Reject implausible changes before committing memory. Identity is never
        # recovered automatically after rejection: the user must confirm a box.
        if valid and self.last_box is not None and self.policy != 'sam_memory':
            a, b = np.array(self.last_box), np.array(bbox)
            area_ratio = np.prod(b[2:] - b[:2] + 1) / np.prod(a[2:] - a[:2] + 1)
            displacement = np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2)
            valid = .25 <= area_ratio <= 4 and displacement <= max(w, h) * (.08 + .5 * dt)
            displacement_limit = max(w, h) * (.08 + .5 * dt)
            if not .25 <= area_ratio <= 4:
                reasons.append('bbox_area_ratio_outside_0.25_4')
            if displacement > displacement_limit:
                reasons.append('bbox_displacement_exceeds_limit')
        if self.policy in ('object', 'sam_memory'):
            # Expose SAM observations to the class-agnostic hysteresis gate.
            # A single weak mask must not permanently latch the SAM adapter.
            valid = bbox is not None
        if self.policy == 'sam_memory':
            # Identity gate decides whether this observation may enter memory.
            self.pending_memory = (self.index, out, bbox, score, obj)
        if valid and self.policy != 'sam_memory':
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
        elif not valid:
            self.lost = self.policy != 'object'
            self.loss_reason = '|'.join(reasons)
            mask, bbox = None, None
        self.last_time = timestamp
        self.index += 1
        torch.cuda.current_stream().synchronize()
        if measure:
            stage_metrics.update(postprocess_ms=(time.perf_counter()-post_start)*1000,
                                 sam_core_wall_ms=(time.perf_counter()-start)*1000)
        return dict(status='TRACKING' if valid else 'LOST - press R', mask=mask, box=bbox,
                    **stage_metrics,
                    dx=None if bbox is None else (bbox[0] + bbox[2]) / 2 - w / 2,
                    dy=None if bbox is None else (bbox[1] + bbox[3]) / 2 - h / 2,
                    iou=score, object_logit=obj, predictor_called=True,
                    predicted_iou=score, object_score_logits=obj,
                    inference_ms=(time.perf_counter() - start) * 1000,
                    mask_area=int(len(xs)), sam_reason='|'.join(reasons) or 'accepted',
                    sam_threshold=self.min_iou, bbox_area_ratio=None if area_ratio is None else float(area_ratio),
                    displacement=None if displacement is None else float(displacement),
                    displacement_limit=displacement_limit,
                    memory=len(self.outputs['non_cond_frame_outputs']))

    def snapshot_motion(self):
        return {key: copy.deepcopy(getattr(self.model, key)) for key in
                ('kf_mean', 'kf_covariance', 'stable_frames', 'frame_cnt', 'history')}

    def restore_motion(self, snapshot):
        for key, value in snapshot.items():
            setattr(self.model, key, value)

    @torch.inference_mode()
    @timed('sam_anchor_probe')
    def probe_anchor(self, shape):
        """Same-frame unprompted SAM observation using ONLY the selected memory.

        Never commits memory or advances the stream; saves/restores Kalman state.
        No point/box/mask prompt is supplied to manufacture an identity match.
        """
        motion = self.snapshot_motion()
        try:
            self.reset_motion()
            feats, pos, sizes = self.cached_features
            bank = {'cond_frame_outputs': {self.anchor_index:
                    self.outputs['cond_frame_outputs'][self.anchor_index]}, 'non_cond_frame_outputs': {}}
            index = self.index-1
            with torch.autocast('cuda', dtype=self.amp_dtype):
                out = self.model.track_step(index, False, feats, pos, sizes,
                                           None, None, bank, max(2,index+1))
                mask = (F.interpolate(out['pred_masks'].float(), shape[:2], mode='bilinear',
                                      align_corners=False)[0,0] > 0).cpu().numpy()
            y,x = np.nonzero(mask)
            box = None if len(x)<16 else [int(x.min()),int(y.min()),int(x.max()),int(y.max())]
            return dict(mask=mask if box else None, box=box,
                        iou=float(out['best_iou_score']), object_logit=float(out['object_score_logits']))
        finally:
            self.restore_motion(motion)

    def snapshot_pending_memory(self):
        index,out,bbox,score,obj=self.pending_memory
        def clone(v):
            if isinstance(v,torch.Tensor):
                return v.detach().clone()
            if isinstance(v,(tuple,list)):
                return type(v)(clone(x) for x in v)
            return v
        compact={key:clone(out[key]) for key in ('maskmem_features','maskmem_pos_enc','obj_ptr')}
        compact.update(best_iou_score=torch.tensor(score),object_score_logits=torch.tensor(obj),
                       kf_score=None if out['kf_ious'] is None else torch.as_tensor(out['kf_ious']).detach().cpu().clone())
        return index,compact,bbox,score

    def commit_memory(self, pinned=4, pin=True, condition=False):
        """Keep prompt anchor + top-quality conditioning frames + bounded recency.

        Called only after the SAM-only identity/quality gate accepts an observation.
        No weak, absent or unverified candidate is committed.
        """
        index,compact,bbox,score=self.snapshot_pending_memory()
        self.store_verified_memory(index,compact,bbox,score,pinned,pin,condition)

    def store_verified_memory(self,index,compact,bbox,score,pinned=4,pin=True,condition=False):
        bank = self.outputs['non_cond_frame_outputs']
        cond = self.outputs['cond_frame_outputs']
        if not cond:
            cond[index] = compact
            self.anchor_index = index
            self.active_prompt_index = index
            self.durable_indices.add(index)
        else:
            bank[index] = compact
            if condition and (index in self.durable_indices or len(self.durable_indices)<pinned+1):
                self.active_prompt_index = index
                cond[index] = compact
                # Bounded number of permanent, verified refresh anchors.
                self.durable_indices.add(index)
            if pin:
                self.pinned_quality[index] = score
            best = sorted(self.pinned_quality, key=self.pinned_quality.get, reverse=True)[:pinned]
            for key in list(cond):
                if key not in self.durable_indices and key not in (self.anchor_index, self.active_prompt_index) and key not in best:
                    del cond[key]
            if index in best:
                cond[index] = compact
            self.pinned_quality = {key: self.pinned_quality[key] for key in best}
            while len(bank) > self.memory:
                del bank[min(bank)]
        self.last_box, self.lost = bbox, False

    def rebase_verified(self,frame,stamp,mask):
        """Rebase clean identity memory and all predictor counters together.

        Caller has already finished probation. Re-encode the verified SAM mask,
        not a flow/Kalman estimate; reuse this frame's encoder features.
        """
        anchors=[self.outputs['cond_frame_outputs'][i] for i in sorted(self.durable_indices)
                 if i in self.outputs['cond_frame_outputs']]
        if not anchors:
            anchors=[self.outputs['cond_frame_outputs'][self.anchor_index]]
        features=self.cached_features
        self.reset()
        self.outputs['cond_frame_outputs']=dict(enumerate(anchors))
        self.durable_indices=set(range(len(anchors)))
        self.anchor_index=0;self.active_prompt_index=0;self.index=len(anchors)
        self.cached_features=features
        result=self.step(frame,stamp,mask_prompt=mask,reuse_features=True,continue_lost=True)
        self.pending_memory=None  # New observation must pass quarantine again.
        return result

    @torch.inference_mode()
    @timed('sam_point_decoder')
    def decode_points(self, points, shape, labels=None):
        """Batch point prompts on the current encoder output; no second encoder."""
        feats, _, sizes = self.cached_features
        batch = len(points) if labels is None else 1
        maps = [f.permute(1, 2, 0).reshape(1, -1, *size) for f, size in zip(feats, sizes)]
        deepest = maps[-1]
        if self.model.directly_add_no_mem_embed:
            deepest = deepest + self.model.no_mem_embed.reshape(1, -1, 1, 1)
        coords = torch.as_tensor(points, dtype=torch.float32, device='cuda').reshape(batch, -1, 2)
        coords *= torch.tensor([self.model.image_size/shape[1], self.model.image_size/shape[0]], device='cuda')
        # Independent proposals must not update/use the active target's Kalman
        # state. Tracking continues with SAMURAI after this SAM2 decoder call.
        samurai = self.model.samurai_mode
        try:
            self.model.samurai_mode = False
            with torch.autocast('cuda', dtype=self.amp_dtype):
                outputs = self.model._forward_sam_heads(
                    deepest.expand(batch, -1, -1, -1),
                    point_inputs={'point_coords': coords, 'point_labels': (
                        torch.ones((batch, 1), device='cuda', dtype=torch.int64) if labels is None else
                        torch.as_tensor(labels,device='cuda',dtype=torch.int64).reshape(1,-1))},
                    high_res_features=[f.expand(batch, -1, -1, -1) for f in maps[:-1]], multimask_output=False)
        finally:
            self.model.samurai_mode = samurai
        # SAMURAI appends best-IoU/Kalman diagnostics to SAM2's seven head outputs.
        _, _, ious, low, _, _, logits = outputs[:7]
        masks = F.interpolate(low.float(), shape, mode='bilinear', align_corners=False)[:, 0]
        stability = ((masks > 1).sum((1, 2)) / (masks > -1).sum((1, 2)).clamp_min(1)).cpu().numpy()
        return (masks > 0).cpu().numpy(), ious.flatten().float().cpu().numpy(), logits.flatten().float().cpu().numpy(), stability

    def reset_motion(self):
        """Reset stale SAMURAI motion on re-entry without discarding identity memory."""
        self.model.kf_mean = self.model.kf_covariance = None
        self.model.stable_frames = self.model.frame_cnt = 0
        self.model.history = {}

    @torch.inference_mode()
    @timed('sam_tile_encoder')
    def encode_features(self, frame):
        """Encode a native search tile once without modifying temporal memory."""
        size = self.model.image_size
        from sam_preprocess import prepare_rgb
        rgb = prepare_rgb(frame,size,getattr(self,'resize_mode','linear'),getattr(self,'jpeg_quality',0))
        image = torch.from_numpy(rgb).to('cuda').permute(2,0,1).float()/255.
        image = ((image-self.mean)/self.std)[None]
        with torch.autocast('cuda', dtype=self.amp_dtype):
            _, feats, pos, sizes = self.model._prepare_backbone_features(self.backbone(image))
        self.cached_features = feats, pos, sizes
        return self.cached_features
