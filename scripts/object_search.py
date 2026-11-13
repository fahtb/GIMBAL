"""Bounded asynchronous object search: COCO class or dense DINOv2 proposals."""
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from object_tracking import histogram
from person_reid import iou

ROOT = Path(__file__).resolve().parents[1]


class DinoSearch:
    def __init__(self, cfg, reference):
        import torch
        self.torch, self.cfg = torch, cfg
        hub = Path(cfg.dino_hub)
        if not hub.is_absolute():
            hub = ROOT / hub
        repo = hub / 'facebookresearch_dinov2_main'
        weights = hub / 'checkpoints/dinov2_vits14_pretrain.pth'
        if not repo.is_dir() or not weights.is_file():
            raise FileNotFoundError('DINOv2-small missing; run scripts/prepare_object_models.py')
        self.model = torch.hub.load(str(repo), 'dinov2_vits14', source='local', pretrained=False).eval().to(cfg.dino_device)
        self.model.load_state_dict(torch.load(weights, map_location=cfg.dino_device, weights_only=True))
        frame, box = reference
        x1,y1,x2,y2 = np.rint(box).astype(int)
        self.reference_cls, self.reference_dense = self.features(frame[y1:y2,x1:x2], 224)
        self.reference_dense = self.reference_dense.mean(0)
        self.reference_dense /= np.linalg.norm(self.reference_dense) + 1e-8

    def features(self, image, size):
        torch = self.torch
        rgb = cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).to(self.cfg.dino_device).permute(2,0,1).float()[None] / 255.
        mean = tensor.new_tensor([.485,.456,.406])[None,:,None,None]
        std = tensor.new_tensor([.229,.224,.225])[None,:,None,None]
        with torch.inference_mode():
            output = self.model.forward_features((tensor-mean)/std)
        cls = output['x_norm_clstoken'][0].cpu().numpy()
        dense = output['x_norm_patchtokens'][0].cpu().numpy()
        return cls/(np.linalg.norm(cls)+1e-8), dense/(np.linalg.norm(dense,axis=1,keepdims=True)+1e-8)

    def rank(self, frame, last_box, stop):
        _, dense = self.features(frame, self.cfg.dino_size)
        grid = self.cfg.dino_size // 14
        heat = (dense @ self.reference_dense).reshape(grid, grid)
        h,w = frame.shape[:2]
        bw,bh = last_box[2]-last_box[0], last_box[3]-last_box[1]
        proposals = []
        for _ in range(8):
            y,x = np.unravel_index(np.argmax(heat), heat.shape)
            if not np.isfinite(heat[y,x]):
                break
            cx,cy = (x+.5)*w/grid, (y+.5)*h/grid
            for factor in (.65, 1., 1.5):
                box = [max(0,round(cx-bw*factor/2)), max(0,round(cy-bh*factor/2)),
                       min(w,round(cx+bw*factor/2)), min(h,round(cy+bh*factor/2))]
                if box[2]-box[0] >= 4 and box[3]-box[1] >= 4:
                    proposals.append(box)
            rx,ry = max(1,round(bw/w*grid/2)), max(1,round(bh/h*grid/2))
            heat[max(0,y-ry):y+ry+1, max(0,x-rx):x+rx+1] = -np.inf
        ranked = []
        for box in proposals:
            if stop.is_set():
                break
            x1,y1,x2,y2 = box
            vector, _ = self.features(frame[y1:y2,x1:x2], 224)
            ranked.append((float(vector @ self.reference_cls), box))
        return ranked


class ObjectSearch:
    def __init__(self, detector, cfg, reference, class_id):
        self.detector, self.cfg, self.reference, self.class_id = detector, cfg, reference, class_id
        self.anchor = histogram(*reference)
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.lock = threading.Lock()
        self.pending = self.result = None
        self.error = None
        self.dino = None
        self.thread = threading.Thread(target=self.run, daemon=True, name='object-search')
        self.thread.start()

    def submit(self, frame, stamp, last_box):
        with self.lock:
            self.pending = frame, stamp, list(last_box)
        self.wake.set()

    def get(self):
        if self.error:
            raise RuntimeError(f'Object search failed: {self.error}')
        with self.lock:
            return self.result

    def run(self):
        try:
            while not self.stop.is_set():
                if not self.wake.wait(.05):
                    continue
                with self.lock:
                    job, self.pending = self.pending, None
                    self.wake.clear()
                if job is None:
                    continue
                frame, stamp, last_box = job
                tick = time.perf_counter()
                if self.class_id is not None:
                    found = self.detector.detect_all(frame, classes=[self.class_id], size=self.cfg.detector_size)
                    ranked = []
                    for detection in found:
                        feature = histogram(frame, detection['box'])
                        score = float(feature @ self.anchor) if feature is not None and self.anchor is not None else 0.
                        ranked.append((score, detection['box']))
                else:
                    if self.dino is None:
                        self.dino = DinoSearch(self.cfg, self.reference)
                    ranked = self.dino.rank(frame, last_box, self.stop)
                ranked.sort(key=lambda row: row[0], reverse=True)
                # Different windows around the same object aren't distractors.
                distinct = []
                for candidate in ranked:
                    if all(iou(candidate[1], other[1]) < .3 for other in distinct):
                        distinct.append(candidate)
                top1 = distinct[0][0] if distinct else 0.
                top2 = distinct[1][0] if len(distinct) > 1 else 0.
                reason = ('no_candidates' if not distinct else 'below_similarity' if top1 < self.cfg.search_similarity else
                          'insufficient_margin' if top1-top2 < self.cfg.search_margin else 'qualified')
                result = dict(frame=frame, stamp=stamp, box=distinct[0][1] if reason == 'qualified' else None,
                              top1=top1, top2=top2, reason=reason, search_ms=(time.perf_counter()-tick)*1000)
                with self.lock:
                    self.result = result
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        self.stop.set()
        self.wake.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('Object search did not stop within 10 seconds')
