"""Person identity gate around SAMURAI; detectors never establish identity alone."""
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
import csv
import json
import os
import time
import uuid
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ReIDConfig:
    weights: str = 'sam2/checkpoints/yolov8n.pt'
    detector_confidence: float = .4
    detector_size: int = 640
    max_candidates: int = 20
    gallery_size: int = 20
    similarity: float = .88
    margin: float = .08
    consecutive_frames: int = 3
    confirmation_frames: int = 3
    candidate_iou: float = .3
    candidate_max_gap: float = .8
    timeout: float = 30.
    prediction_seconds: float = .7
    search_interval: float = .15
    gallery_interval: float = 1.
    gallery_iou: float = .8
    gallery_object_logit: float = 2.
    gallery_similarity: float = .92
    person_match_iou: float = .5
    other_person_iou: float = .1
    csv_path: str = 'runtime/reid_events.csv'

    @classmethod
    def load(cls, path):
        cfg = cls(**json.loads(Path(path).read_text(encoding='utf-8')))
        cfg.validate()
        return cfg

    def validate(self):
        for name in ('similarity', 'margin', 'candidate_iou', 'gallery_iou',
                     'gallery_similarity', 'person_match_iou', 'other_person_iou',
                     'detector_confidence'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        for name in ('timeout', 'search_interval', 'gallery_interval', 'candidate_max_gap'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('gallery_size', 'max_candidates', 'consecutive_frames', 'confirmation_frames', 'detector_size'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.gallery_size > 20 or not 0 <= self.prediction_seconds <= self.timeout:
            raise ValueError('gallery_size <= 20 and 0 <= prediction_seconds <= timeout required')


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(intersection / max(1., np.maximum(0, a[2:] - a[:2]).prod()
                                  + np.maximum(0, b[2:] - b[:2]).prod() - intersection))


class PersonEmbedder(Protocol):
    """Replace encode with OSNet later; return a normalized vector or None."""
    def encode(self, frame, box, mask=None): ...


class HSVRegionsEmbedder:
    """L2-normalized HSV distributions in head / torso / legs, no stored images."""
    def encode(self, frame, box, mask=None):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        if x2 - x1 < 8 or y2 - y1 < 24:
            return None
        crop = cv2.resize(frame[y1:y2, x1:x2], (64, 128))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Central region reduces background contamination for detector-only crops.
        valid = np.zeros((128, 64), np.uint8)
        valid[:, 12:52] = 255
        if mask is not None:
            valid &= cv2.resize(mask[y1:y2, x1:x2].astype('uint8') * 255,
                                (64, 128), interpolation=cv2.INTER_NEAREST)
        parts = []
        for start, end in ((0, 26), (26, 77), (77, 128)):
            m = valid[start:end]
            if np.count_nonzero(m) < 32:
                return None
            region = hsv[start:end]
            hs = cv2.calcHist([region], [0, 1], m, [16, 8], [0, 180, 0, 256]).ravel()
            v = cv2.calcHist([region], [2], m, [16], [0, 256]).ravel()
            # Square-root histograms tolerate modest illumination changes.
            feature = np.concatenate((np.sqrt(hs / max(hs.sum(), 1)),
                                      np.sqrt(v / max(v.sum(), 1))))
            parts.append(feature / max(np.linalg.norm(feature), 1e-8))
        result = np.concatenate(parts).astype(np.float32)
        return result / np.linalg.norm(result)


class YOLOPersonDetector:
    def __init__(self, config):
        (ROOT / 'runtime/ultralytics/Ultralytics').mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runtime/ultralytics'))
        from ultralytics import YOLO
        self.config = config
        weights = Path(config.weights)
        if not weights.is_absolute():
            weights = ROOT / weights
        if not weights.is_file():
            raise FileNotFoundError(f'YOLOv8 checkpoint missing: {weights}; see REID.md')
        self.model = YOLO(str(weights))
        self.detect(np.zeros((480, 640, 3), np.uint8))

    def detect(self, frame):
        result = self.model.predict(frame, classes=[0], conf=self.config.detector_confidence,
                                    imgsz=self.config.detector_size, max_det=self.config.max_candidates,
                                    device=0, quantize=16, verbose=False)[0]
        return result.boxes.xyxy.cpu().numpy().tolist()


class EventLog:
    fields = ('session', 'event', 'utc', 'source_time', 'lost_source_time',
              'reacquired_source_time', 'reacquisition_seconds', 'top1', 'top2', 'note')

    def __init__(self, path):
        self.path = Path(path)
        self.last_warning = 0.

    def write(self, **values):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            header = not self.path.exists() or self.path.stat().st_size == 0
            with self.path.open('a', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fields)
                if header:
                    writer.writeheader()
                writer.writerow(values)
        except OSError as exc:
            if time.monotonic() - self.last_warning > 5:
                print(f'ReID CSV unavailable: {exc}', flush=True)
                self.last_warning = time.monotonic()


class ReIDTracker:
    """All model operations run on the existing inference worker, never on the UI."""
    def __init__(self, tracker, detector, config, embedder=None):
        config.validate()
        self.tracker, self.detector, self.cfg = tracker, detector, config
        self.embedder = embedder or HSVRegionsEmbedder()
        path = Path(config.csv_path)
        self.log = EventLog(path if path.is_absolute() else ROOT / path)
        self.gallery = deque(maxlen=config.gallery_size)
        self.successful = self.suspected_wrong = 0
        self.confirmed_wrong = None  # Unknown without external identity labels.
        self.state = 'WAIT_SELECTION'
        self.session = ''
        self.lost_at = None
        self.last_box = self.last_stamp = None
        self.velocity = np.zeros(4)
        self.top1 = self.top2 = 0.
        self.pending = None
        self.hits = self.confirmations = 0
        self.last_scan = self.last_gallery = -float('inf')
        self.last_input = None

    def event(self, event, stamp, note=''):
        recovered = stamp if event == 'reacquired' else None
        self.log.write(session=self.session, event=event,
                       utc=datetime.now(timezone.utc).isoformat(), source_time=stamp,
                       lost_source_time=self.lost_at, reacquired_source_time=recovered,
                       reacquisition_seconds=None if recovered is None else stamp - self.lost_at,
                       top1=self.top1, top2=self.top2, note=note)

    def score(self, embedding):
        if embedding is None or not self.gallery:
            return 0.
        scores = sorted((float(np.dot(embedding, item)) for item in self.gallery), reverse=True)
        return float(np.mean(scores[:min(3, len(scores))]))

    def rank(self, frame):
        candidates = self.detector.detect(frame)[:self.cfg.max_candidates]
        ranked = [(self.score(self.embedder.encode(frame, candidate)), candidate)
                  for candidate in candidates]
        ranked.sort(key=lambda item: item[0], reverse=True)
        self.top1 = ranked[0][0] if ranked else 0.
        self.top2 = ranked[1][0] if len(ranked) > 1 else 0.
        return ranked

    def unambiguous(self):
        return self.top1 > self.cfg.similarity and self.top1 - self.top2 > self.cfg.margin

    def remember_position(self, box, stamp):
        box = np.array(box, dtype=float)
        if self.last_stamp is not None and stamp > self.last_stamp:
            measured = (box - self.last_box) / (stamp - self.last_stamp)
            self.velocity = .5 * self.velocity + .5 * measured
        self.last_box, self.last_stamp = box, stamp

    def prediction(self, stamp, shape):
        if self.last_box is None or stamp - self.last_stamp > self.cfg.prediction_seconds:
            return None
        box = self.last_box + self.velocity * max(0, stamp - self.last_stamp)
        box[[0, 2]] = np.clip(box[[0, 2]], 0, shape[1] - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, shape[0] - 1)
        return box.tolist() if box[2] > box[0] and box[3] > box[1] else None

    def lose(self, stamp):
        self.state, self.lost_at = 'LOST', stamp
        self.pending, self.hits = None, 0
        self.top1 = self.top2 = 0.
        self.last_scan = -float('inf')
        self.event('lost', stamp)

    def step(self, frame, stamp, box=None):
        start = time.perf_counter()
        if box is None and self.last_input is not None and stamp <= self.last_input:
            raise ValueError('ReID requires distinct, increasing frame timestamps')
        self.last_input = stamp
        result = dict(mask=None, box=None, dx=None, dy=None, inference_ms=0., memory=0)
        if box is not None:
            if self.lost_at is not None:
                self.event('manual_cancel', stamp)
            self.session = uuid.uuid4().hex[:12]
            self.gallery.clear()
            self.lost_at = self.last_box = self.last_stamp = None
            self.velocity[:] = 0
            self.pending, self.hits, self.confirmations = None, 0, 0
            self.top1 = self.top2 = 0.
            result = self.tracker.step(frame, stamp, box)
            self.state = 'TRACKING' if result['box'] is not None else 'WAIT_SELECTION'
            if self.state == 'TRACKING':
                # Use identical box-crop descriptors for SAM and detector boxes.
                # Mask-only histograms have a different background distribution.
                feature = self.embedder.encode(frame, result['box'])
                if feature is not None:
                    self.gallery.append(feature)
                self.remember_position(result['box'], stamp)
            self.last_gallery = stamp
            self.event('selected', stamp, 'gallery seeded' if self.gallery else 'no usable descriptor')
        elif self.state == 'TRACKING':
            result = self.tracker.step(frame, stamp)
            if result['box'] is None:
                self.lose(stamp)
            else:
                self.remember_position(result['box'], stamp)
                if stamp - self.last_gallery >= self.cfg.gallery_interval:
                    self.last_gallery = stamp
                    # Fresh detections on this exact frame; never trust stale overlap.
                    people = self.detector.detect(frame)
                    overlaps = [iou(result['box'], person) for person in people]
                    match = int(np.argmax(overlaps)) if overlaps else -1
                    isolated = match >= 0 and overlaps[match] >= self.cfg.person_match_iou
                    isolated &= all(value <= self.cfg.other_person_iou
                                    for index, value in enumerate(overlaps) if index != match)
                    feature = self.embedder.encode(frame, result['box'])
                    if (isolated and result.get('iou', 0) >= self.cfg.gallery_iou
                            and result.get('object_logit', 0) >= self.cfg.gallery_object_logit
                            and self.score(feature) >= self.cfg.gallery_similarity):
                        self.gallery.append(feature)
        elif self.state in ('LOST', 'SEARCHING', 'REACQUIRE'):
            if stamp - self.lost_at >= self.cfg.timeout:
                self.state = 'WAIT_SELECTION'
                self.event('timeout', stamp)
            elif not self.gallery:
                self.state = 'WAIT_SELECTION'
                self.event('no_descriptor', stamp)
            elif self.state == 'REACQUIRE':
                result = self.tracker.step(frame, stamp)
                feature = None if result['box'] is None else self.embedder.encode(frame, result['box'])
                ranked = self.rank(frame)
                if (result['box'] is None or self.score(feature) < self.cfg.similarity
                        or not ranked or not self.unambiguous()
                        or iou(result['box'], ranked[0][1]) < self.cfg.person_match_iou):
                    self.suspected_wrong += 1
                    self.event('reacquire_rejected', stamp, 'SAM/appearance verification failed; not ground truth')
                    self.state, self.pending, self.hits = 'SEARCHING', None, 0
                else:
                    self.confirmations += 1
                    if self.confirmations >= self.cfg.confirmation_frames:
                        self.state = 'TRACKING'
                        self.successful += 1
                        self.event('reacquired', stamp)
                        self.lost_at = None
                        self.last_box = self.last_stamp = None
                        self.velocity[:] = 0
                        self.remember_position(result['box'], stamp)
                        self.last_gallery = stamp
            else:
                self.state = 'SEARCHING'
                if stamp - self.last_scan >= self.cfg.search_interval:
                    previous_scan = self.last_scan
                    self.last_scan = stamp
                    ranked = self.rank(frame)
                    # Position associates consecutive detections; it cannot rescue ambiguous identity.
                    if ranked and self.unambiguous():
                        candidate = ranked[0][1]
                        consistent = (self.pending is not None and stamp - previous_scan <= self.cfg.candidate_max_gap
                                      and iou(self.pending, candidate) >= self.cfg.candidate_iou)
                        self.hits = self.hits + 1 if consistent else 1
                        self.pending = candidate
                        if self.hits >= self.cfg.consecutive_frames:
                            self.state = 'REACQUIRE'
                            self.confirmations = 0
                            self.event('reacquire_candidate', stamp)
                            # LiveTracker resets SAMURAI memory, prompt, and Kalman here.
                            result = self.tracker.step(frame, stamp, candidate)
                            feature = None if result['box'] is None else self.embedder.encode(frame, result['box'])
                            if self.score(feature) < self.cfg.similarity:
                                self.suspected_wrong += 1
                                self.event('reacquire_rejected', stamp, 'initial SAM mask disagrees')
                                self.state, self.pending, self.hits = 'SEARCHING', None, 0
                    else:
                        self.pending, self.hits = None, 0
        if self.state != 'TRACKING':
            result.update(mask=None, box=None, dx=None, dy=None)
        result.update(status=self.state, valid=self.state == 'TRACKING' and result['box'] is not None,
                      reid_top1=self.top1, reid_top2=self.top2, reid_hits=self.hits,
                      reacquired=self.successful, suspected_wrong=self.suspected_wrong,
                      confirmed_wrong=self.confirmed_wrong, gallery_size=len(self.gallery),
                      predicted_box=self.prediction(stamp, frame.shape) if self.state in ('LOST', 'SEARCHING') else None,
                      candidate_box=self.pending if self.state in ('SEARCHING', 'REACQUIRE') else None,
                      pipeline_ms=(time.perf_counter() - start) * 1000)
        return result
