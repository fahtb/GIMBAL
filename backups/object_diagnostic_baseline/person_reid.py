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
import threading
import cv2
import numpy as np
from reid_features import VisibleHSV, IdentityGallery, HSVFallbackBackend
from reid_diagnostics import SearchLog

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ReIDConfig:
    weights: str = 'sam2/checkpoints/yolov8n.pt'
    detector_confidence: float = .4
    detector_size: int = 320
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
    search_csv: str = 'runtime/reid_search_frames.csv'
    high_similarity: float = .96
    high_frames: int = 1
    score_top_k: int = 1
    duplicate_similarity: float = .985
    single_candidate_decay: float = 0.
    minimum_similarity: float = .88
    decay_after: float = 5.
    provisional_gain: float = .25
    edge_fraction: float = .015
    minimum_visible_fraction: float = .35

    @classmethod
    def load(cls, path):
        cfg = cls(**json.loads(Path(path).read_text(encoding='utf-8')))
        cfg.validate()
        return cfg

    def validate(self):
        for name in ('similarity', 'margin', 'candidate_iou', 'gallery_iou',
                     'gallery_similarity', 'person_match_iou', 'other_person_iou',
                     'detector_confidence', 'duplicate_similarity'):
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
        if not 0 <= self.minimum_similarity <= self.similarity <= self.high_similarity <= 1:
            raise ValueError('require 0 <= minimum_similarity <= similarity <= high_similarity <= 1')
        if not 1 <= self.high_frames <= 2 or not 1 <= self.score_top_k <= self.gallery_size:
            raise ValueError('high_frames must be 1..2; score_top_k must fit gallery')
        if not 0 <= self.single_candidate_decay <= .01 or not 0 <= self.provisional_gain <= 1:
            raise ValueError('invalid decay rate or provisional gain')
        if not 0 < self.edge_fraction < .1 or not .35 <= self.minimum_visible_fraction <= 1:
            raise ValueError('invalid visibility configuration')
        if not np.isfinite(self.decay_after) or self.decay_after < 0:
            raise ValueError('decay_after must be finite and nonnegative')


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
        self.lock = threading.Lock()
        self.detect(np.zeros((480, 640, 3), np.uint8))

    def detect(self, frame):
        with self.lock:
            tick = time.perf_counter()
            result = self.model.predict(frame, classes=[0], conf=self.config.detector_confidence,
                                        imgsz=self.config.detector_size, max_det=self.config.max_candidates,
                                        device=0, quantize=16, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            self.last_ms = (time.perf_counter()-tick)*1000
            return boxes


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
    """Identity state machine. step always consumes detections from its exact frame."""
    def __init__(self, tracker, detector, config, embedder=None):
        config.validate()
        self.tracker, self.detector, self.cfg = tracker, detector, config
        hsv = VisibleHSV(config.edge_fraction, config.minimum_visible_fraction)
        self.embedder = HSVFallbackBackend(embedder, hsv) if embedder is not None else hsv
        self.gallery = IdentityGallery(self.embedder, config.gallery_size, config.score_top_k,
                                       config.duplicate_similarity)
        self.log = EventLog(ROOT / config.csv_path)
        self.search_log = SearchLog(ROOT / config.search_csv)
        self.session = ''
        self.state = 'WAIT_SELECTION'
        self.successful = self.suspected_wrong = 0
        self.confirmed_correct = self.confirmed_wrong = 0
        self.attempt_id = 0
        self.attempt_counter = 0
        self.last_labeled_attempt = 0
        self.current_label = None
        self.lost_at = self.first_candidate = self.accepted_at = None
        self.last_stamp = self.last_box = self.last_input = None
        self.velocity = np.zeros(4)
        self.pending = self.flow = None
        self.hits = self.confirmations = self.frame_index = 0
        self.top1 = self.top2 = 0.
        self.region = 'none'
        self.reason = 'waiting_selection'
        self.last_scan = self.last_gallery = -float('inf')
        self.detector_ms = 0.
        self.candidate_count = 0
        self.scan_data = None
        self.threshold = config.similarity
        self.required_streak = config.consecutive_frames

    def event(self, name, stamp, note=''):
        recovered = stamp if name == 'reacquired' else None
        self.log.write(session=self.session, event=name, utc=datetime.now(timezone.utc).isoformat(),
                       source_time=stamp, lost_source_time=self.lost_at,
                       reacquired_source_time=recovered,
                       reacquisition_seconds=stamp-self.lost_at if recovered is not None and self.lost_at is not None else None,
                       top1=self.top1, top2=self.top2,
                       note=f'attempt={self.attempt_id}; gallery={len(self.gallery)}; {note}')

    def label(self, correct, stamp, attempt_id=None):
        # UI supplies the displayed attempt ID: never label a newer unseen result.
        if not self.attempt_id or (attempt_id is not None and attempt_id != self.attempt_id):
            return False
        if self.last_labeled_attempt == self.attempt_id:
            if self.current_label == correct:
                return False
            if self.current_label:
                self.confirmed_correct -= 1
            else:
                self.confirmed_wrong -= 1
        self.last_labeled_attempt, self.current_label = self.attempt_id, correct
        if correct:
            self.confirmed_correct += 1
        else:
            self.confirmed_wrong += 1
            # Drop potentially contaminated recent samples; keep immutable anchor.
            self.gallery.rolling.clear()
            if self.state in ('TRACKING', 'REACQUIRE'):
                self.lose(stamp)
        self.event('label_correct' if correct else 'label_wrong', stamp, 'human W/C label')
        return True

    def score(self, feature):
        return self.gallery.score(feature)[0]

    def rank(self, frame):
        if self.scan_data is None:
            tick = time.perf_counter()
            boxes = self.detector.detect(frame)
            self.detector_ms = getattr(self.detector, 'last_ms', (time.perf_counter()-tick)*1000)
        else:
            boxes, self.detector_ms = self.scan_data
        self.diagnostic_people = boxes
        ranked = []
        for box in boxes[:self.cfg.max_candidates]:
            feature = self.embedder.encode(frame, box)
            value, region = self.gallery.score(feature)
            ranked.append((value, box, region))
        ranked.sort(key=lambda row: row[0], reverse=True)
        self.candidate_count = len(ranked)
        self.top1 = ranked[0][0] if ranked else 0.
        self.top2 = ranked[1][0] if len(ranked) > 1 else 0.
        self.region = ranked[0][2] if ranked else 'none'
        if ranked and self.first_candidate is None and self.lost_at is not None:
            self.first_candidate = self.last_input
            self.event('first_candidate', self.last_input, 'first detected person, not identity ground truth')
        return ranked

    def gate(self, stamp):
        elapsed = max(0., stamp-(self.lost_at if self.lost_at is not None else stamp)-self.cfg.decay_after)
        # Default decay=0: keep the measured baseline T_mid unchanged.
        self.threshold = max(self.cfg.minimum_similarity, self.cfg.similarity-
                             (elapsed*self.cfg.single_candidate_decay if self.candidate_count == 1 else 0.))
        self.required_streak = self.cfg.high_frames if self.top1 >= self.cfg.high_similarity else self.cfg.consecutive_frames
        if not self.candidate_count:
            return 'no_candidates'
        if self.top1 < self.threshold:
            return 'below_threshold'
        if self.top1-self.top2 < self.cfg.margin:
            return 'insufficient_margin'
        return 'qualified'

    def remember_position(self, box, stamp):
        current = np.asarray(box, float)
        if self.last_stamp is not None and stamp > self.last_stamp:
            self.velocity = .5*self.velocity + .5*(current-self.last_box)/(stamp-self.last_stamp)
        self.last_box, self.last_stamp = current, stamp

    def prediction(self, stamp, shape):
        if self.last_box is None or stamp-self.last_stamp > self.cfg.prediction_seconds:
            return None
        box = self.last_box + self.velocity*max(0, stamp-self.last_stamp)
        box[[0,2]] = np.clip(box[[0,2]], 0, shape[1]-1)
        box[[1,3]] = np.clip(box[[1,3]], 0, shape[0]-1)
        return box.tolist()

    def lose(self, stamp):
        self.state, self.lost_at = 'LOST', stamp
        self.first_candidate = self.accepted_at = None
        self.pending = self.flow = None
        self.hits = self.confirmations = 0
        self.top1 = self.top2 = 0.
        self.reason = 'lost'
        self.candidate_count = 0
        self.detector_ms = 0.
        self.event('lost', stamp)

    def init_flow(self, frame, result, stamp):
        from live_motion import FlowState
        mask = result.get('mask')
        if mask is None:
            mask = np.zeros(frame.shape[:2], bool)
            x1,y1,x2,y2 = map(int, result['box'])
            mask[y1:y2, x1:x2] = True
        self.flow = FlowState(frame, mask, result['box'], self.frame_index, stamp)

    def step(self, frame, stamp, box=None, detections=None):
        tick = time.perf_counter()
        if box is None and self.last_input is not None and stamp <= self.last_input:
            raise ValueError('ReID requires distinct, increasing frame timestamps')
        self.last_input = stamp
        self.frame_index += 1
        self.scan_data = detections
        self.diagnostic_people = None
        before = self.state
        self.reason = 'tracking'
        result = dict(mask=None, box=None, dx=None, dy=None, inference_ms=0., memory=0)
        if box is not None:
            if self.lost_at is not None:
                self.event('manual_cancel', stamp)
            self.session = uuid.uuid4().hex[:12]
            self.attempt_id = 0
            self.gallery.seed(None)
            self.flow = self.last_box = self.last_stamp = None
            self.lost_at = self.first_candidate = self.accepted_at = None
            self.pending, self.hits = None, 0
            self.velocity[:] = 0
            self.top1 = self.top2 = 0.
            result = self.tracker.step(frame, stamp, box)
            self.state = 'TRACKING' if result['box'] is not None else 'WAIT_SELECTION'
            if self.state == 'TRACKING':
                self.gallery.seed(self.embedder.encode(frame, result['box']))
                self.remember_position(result['box'], stamp)
                self.init_flow(frame, result, stamp)
            self.last_gallery = stamp
            self.event('selected', stamp, 'explicit manual selection: new immutable anchor')
        elif self.state == 'TRACKING':
            result = self.tracker.step(frame, stamp)
            if result['box'] is None:
                self.lose(stamp)
            else:
                self.remember_position(result['box'], stamp)
                if stamp-self.last_gallery >= self.cfg.gallery_interval:
                    self.last_gallery = stamp
                    people = self.detector.detect(frame)
                    self.diagnostic_people = people
                    overlaps = [iou(result['box'], person) for person in people]
                    match = int(np.argmax(overlaps)) if overlaps else -1
                    isolated = match >= 0 and overlaps[match] >= self.cfg.person_match_iou
                    isolated &= all(value <= self.cfg.other_person_iou for index,value in enumerate(overlaps) if index != match)
                    feature = self.embedder.encode(frame, result['box'])
                    if (isolated and result.get('iou',0) >= self.cfg.gallery_iou
                            and result.get('object_logit',0) >= self.cfg.gallery_object_logit
                            and self.score(feature) >= self.cfg.gallery_similarity):
                        if self.gallery.add(feature):
                            self.event('gallery_added', stamp)
        elif self.state in ('LOST', 'SEARCHING', 'REACQUIRE'):
            if stamp-self.lost_at >= self.cfg.timeout:
                self.state, self.reason = 'WAIT_SELECTION', 'timeout'
                self.event('timeout', stamp)
            elif not len(self.gallery):
                self.state, self.reason = 'WAIT_SELECTION', 'empty_gallery'
            elif self.state == 'REACQUIRE':
                # Optical flow is motion evidence during probation, never identity.
                from live_motion import gray_frame
                flow_ok = self.flow is not None and self.flow.advance(self.frame_index, stamp, gray_frame(frame))
                result = self.tracker.step(frame, stamp)
                ranked = self.rank(frame)
                self.reason = self.gate(stamp)
                feature = self.embedder.encode(frame,result['box']) if result['box'] is not None else None
                evidence = (result['box'] is not None and ranked and self.reason == 'qualified'
                            and self.score(feature) >= self.threshold
                            and iou(result['box'],ranked[0][1]) >= self.cfg.person_match_iou)
                if flow_ok and result['box'] is not None:
                    evidence = evidence and iou(self.flow.box(),result['box']) >= self.cfg.candidate_iou
                if not evidence:
                    self.reason = 'verification_failed:'+self.reason
                    self.suspected_wrong += 1
                    self.event('reacquire_rejected', stamp, self.reason)
                    self.state, self.pending, self.hits, self.flow = 'SEARCHING', None, 0, None
                else:
                    self.confirmations += 1
                    self.reason = 'sam_verification'
                    if self.confirmations >= self.cfg.confirmation_frames:
                        self.successful += 1
                        self.state, self.reason = 'TRACKING', 'confirmed'
                        self.event('reacquired', stamp)
                        self.lost_at = None
                        self.last_box = self.last_stamp = None
                        self.velocity[:] = 0
                        self.remember_position(result['box'],stamp)
                        self.last_gallery = stamp
            else:
                self.state = 'SEARCHING'
                ranked = self.rank(frame)  # every consumed search frame, no interval throttle
                previous_scan, self.last_scan = self.last_scan, stamp
                self.reason = self.gate(stamp)
                if self.reason == 'qualified':
                    candidate = ranked[0][1]
                    consistent = (self.pending is not None and stamp-previous_scan <= self.cfg.candidate_max_gap
                                  and iou(self.pending,candidate) >= self.cfg.candidate_iou)
                    self.hits = self.hits+1 if consistent else 1
                    self.pending = candidate
                    if self.hits >= self.required_streak:
                        self.attempt_counter += 1
                        self.attempt_id = self.attempt_counter
                        self.accepted_at = stamp
                        self.event('provisional',stamp)
                        # Same source frame and box initialize SAMURAI memory + LK.
                        result = self.tracker.step(frame,stamp,candidate)
                        if result['box'] is not None and self.score(self.embedder.encode(frame,result['box'])) >= self.threshold:
                            self.state, self.confirmations = 'REACQUIRE', 0
                            self.init_flow(frame,result,stamp)
                            self.reason = 'provisional'
                        else:
                            self.reason = 'sam_initialization_failed'
                            self.suspected_wrong += 1
                            self.event('reacquire_rejected',stamp,self.reason)
                            self.pending, self.hits = None, 0
                    else:
                        self.reason = 'insufficient_streak'
                else:
                    self.pending, self.hits = None, 0
        provisional = self.state == 'REACQUIRE' and result['box'] is not None
        valid = self.state in ('TRACKING','REACQUIRE') and result['box'] is not None
        if not valid:
            result.update(mask=None,box=None,dx=None,dy=None)
        if before in ('LOST','SEARCHING','REACQUIRE') or self.state in ('LOST','SEARCHING','REACQUIRE'):
            self.search_log.write(session=self.session, frame_time=stamp,state=self.state,
                candidates=self.candidate_count,top1=self.top1,top2=self.top2,
                threshold=self.threshold,high_threshold=self.cfg.high_similarity,
                required_margin=self.cfg.margin,actual_margin=self.top1-self.top2,
                streak=self.hits,required_streak=self.required_streak,reason=self.reason,
                detector_ms=self.detector_ms,new_detection=before in ('LOST','SEARCHING','REACQUIRE'),
                lost_time=self.lost_at,first_candidate_time=self.first_candidate,
                accepted_time=self.accepted_at,region=self.region,gallery_size=len(self.gallery))
        result.update(status=self.state, valid=valid, provisional=provisional,
            transition=f'{before}->{self.state}',
            transition_reason=result.get('sam_reason', self.reason) if self.state == 'LOST' else self.reason,
            yolo_person=self.diagnostic_people,
            thresholds=dict(sam_iou=self.tracker.min_iou if hasattr(self.tracker, 'min_iou') else None,
                            similarity=self.threshold, margin=self.cfg.margin,
                            person_match_iou=self.cfg.person_match_iou, consecutive=self.required_streak),
            control_gain=self.cfg.provisional_gain if provisional else 1. if valid else 0.,
            reid_top1=self.top1,reid_top2=self.top2,reid_hits=self.hits,reid_reason=self.reason,
            reid_region=self.region,reid_threshold=self.threshold,attempt_id=self.attempt_id,
            reacquired=self.successful,suspected_wrong=self.suspected_wrong,
            confirmed_wrong=self.confirmed_wrong,confirmed_correct=self.confirmed_correct,
            gallery_size=len(self.gallery),gallery_anchor=int(self.gallery.anchor is not None),
            gallery_rolling=len(self.gallery.rolling), session=self.session,
            predicted_box=self.prediction(stamp,frame.shape) if self.state in ('LOST','SEARCHING') else None,
            candidate_box=self.pending if self.state == 'SEARCHING' else None,
            flow_valid=bool(provisional and self.flow and self.flow.valid),detector_ms=self.detector_ms,
            pipeline_ms=(time.perf_counter()-tick)*1000)
        return result


