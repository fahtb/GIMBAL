"""Class-agnostic SAM gating, crop transforms and image-motion compensation."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time

import cv2
import numpy as np

from person_reid import iou


@dataclass
class ObjectConfig:
    low_score: float = .35
    high_score: float = .55
    bad_frames: int = 3
    min_area: int = 16
    area_collapse: float = .12
    histogram_low: float = .30
    histogram_high: float = .45
    small_side: int = 48
    small_encoder_side: int = 48
    crop_scale: float = 4.
    crop_min_size: int = 128
    motion_width: int = 960
    class_match_iou: float = .3
    detector_size: int = 640
    search_similarity: float = .72
    search_margin: float = .06
    search_frames: int = 3
    search_max_gap: float = .8
    search_candidate_iou: float = .25
    search_max_age: float = .5
    dino_size: int = 518
    dino_device: str = 'cuda'
    dino_hub: str = 'runtime/torch_hub'

    @classmethod
    def load(cls, path):
        cfg = cls(**json.loads(Path(path).read_text(encoding='utf-8')))
        for name in ('low_score', 'high_score', 'area_collapse', 'histogram_low', 'histogram_high',
                     'class_match_iou', 'search_similarity', 'search_margin', 'search_candidate_iou'):
            if not 0 <= getattr(cfg, name) <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        if cfg.high_score <= cfg.low_score or cfg.histogram_high <= cfg.histogram_low:
            raise ValueError('hysteresis high thresholds must exceed low thresholds')
        for name in ('bad_frames', 'search_frames', 'min_area', 'small_side', 'small_encoder_side',
                     'crop_min_size', 'motion_width', 'detector_size', 'dino_size'):
            if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('crop_scale', 'search_max_gap', 'search_max_age'):
            if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if cfg.dino_size % 14 or cfg.dino_device not in ('cpu', 'cuda'):
            raise ValueError('DINO size must be divisible by 14; device must be cpu/cuda')
        return cfg


def histogram(frame, box):
    """Whole-object color descriptor; no body parts or person aspect checks."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = np.rint(box).astype(int)
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (64, 64)), cv2.COLOR_BGR2HSV)
    hs = cv2.calcHist([patch], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
    value = cv2.calcHist([patch], [2], None, [32], [0, 256]).ravel()
    descriptor = np.r_[np.sqrt(hs / max(1., hs.sum())), np.sqrt(value / max(1., value.sum()))]
    return descriptor / max(1e-8, np.linalg.norm(descriptor))


class ObjectGate:
    def __init__(self, cfg):
        self.cfg = cfg
        self.low_scores = self.low_appearance = 0
        self.area = None

    def check(self, result, similarity):
        c = self.cfg
        score, area = result.get('iou', 0.), result.get('mask_area', 0)
        self.low_scores = self.low_scores + 1 if score < c.low_score else 0 if score >= c.high_score else self.low_scores
        self.low_appearance = (self.low_appearance + 1 if similarity < c.histogram_low else
                               0 if similarity >= c.histogram_high else self.low_appearance)
        reason = None
        if self.area is not None and area < max(c.min_area, self.area * c.area_collapse):
            reason = 'mask_area_collapse'
        elif self.low_scores >= c.bad_frames:
            reason = 'mask_score_low_streak'
        elif self.low_appearance >= c.bad_frames:
            reason = 'anchor_histogram_low_streak'
        if area >= c.min_area and reason is None and self.low_scores == 0:
            self.area = area if self.area is None else .8 * self.area + .2 * area
        return reason


class ImageMotion:
    """Robust background affine plus local LK translation; never an identity gate."""
    def __init__(self, width):
        self.width, self.previous = width, None

    def advance(self, frame, box):
        h, w = frame.shape[:2]
        scale = min(1., self.width / w)
        gray = cv2.cvtColor(cv2.resize(frame, (round(w * scale), round(h * scale))), cv2.COLOR_BGR2GRAY)
        old, self.previous = self.previous, gray
        if old is None or old.shape != gray.shape or box is None:
            return box, 'motion_initialization'
        x1, y1, x2, y2 = np.rint(np.array(box) * scale).astype(int)
        mask = np.full(gray.shape, 255, np.uint8)
        mask[max(0,y1-5):min(gray.shape[0],y2+5), max(0,x1-5):min(gray.shape[1],x2+5)] = 0
        affine = None
        reasons = []
        for local in (False, True):
            region = cv2.bitwise_not(mask) if local else mask
            points = cv2.goodFeaturesToTrack(old, 100 if local else 250, .01, 5, mask=region)
            if points is None or len(points) < 8:
                continue
            moved, ok, err = cv2.calcOpticalFlowPyrLK(old, gray, points, None, winSize=(25,25), maxLevel=3)
            if moved is None:
                continue
            back, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, old, moved, None, winSize=(25,25), maxLevel=3)
            if back is None:
                continue
            valid = (ok.ravel() > 0) & (back_ok.ravel() > 0) & (np.linalg.norm(points-back, axis=2).ravel() < 1.5)
            valid &= np.isfinite(moved).all(axis=(1,2))
            if valid.sum() < 8:
                continue
            if local:
                delta = np.median((moved-points)[valid,0], axis=0) / scale
                reasons.append('target_lk_prediction')
                if affine is not None:
                    # Apply background scale/rotation, then the target's motion
                    # relative to that background (median foreground residual).
                    corners = np.array([[box[0],box[1],1], [box[2],box[1],1],
                                        [box[2],box[3],1], [box[0],box[3],1]], float)
                    corners[:,:2] *= scale
                    transformed = corners @ affine.T / scale
                    center = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])
                    bg_delta = np.r_[center*scale,1] @ affine.T / scale - center
                    transformed += delta - bg_delta
                    return np.r_[transformed.min(0), transformed.max(0)].tolist(), '|'.join(reasons)
                return (np.array(box) + np.tile(delta, 2)).tolist(), '|'.join(reasons)
            affine, inliers = cv2.estimateAffinePartial2D(points[valid], moved[valid], method=cv2.RANSAC)
            if affine is not None and inliers.sum() >= 12:
                reasons.append('background_affine_compensation')
            else:
                affine = None
        if affine is None:
            return box, 'flow_unreliable_use_sam'
        corners = np.array([[box[0],box[1],1], [box[2],box[1],1],
                            [box[2],box[3],1], [box[0],box[3],1]], float)
        corners[:,:2] *= scale
        transformed = corners @ affine.T / scale
        return np.r_[transformed.min(0), transformed.max(0)].tolist(), '|'.join(reasons)


class CropSAM:
    """Hold a stable crop coordinate system; explicitly re-prompt on recenter."""
    def __init__(self, sam, cfg):
        self.sam, self.cfg, self.region = sam, cfg, None

    def small(self, box, frame):
        side = min(box[2]-box[0], box[3]-box[1])
        return side < self.cfg.small_side or side * self.sam.model.image_size / max(frame.shape[:2]) < self.cfg.small_encoder_side

    def region_for(self, box, frame):
        h, w = frame.shape[:2]
        side = max(self.cfg.crop_min_size, max(box[2]-box[0], box[3]-box[1]) * self.cfg.crop_scale)
        cw, ch = min(w, round(side)), min(h, round(side))
        x = round(np.clip((box[0]+box[2]-cw)/2, 0, w-cw))
        y = round(np.clip((box[1]+box[3]-ch)/2, 0, h-ch))
        return [x, y, x+cw, y+ch]

    def step(self, frame, stamp, prompt, predicted):
        reason = ''
        if prompt is not None:
            self.region = self.region_for(prompt, frame) if self.small(prompt, frame) else None
            if self.region:
                reason = 'small_target_crop_init'
        elif predicted is not None:
            if self.region is None and self.small(predicted, frame):
                self.region, prompt = self.region_for(predicted, frame), predicted
                reason = 'small_target_crop_enter'
            elif self.region is not None:
                x1,y1,x2,y2 = self.region
                margin = min(x2-x1, y2-y1) * .15
                desired = self.region_for(predicted, frame)
                ratio = max(desired[2]-desired[0], desired[3]-desired[1]) / max(x2-x1,y2-y1)
                if ratio < .67 or ratio > 1.5:
                    self.region, prompt = desired, predicted
                    reason = 'crop_rescale_size_change'
                elif (predicted[0] < x1+margin or predicted[1] < y1+margin or
                        predicted[2] > x2-margin or predicted[3] > y2-margin):
                    self.region, prompt = self.region_for(predicted, frame), predicted
                    reason = 'crop_recenter_motion_prediction'
        if self.region is None:
            return self.sam.step(frame, stamp, prompt), reason
        x1,y1,x2,y2 = self.region
        offset = np.array([x1,y1,x1,y1])
        local = None if prompt is None else (np.asarray(prompt)-offset).tolist()
        result = self.sam.step(frame[y1:y2,x1:x2], stamp, local)
        if result.get('box') is not None:
            result['box'] = (np.array(result['box'])+offset).astype(int).tolist()
            mask = np.zeros(frame.shape[:2], bool)
            mask[y1:y2,x1:x2] = result['mask']
            result['mask'] = mask
            b = result['box']
            result.update(dx=(b[0]+b[2]-frame.shape[1])/2, dy=(b[1]+b[3]-frame.shape[0])/2)
        result['crop_box'] = list(self.region)
        return result, reason


class ObjectTracker:
    def __init__(self, sam, detector, cfg):
        self.sam, self.detector, self.cfg = sam, detector, cfg
        self.state = 'WAIT_SELECTION'
        self.search = None
        self.reacquired = 0
        self.last_box = self.anchor = None
        self.last_search_stamp = -float('inf')
        self.candidate = None
        self.hits = 0
        self.class_id = None
        self.result = None

    def initialize(self, frame, stamp, box, class_id=None):
        self.close()
        self.sam.policy = 'object'
        self.class_id = class_id
        self.anchor = histogram(frame, box)
        self.reference = frame.copy(), list(box)
        self.gate = ObjectGate(self.cfg)
        self.motion = ImageMotion(self.cfg.motion_width)
        self.crop = CropSAM(self.sam, self.cfg)
        self.last_box, self.hits, self.candidate = list(box), 0, None
        self.last_search_stamp = -float('inf')
        self.state = 'WAIT_SELECTION'
        return self.track(frame, stamp, box)

    def decorate(self, result, before, reason, tick):
        valid = self.state == 'TRACKING' and result.get('box') is not None
        result.update(status=self.state, valid=valid, target_mode='OBJECT', transition=f'{before}->{self.state}',
                      transition_reason=reason, thresholds=asdict(self.cfg), reacquired=self.reacquired,
                      coco_class=self.class_id, pipeline_ms=(time.perf_counter()-tick)*1000,
                      control_gain=1. if valid else 0.)
        if not valid:
            result.update(box=None, mask=None, dx=None, dy=None)
        self.result = result
        return result

    def track(self, frame, stamp, prompt=None):
        tick, before = time.perf_counter(), self.state
        predicted, motion_reason = self.motion.advance(frame, self.last_box)
        result, correction = self.crop.step(frame, stamp, prompt, predicted)
        feature = histogram(frame, result['box']) if result.get('box') is not None else None
        similarity = float(feature @ self.anchor) if feature is not None and self.anchor is not None else 0.
        reason = self.gate.check(result, similarity)
        if prompt is not None and result.get('box') is None:
            reason = 'initial_mask_empty'
        if reason:
            self.state = 'LOST'
        else:
            self.state = 'TRACKING'
            if result.get('box') is not None:
                if predicted is not None and iou(predicted, result['box']) < .5:
                    correction = '|'.join(filter(None, (correction, 'flow_drift_corrected_by_sam')))
                self.last_box = list(result['box'])
        result.update(reid_top1=similarity, correction_reason=correction, motion_reason=motion_reason,
                      low_score_streak=self.gate.low_scores, low_histogram_streak=self.gate.low_appearance)
        return self.decorate(result, before, reason or ('initialized' if prompt is not None else 'sam_accepted'), tick)

    def start_search(self, frame, stamp):
        if self.search is None:
            from object_search import ObjectSearch
            self.search = ObjectSearch(self.detector, self.cfg, self.reference, self.class_id)
        self.search.submit(frame, stamp, self.last_box)

    def step(self, frame, stamp, box=None, detections=None):
        if box is not None:
            return self.initialize(frame, stamp, box, self.class_id)
        if self.state == 'TRACKING':
            return self.track(frame, stamp)
        # Detection and DINO inference live in a separate bounded worker.
        resolved = self.poll_search(frame, stamp)
        if resolved is not None and self.state == 'TRACKING':
            return resolved
        self.start_search(frame, stamp)
        before, self.state = self.state, 'SEARCHING'
        result = dict(mask=None, box=None, dx=None, dy=None, inference_ms=0., search_pending=True,
                      search_reason='worker_pending', reid_top1=0.)
        return self.decorate(result, before, 'async_search', time.perf_counter())

    def poll_search(self, frame, stamp):
        if self.search is None:
            return None
        found = self.search.get()
        if found is None or found['stamp'] <= self.last_search_stamp:
            return None
        tick, before = time.perf_counter(), self.state
        prior_stamp, self.last_search_stamp = self.last_search_stamp, found['stamp']
        result = dict(mask=None, box=None, dx=None, dy=None, inference_ms=0.,
                      reid_top1=found['top1'], search_reason=found['reason'], search_top2=found['top2'])
        candidate = found['box']
        if stamp-found['stamp'] > self.cfg.search_max_age:
            candidate, result['search_reason'] = None, 'stale_search_result'
        if candidate is None:
            self.hits, self.candidate = 0, None
        else:
            consistent = (self.candidate is not None and found['stamp']-prior_stamp <= self.cfg.search_max_gap
                          and iou(candidate, self.candidate) >= self.cfg.search_candidate_iou)
            self.hits = self.hits+1 if consistent else 1
            self.candidate = candidate
            if self.hits >= self.cfg.search_frames:
                # Initialize on the detection's EXACT image, then advance to the
                # current image if the camera moved on while search ran.
                self.gate = ObjectGate(self.cfg)
                self.motion = ImageMotion(self.cfg.motion_width)
                result = self.track(found['frame'], found['stamp'], candidate)
                verified = (self.state == 'TRACKING' and result.get('box') is not None
                            and result.get('iou', 0.) >= self.cfg.high_score
                            and result.get('reid_top1', 0.) >= self.cfg.histogram_high
                            and iou(result['box'], candidate) >= self.cfg.search_candidate_iou)
                if verified and stamp > found['stamp']:
                    result = self.track(frame, stamp)
                if verified and self.state == 'TRACKING' and result.get('box') is not None:
                    self.reacquired += 1
                    self.hits, self.candidate = 0, None
                    result['search_reason'] = 'verified_similarity_margin_streak_and_sam'
                    result['correction_reason'] = 'object_reacquired_sam_reset'
                    return self.decorate(result, before, 'object_reacquired', tick)
                self.hits, self.candidate = 0, None
                result['search_reason'] = 'sam_verification_failed'
        self.state = 'SEARCHING'
        result.update(search_hits=self.hits, search_pending=False)
        return self.decorate(result, before, result['search_reason'], tick)

    def close(self):
        if self.search:
            self.search.close()
            self.search = None
