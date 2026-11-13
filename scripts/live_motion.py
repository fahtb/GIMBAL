"""Bounded short-term image motion between asynchronous SAMURAI confirmations.

This is not segmentation or re-identification. It never feeds estimated masks
back into SAMURAI and never reacquires after a loss without a new user selection.
"""
from collections import deque
import threading
import time
import cv2
import numpy as np


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - intersection
    return float(intersection / max(union, 1))


def gray_frame(frame):
    h, w = frame.shape[:2]
    width = min(w, 640)
    return cv2.cvtColor(cv2.resize(frame, (width, max(1, round(h * width / w)))), cv2.COLOR_BGR2GRAY)


class FlowState:
    def __init__(self, frame, mask, box, seq, stamp):
        self.gray = gray_frame(frame)
        h, w = frame.shape[:2]
        gh, gw = self.gray.shape
        self.scale = np.array([gw / w, gh / h], dtype=np.float32)
        foreground = cv2.resize(mask.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST)
        foreground = cv2.erode(foreground, np.ones((3, 3), np.uint8))
        self.points = cv2.goodFeaturesToTrack(self.gray, maxCorners=100, qualityLevel=.015,
                                              minDistance=5, mask=foreground, blockSize=5)
        x1, y1, x2, y2 = box
        self.corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32) * self.scale
        self.seq, self.stamp = seq, stamp
        self.valid = self.points is not None and len(self.points) >= 8

    def box(self):
        corners = self.corners / self.scale
        return np.r_[corners.min(0), corners.max(0)].tolist()

    def advance(self, seq, stamp, gray):
        if seq == self.seq:
            return self.valid
        if not self.valid or stamp <= self.stamp or stamp - self.stamp > .2 or gray.shape != self.gray.shape:
            self.valid = False
            return False
        options = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .02))
        moved, ok, error = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None, **options)
        if moved is None:
            self.valid = False
            return False
        back, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, moved, None, **options)
        if back is None:
            self.valid = False
            return False
        residual = np.linalg.norm(self.points.reshape(-1, 2) - back.reshape(-1, 2), axis=1)
        good = (ok.ravel() != 0) & (back_ok.ravel() != 0) & (residual < 1.2) & (error.ravel() < 22)
        good &= np.isfinite(moved.reshape(-1, 2)).all(1)
        if good.sum() < 8 or good.mean() < .5:
            self.valid = False
            return False
        old, new = self.points.reshape(-1, 2)[good], moved.reshape(-1, 2)[good]
        affine, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC,
                                                     ransacReprojThreshold=2., maxIters=300)
        if affine is None or inliers is None or inliers.sum() < 8 or inliers.mean() < .65:
            self.valid = False
            return False
        scale = np.linalg.norm(affine[:, 0])
        predicted = cv2.transform(self.corners[None], affine)[0]
        displacement = np.linalg.norm(predicted.mean(0) - self.corners.mean(0))
        gh, gw = gray.shape
        if not .85 < scale < 1.18 or displacement > max(gh, gw) * .18:
            self.valid = False
            return False
        center = predicted.mean(0)
        if not (0 <= center[0] < gw and 0 <= center[1] < gh):
            self.valid = False
            return False
        self.points = new[inliers.ravel() != 0].astype(np.float32).reshape(-1, 1, 2)
        self.corners, self.gray = predicted, gray
        self.seq, self.stamp = seq, stamp
        return True


class MotionTracker:
    def __init__(self, max_sam_age=1.2):
        self.max_sam_age = max_sam_age
        self.history = deque(maxlen=45)  # <=45 grayscale frames, never a video archive
        self.reset(0)

    def reset(self, generation):
        self.generation = generation
        self.flow = None
        self.lost = False
        self.reason = 'WAITING FOR SAMURAI'
        self.sam_stamp = None
        self.sam_seq = -1
        self.sam_ms = 0.
        self.history.clear()
        self.geometry = {}
        self.completed = deque(maxlen=30)

    def fail(self, reason):
        self.lost = True
        self.flow = None
        self.reason = reason

    def step(self, packet, sam=None):
        start = time.perf_counter()
        seq, stamp, frame = packet
        gray = gray_frame(frame)
        if not self.history or seq > self.history[-1][0]:
            self.history.append((seq, stamp, gray))
            for old_seq in list(self.geometry):
                if old_seq < self.history[0][0]:
                    del self.geometry[old_seq]
        if self.flow is not None and not self.lost:
            if not self.flow.advance(seq, stamp, gray):
                self.fail('LOST: motion evidence weak - R')
        if (sam is not None and sam.get('generation') == self.generation
                and self.sam_seq < sam['seq'] <= seq and not self.lost):
            self.sam_seq = sam['seq']
            self.sam_ms = sam['inference_ms']
            if sam['box'] is None:
                self.fail('LOST: SAMURAI rejected target - R')
            elif stamp - sam['captured'] > self.max_sam_age:
                self.fail('LOST: SAMURAI confirmation expired - R')
            elif sam['seq'] <= seq:
                candidate = FlowState(sam['frame'], sam['mask'], sam['box'], sam['seq'], sam['captured'])
                if self.flow is not None:
                    # Reuse already verified target motion, rather than rerunning
                    # LK over 10-20 old frames at every delayed SAM confirmation.
                    anchor_corners = candidate.corners.copy()
                    match = None
                    for old_seq, old_stamp, old_gray in self.history:
                        if old_seq >= sam['seq'] and (old_seq in self.geometry or old_seq == seq):
                            match = (old_seq, old_stamp, old_gray)
                            break
                    if match is not None:
                        old_seq, old_stamp, old_gray = match
                        old_corners = self.flow.corners if old_seq == seq else self.geometry[old_seq]
                        candidate.advance(old_seq, old_stamp, old_gray)
                        old_box = np.r_[old_corners.min(0), old_corners.max(0)] / np.tile(self.flow.scale, 2)
                        if candidate.valid and box_iou(candidate.box(), old_box) >= .60:
                            transform = cv2.getAffineTransform(old_corners[:3].astype(np.float32),
                                                               self.flow.corners[:3].astype(np.float32))
                            aligned = cv2.transform(candidate.corners[None], transform)[0]
                            full_transform = cv2.getAffineTransform(anchor_corners[:3], aligned[:3].astype(np.float32))
                            gh, gw = gray.shape
                            mask = cv2.resize(sam['mask'].astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST)
                            mask = cv2.warpAffine(mask, full_transform, (gw, gh), flags=cv2.INTER_NEAREST)
                            box = np.r_[aligned.min(0), aligned.max(0)] / np.tile(self.flow.scale, 2)
                            candidate = FlowState(frame, mask, box, seq, stamp)
                        else:
                            candidate.valid = False
                    else:
                        candidate.valid = False
                else:
                    # First confirmation: bounded catch-up from the actual SAM
                    # frame. No motion has yet been accepted to reuse.
                    for old_seq, old_stamp, old_gray in self.history:
                        if old_seq > candidate.seq and not candidate.advance(old_seq, old_stamp, old_gray):
                            break
                if not candidate.valid or candidate.seq != seq:
                    self.fail('LOST: cannot align SAMURAI to live frame - R')
                elif self.flow is not None and box_iou(candidate.box(), self.flow.box()) < .60:
                    self.fail('LOST: motion / SAMURAI disagree - R')
                else:
                    self.flow = candidate
                    self.sam_stamp = sam['captured']
                    self.reason = 'LIVE MOTION | SAMURAI confirmed'
        if self.sam_stamp is not None and stamp - self.sam_stamp > self.max_sam_age:
            self.fail('LOST: SAMURAI confirmation expired - R')
        bbox = self.flow.box() if self.flow is not None and not self.lost else None
        if self.flow is not None and not self.lost:
            self.geometry[seq] = self.flow.corners.copy()
        h, w = frame.shape[:2]
        self.completed.append(time.perf_counter())
        fps = ((len(self.completed) - 1) / (self.completed[-1] - self.completed[0])
               if len(self.completed) > 1 else 0.)
        return dict(frame=frame, seq=seq, captured=stamp, box=bbox, mask=None,
                    status=self.reason, estimated=True, generation=self.generation,
                    valid=bbox is not None,
                    dx=None if bbox is None else (bbox[0] + bbox[2] - w) / 2,
                    dy=None if bbox is None else (bbox[1] + bbox[3] - h) / 2,
                    flow_ms=(time.perf_counter() - start) * 1000,
                    latency_ms=(time.perf_counter() - stamp) * 1000,
                    sam_age_ms=None if self.sam_stamp is None else (stamp - self.sam_stamp) * 1000,
                    sam_inference_ms=self.sam_ms, tracking_fps=fps,
                    history_frames=len(self.history))


class MotionWorker:
    def __init__(self, capture, sam_worker, max_sam_age=1.2):
        self.capture, self.sam_worker = capture, sam_worker
        self.max_sam_age = max_sam_age
        self.lock = threading.Lock()
        self.result = None
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        tracker = MotionTracker(self.max_sam_age)
        last_seq = -1
        try:
            while not self.capture.stop.is_set():
                packet = self.capture.get()
                with self.sam_worker.lock:
                    generation, sam = self.sam_worker.generation, self.sam_worker.result
                if generation != tracker.generation:
                    tracker.reset(generation)
                    last_seq = -1
                if packet is None or packet[0] == last_seq:
                    self.capture.stop.wait(.002)
                    continue
                result = tracker.step(packet, sam)
                last_seq = packet[0]
                with self.lock:
                    self.result = result
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error = str(exc)
