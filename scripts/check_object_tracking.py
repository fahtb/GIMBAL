"""Deterministic object gates, crop transforms, worker and routing regressions."""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from object_tracking import ObjectConfig, ObjectGate, CropSAM, ObjectTracker
from object_search import ObjectSearch


class FakeSAM:
    def __init__(self):
        self.model = SimpleNamespace(image_size=512)
        self.min_iou = .5
        self.calls = []
        self.last_box = None

    def step(self, frame, stamp, box=None):
        self.calls.append((frame.shape, stamp, box))
        if box is not None:
            self.last_box = [int(v) for v in box]
        b = self.last_box
        mask = np.zeros(frame.shape[:2], bool)
        mask[b[1]:b[3],b[0]:b[2]] = True
        return dict(box=b.copy(), mask=mask, dx=0., dy=0., inference_ms=1., iou=.95,
                    mask_area=int(mask.sum()), object_logit=5.)


class ObjectTests(unittest.TestCase):
    def test_hysteresis_and_area_collapse(self):
        gate = ObjectGate(ObjectConfig())
        row = dict(iou=.9, mask_area=100)
        self.assertIsNone(gate.check(row, .9))
        row['iou'] = .1
        self.assertIsNone(gate.check(row, .9))
        self.assertIsNone(gate.check(row, .9))
        self.assertEqual(gate.check(row, .9), 'mask_score_low_streak')
        row['iou'] = .9
        self.assertIsNone(gate.check(row, .9))
        self.assertEqual(gate.low_scores, 0)
        row['mask_area'] = 5
        self.assertEqual(gate.check(row, .9), 'mask_area_collapse')

    def test_appearance_anchor_hysteresis(self):
        gate = ObjectGate(ObjectConfig())
        row = dict(iou=.9, mask_area=100)
        self.assertIsNone(gate.check(row, .1))
        self.assertIsNone(gate.check(row, .4))  # neutral band holds streak
        self.assertEqual(gate.low_appearance, 1)
        self.assertIsNone(gate.check(row, .6))  # recovered appearance resets it
        self.assertEqual(gate.low_appearance, 0)
        for _ in range(2):
            self.assertIsNone(gate.check(row, .1))
        self.assertEqual(gate.check(row, .1), 'anchor_histogram_low_streak')

    def test_crop_roundtrip_and_recenter(self):
        sam, cfg = FakeSAM(), ObjectConfig()
        crop = CropSAM(sam, cfg)
        frame = np.zeros((300,400,3), np.uint8)
        box = [100,100,120,120]
        result, reason = crop.step(frame, 0., box, box)
        self.assertEqual(result['box'], box)
        self.assertEqual(result['mask'].shape, frame.shape[:2])
        self.assertEqual(result['mask'].sum(), 400)
        self.assertEqual(reason, 'small_target_crop_init')
        moved = [175,100,195,120]
        result, reason = crop.step(frame, .1, None, moved)
        self.assertEqual(reason, 'crop_recenter_motion_prediction')
        self.assertEqual(result['box'], moved)
        self.assertEqual(result['dx'], -15.)

    def test_known_class_worker_is_async_and_bounded(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        class Detector:
            def detect_all(self, frame, classes, size):
                calls.append(classes)
                entered.set()
                release.wait(2)
                return [dict(box=[10,10,40,40], class_id=7)]
        frame = np.full((60,60,3), 80, np.uint8)
        worker = ObjectSearch(Detector(), ObjectConfig(), (frame,[10,10,40,40]), 7)
        try:
            start = time.perf_counter()
            worker.submit(frame, 0., [10,10,40,40])
            self.assertLess(time.perf_counter()-start, .1)
            self.assertTrue(entered.wait(1))
            for n in range(20):
                worker.submit(frame, n+1., [10,10,40,40])
            self.assertEqual(worker.pending[1], 20.)
            release.set()
            deadline = time.perf_counter()+3
            while worker.get() is None or worker.get()['stamp'] != 20.:
                self.assertLess(time.perf_counter(), deadline)
                time.sleep(.005)
            self.assertEqual(worker.get()['reason'], 'qualified')
            self.assertTrue(all(c == [7] for c in calls))
        finally:
            release.set()
            worker.close()
        self.assertFalse(worker.thread.is_alive())

    def test_recovery_requires_distinct_consistent_frames(self):
        frame = np.full((100,100,3), 80, np.uint8)
        tracker = ObjectTracker(FakeSAM(), None, ObjectConfig())
        tracker.initialize(frame, 0., [10,10,40,40], 7)
        tracker.state = 'LOST'
        result = dict(stamp=.1, box=[10,10,40,40], top1=.95, top2=.1,
                      reason='qualified', frame=frame)
        tracker.search = SimpleNamespace(get=lambda: result, close=lambda: None)
        self.assertEqual(tracker.poll_search(frame,.1)['status'], 'SEARCHING')
        self.assertIsNone(tracker.poll_search(frame,.1))
        result['stamp'] = .2
        self.assertEqual(tracker.poll_search(frame,.2)['status'], 'SEARCHING')
        result['stamp'] = .3
        self.assertEqual(tracker.poll_search(frame,.3)['status'], 'TRACKING')
        self.assertEqual(tracker.reacquired, 1)
        tracker.close()

    def test_unknown_class_uses_dino_worker(self):
        frame = np.full((100,100,3), 80, np.uint8)
        class Dino:
            def __init__(self, cfg, reference):
                pass
            def rank(self, frame, last_box, stop):
                return [(.95,[10,10,40,40]), (.94,[11,11,39,39]), (.5,[60,60,90,90])]
        with patch('object_search.DinoSearch', Dino):
            worker = ObjectSearch(None, ObjectConfig(), (frame,[10,10,40,40]), None)
            try:
                worker.submit(frame, .1, [10,10,40,40])
                deadline = time.perf_counter()+2
                while worker.get() is None:
                    self.assertLess(time.perf_counter(), deadline)
                    time.sleep(.005)
                self.assertEqual(worker.get()['reason'], 'qualified')
                self.assertEqual(worker.get()['top2'], .5)
            finally:
                worker.close()


if __name__ == '__main__':
    unittest.main()
