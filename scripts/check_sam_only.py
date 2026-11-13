"""Small CPU tests for SAM-only routing and diagnostic state transitions."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from sam_only import SAMOnlyTracker, sam_fps


class FakeSAM:
    min_iou = .5

    def __init__(self):
        self.calls = []

    def step(self, frame, stamp, box=None, continue_lost=False):
        self.calls.append((stamp, box, continue_lost))
        good = stamp != 1
        return dict(status='TRACKING' if good else 'LOST - press R',
                    box=[1, 2, 5, 6] if good else None, mask='mask' if good else None,
                    dx=0, dy=0, inference_ms=50., predictor_called=True,
                    predicted_iou=.9 if good else .1, object_score_logits=4 if good else -4,
                    mask_area=16 if good else 0, sam_reason='accepted' if good else 'empty')


class Tests(unittest.TestCase):
    def test_default_stops_and_nulls_observations(self):
        sam = FakeSAM()
        tracker = SAMOnlyTracker(sam)
        tracker.step(None, 0, [1, 2, 5, 6])
        self.assertEqual(tracker.step(None, 1)['status'], 'LOST')
        result = tracker.step(None, 2)
        self.assertEqual(len(sam.calls), 2)
        self.assertFalse(result['predictor_called'])
        self.assertNotIn('predicted_iou', result)
        self.assertEqual(sam_fps(result), 0.)

    def test_keep_running_stride_does_not_claim_recovery(self):
        sam = FakeSAM()
        tracker = SAMOnlyTracker(sam, True, 2)
        tracker.step(None, 0, [1, 2, 5, 6])
        tracker.step(None, 1)
        result = tracker.step(None, 2)
        self.assertTrue(sam.calls[-1][2])
        self.assertTrue(result['sam_observation_valid'])
        self.assertEqual(result['status'], 'SEARCHING')
        self.assertIsNone(result['mask'])
        self.assertFalse(tracker.step(None, 3)['predictor_called'])
        self.assertTrue(tracker.step(None, 4)['predictor_called'])
        selected = tracker.step(None, 5, [1, 2, 5, 6])
        self.assertEqual(selected['status'], 'TRACKING')
        self.assertEqual(selected['mask'], 'mask')

    def test_fps(self):
        self.assertEqual(sam_fps({'inference_ms': 0}), 0)
        self.assertEqual(sam_fps({'inference_ms': 50}), 20)
        self.assertEqual(sam_fps({'inference_ms': 50, 'predictor_called': False}), 0)

    def test_factory_does_not_import_legacy_models_even_for_person(self):
        import camera
        import builtins
        real_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('person_reid', 'target_tracker', 'object_search', 'ultralytics', 'transformers'):
                raise AssertionError('Non-SAM import: ' + name)
            return real_import(name, *args, **kwargs)
        args = SimpleNamespace(checkpoint=None, memory=24, min_iou=.5, max_gap=1., target_mode='person')
        with patch.object(camera, 'LiveTracker') as constructor, patch('builtins.__import__', guarded):
            tracker, detector = camera.create_tracker(args)
        self.assertIsInstance(getattr(tracker, 'tracker', tracker), SAMOnlyTracker)
        self.assertIsNone(detector)
        constructor.return_value.warmup.assert_called_once()


if __name__ == '__main__':
    unittest.main()
