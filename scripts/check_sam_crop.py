import unittest
import cv2
import numpy as np
from source_frame import SourceFrame
from sam_crop import NativeCropTracker
from mask_display import MaskDisplay


class FakeTracker:
    cfg = dict(small_side=64, crop_min_size=256, crop_scale=5)
    state = 'WAIT_SELECTION'

    def step(self, frame, stamp, box=None):
        self.received = frame.copy()
        self.box = box
        self.state = 'TRACKING'
        mask = np.zeros(frame.shape[:2], bool)
        x,y,u,v = map(round, box)
        mask[y:v, x:u] = True
        return dict(status='TRACKING', transition='WAIT_SELECTION->TRACKING', mask=mask, box=box)


class Tests(unittest.TestCase):
    def test_native_crop_and_roundtrip_geometry(self):
        original = np.random.default_rng(42).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
        display = cv2.resize(original, (960, 540))
        frame = SourceFrame(display, original)
        core = FakeTracker()
        tracker = NativeCropTracker(core)
        box = [285., 290., 295., 303.]
        result = tracker.step(frame, 0, box)
        self.assertEqual(result['status'], 'SMALL')
        self.assertEqual(core.received.shape, (256, 256, 3))
        a,b,c,d = tracker.window
        np.testing.assert_array_equal(core.received, original[b:d,a:c])
        np.testing.assert_allclose(result['box'], box)
        self.assertEqual(result['mask'].shape, display.shape[:2])
        self.assertTrue(result['mask'][295, 290])
        self.assertIsNone(frame.copy().original)
        self.assertIs(frame.original, original)

    def test_tiny_mask_survives_realtime_without_eight_corners(self):
        frame = np.full((180, 320, 3), 100, np.uint8)
        mask = np.zeros((180, 320), bool)
        mask[80:88, 100:105] = True
        frame[mask] = (30, 160, 90)
        sam = dict(seq=0, captured=0., frame=frame, mask=mask, box=[100,80,104,87],
                   status='SMALL', target_mode='SAM_ONLY', small=True, valid=True)
        display = MaskDisplay()
        first = display.step((0, 0., frame), sam)
        second = display.step((1, 1/60, frame.copy()), sam)
        self.assertEqual(first['status'], 'SMALL')
        self.assertEqual(second['status'], 'SMALL')
        self.assertIsNotNone(second['mask'])
        self.assertGreater(second['mask'].sum(), 20)


if __name__ == '__main__':
    unittest.main()
