"""Deterministic identity/state tests without GPU or model downloads."""
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from person_reid import ReIDConfig, ReIDTracker, HSVRegionsEmbedder
from reid_features import IdentityGallery, VisibleHSV, HSVFallbackBackend

BOX = [10, 0, 70, 120]
OTHER = [100, 0, 160, 120]


def scene():
    frame = np.zeros((128, 180, 3), np.uint8)
    for x in (10, 100):
        frame[:24, x:x+60] = (50, 100, 180)
        frame[24:72, x:x+60] = (40, 40, 210)
        frame[72:120, x:x+60] = (120, 30, 20)
    return frame


class FakeSAM:
    def __init__(self):
        self.fail = False
        self.box = BOX
        self.resets = 0

    def step(self, frame, stamp, box=None):
        if box is not None:
            self.box = list(box)
            self.resets += 1
        valid = not self.fail
        mask = np.zeros(frame.shape[:2], bool)
        x1, y1, x2, y2 = map(int, self.box)
        mask[y1:y2, x1:x2] = True
        return dict(box=self.box if valid else None, mask=mask if valid else None,
                    dx=0 if valid else None, dy=0 if valid else None,
                    iou=.99, object_logit=5., inference_ms=1., memory=2)


class FakeDetector:
    boxes = [BOX]
    def detect(self, frame):
        return self.boxes


class ReIDTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'test_results')
        self.addCleanup(self.temp.cleanup)
        cfg = ReIDConfig(csv_path=str(Path(self.temp.name) / 'events.csv'),
                         search_csv=str(Path(self.temp.name) / 'frames.csv'),
                         search_interval=.1, gallery_interval=.1, timeout=5)
        self.sam, self.det = FakeSAM(), FakeDetector()
        self.manager = ReIDTracker(self.sam, self.det, cfg)
        self.frame = scene()
        self.manager.step(self.frame, 0., BOX)

    def lose(self):
        self.sam.fail = True
        result = self.manager.step(self.frame, .1)
        self.assertEqual(result['status'], 'LOST')
        self.assertFalse(result['valid'])
        self.assertIsNotNone(result['predicted_box'])
        self.sam.fail = False

    def test_recovery_and_csv(self):
        self.lose()
        states = [self.manager.step(self.frame, t)['status'] for t in (.3, .5, .7, .9, 1.1, 1.3)]
        self.assertEqual(states, ['REACQUIRE', 'REACQUIRE', 'REACQUIRE', 'TRACKING', 'TRACKING', 'TRACKING'])
        self.assertEqual(self.sam.resets, 2)
        self.assertEqual(self.manager.successful, 1)
        with self.manager.log.path.open(newline='') as f:
            recovered = [r for r in csv.DictReader(f) if r['event'] == 'reacquired'][0]
        self.assertAlmostEqual(float(recovered['reacquisition_seconds']), .8)

    def test_identical_people_rejected_and_timeout(self):
        self.lose()
        self.det.boxes = [BOX, OTHER]
        for t in (.3, .5, .7, .9):
            result = self.manager.step(self.frame, t)
            self.assertEqual(result['status'], 'SEARCHING')
            self.assertEqual(result['reid_hits'], 0)
            self.assertIsNone(result['dx'])
        self.assertEqual(self.manager.step(self.frame, 5.2)['status'], 'WAIT_SELECTION')
        self.assertEqual(self.sam.resets, 1)

    def test_switching_candidates_never_confirms(self):
        self.manager.cfg.high_frames = 2
        self.lose()
        for index, t in enumerate((.3, .5, .7, .9, 1.1)):
            self.det.boxes = [BOX if index % 2 else OTHER]
            self.assertEqual(self.manager.step(self.frame, t)['reid_hits'], 1)
        self.assertEqual(self.sam.resets, 1)

    def test_gallery_bounded_and_freezes_for_overlap_low_confidence(self):
        for index in range(1, 26):
            self.manager.step(self.frame, index * .2)
        self.assertEqual(len(self.manager.gallery), 1, 'Duplicate frames must not fill rolling gallery')
        original = self.manager.gallery.anchor
        self.det.boxes = [BOX, [15, 0, 75, 120]]
        self.manager.step(self.frame, 6.)
        self.assertIs(original, self.manager.gallery.anchor)
        self.assertEqual(len(self.manager.gallery), 1)
        self.manager.cfg.gallery_iou = 1.
        self.det.boxes = [BOX]
        self.manager.step(self.frame, 6.2)
        self.assertIs(original, self.manager.gallery.anchor)
        self.assertEqual(len(self.manager.gallery), 1)

    def test_failed_sam_confirmation_never_exports_control(self):
        self.lose()
        for t in (.3, .5, .7):
            self.manager.step(self.frame, t)
        self.sam.fail = True
        result = self.manager.step(self.frame, .9)
        self.assertEqual(result['status'], 'SEARCHING')
        self.assertFalse(result['valid'])
        self.assertEqual(result['suspected_wrong'], 1)
        self.assertEqual(result['confirmed_wrong'], 0)

    def test_anchor_survives_ring_and_reacquisition(self):
        class Backend:
            def compare(self, a, b):
                return (1. if a is b else .95), 'test'
        gallery = IdentityGallery(Backend())
        anchor = object()
        gallery.seed(anchor)
        for _ in range(50):
            self.assertTrue(gallery.add(object()))
        self.assertEqual(len(gallery), 20)
        self.assertIs(gallery.anchor, anchor)
        self.assertFalse(gallery.add(anchor))
        # Full manager must retain both banks through automatic reacquisition.
        anchor = self.manager.gallery.anchor
        self.manager.gallery.rolling.extend([anchor]*19)
        self.lose()
        for t in (.3,.5,.7,.9):
            self.manager.step(self.frame,t)
        self.assertEqual(self.manager.state,'TRACKING')
        self.assertEqual(len(self.manager.gallery),20)
        self.assertIs(self.manager.gallery.anchor,anchor)

    def test_provisional_gain_and_human_labels(self):
        self.lose()
        result = self.manager.step(self.frame,.3)
        self.assertTrue(result['provisional'])
        self.assertTrue(result['valid'])
        self.assertEqual(result['control_gain'],.25)
        self.assertTrue(self.manager.label(True,.31,result['attempt_id']))
        self.assertFalse(self.manager.label(True,.32,result['attempt_id']))
        self.assertEqual(self.manager.confirmed_correct,1)
        self.assertTrue(self.manager.label(False,.33,result['attempt_id']))
        self.assertEqual(self.manager.confirmed_correct,0)
        self.assertEqual(self.manager.confirmed_wrong,1)
        self.assertEqual(self.manager.state,'LOST')

    def test_two_tier_floor_and_single_person_decay(self):
        self.lose()
        self.manager.candidate_count=1
        self.manager.top1=.78
        self.assertEqual(self.manager.gate(1.),'below_threshold')
        self.manager.top1=.91
        self.assertEqual(self.manager.gate(1.),'qualified')
        self.assertEqual(self.manager.required_streak,3)
        self.manager.top1=.98
        self.manager.gate(1.)
        self.assertEqual(self.manager.required_streak,1)
        self.manager.cfg.minimum_similarity=.86
        self.manager.cfg.single_candidate_decay=.002
        self.manager.cfg.decay_after=0
        self.manager.top1=.87
        self.assertEqual(self.manager.gate(10.),'qualified')
        self.manager.candidate_count=2
        self.assertEqual(self.manager.gate(10.),'below_threshold')

    def test_partial_crop_uses_visible_body(self):
        backend=VisibleHSV()
        full=np.zeros((240,120,3),np.uint8)
        full[20:60,30:90]=(40,100,200)
        full[60:140,30:90]=(30,30,220)
        full[140:220,30:90]=(200,20,20)
        reference=backend.encode(full,[30,20,90,220])
        partial=full[:140].copy()
        query=backend.encode(partial,[30,20,90,140])
        score,region=backend.compare(query,reference)
        self.assertGreater(score,.9)
        self.assertTrue(region.startswith('upper'))

    def test_embedding_backend_falls_back_without_mixing_vectors(self):
        class BrokenBackend:
            def encode(self, *args):
                raise RuntimeError('OSNet not installed')
        backend=HSVFallbackBackend(BrokenBackend())
        with self.assertWarns(RuntimeWarning):
            descriptor=backend.encode(self.frame, BOX)
        value,region=backend.compare(descriptor,descriptor)
        self.assertGreater(value,.99)
        self.assertTrue(region.startswith('hsv_fallback:'))

    def test_empty_detections_and_duplicate_frame(self):
        self.lose()
        self.det.boxes = []
        result = self.manager.step(self.frame, .3)
        self.assertEqual(result['reid_hits'], 0)
        with self.assertRaises(ValueError):
            self.manager.step(self.frame, .3)

    def test_ambiguous_person_arrives_during_confirmation(self):
        self.lose()
        for t in (.3, .5, .7):
            self.manager.step(self.frame, t)
        self.det.boxes = [BOX, OTHER]
        result = self.manager.step(self.frame, .9)
        self.assertEqual(result['status'], 'SEARCHING')
        self.assertFalse(result['valid'])
        self.assertEqual(self.manager.successful, 0)

    def test_manual_selection_clears_identity(self):
        self.lose()
        self.manager.step(self.frame, .3, OTHER)
        self.assertEqual(len(self.manager.gallery), 1)
        self.assertIsNone(self.manager.lost_at)
        self.assertEqual(self.manager.state, 'TRACKING')

    def test_embedding_and_config(self):
        embedder = HSVRegionsEmbedder()
        first = embedder.encode(self.frame, BOX)
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1., places=5)
        self.assertIsNone(embedder.encode(self.frame, [0, 0, 2, 2]))
        changed = self.frame.copy()
        changed[:120, 10:70] = (0, 255, 0)
        self.assertLess(float(first @ embedder.encode(changed, BOX)), .88)
        with self.assertRaises(ValueError):
            ReIDConfig(gallery_size=21).validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
