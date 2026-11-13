"""CPU regression tests for memory bounds, identity color and probation overlay."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import torch
from live_tracker import LiveTracker
from sam_memory import SAMMemoryTracker, lab_descriptor, overlap
from mask_display import overlay


class Tests(unittest.TestCase):
    def test_crop_recenter_keeps_anchor_and_checks_identity_before_commit(self):
        for accepted in (True, False):
            anchor, recent = object(), object()
            banks = {'cond_frame_outputs': {0: anchor, 3: recent}, 'non_cond_frame_outputs': {4: recent}}
            sam = SimpleNamespace(last_box=[1,2,5,8], shape=(256,256,3), anchor_index=0,
                                  outputs=banks, pinned_quality={3:.9}, reset_motion=Mock(),
                                  step=Mock(return_value={'mask': np.ones((5,5),bool), 'iou':.9, 'object_logit':3}),
                                  commit_memory=Mock())
            tracker = SAMMemoryTracker.__new__(SAMMemoryTracker)
            tracker.sam,tracker.state,tracker.cfg = sam,'TRACKING',{'pinned_frames':4}
            tracker.gallery = [object()]
            gallery = tracker.gallery
            tracker.assess = Mock(return_value={'box':[2,2,6,7]})
            tracker.search_valid = Mock(return_value=accepted)
            tracker.update_motion = Mock()
            tracker.decorate = Mock(side_effect=lambda result,*args:result)
            tracker.recenter_crop(np.zeros((256,256,3),np.uint8),1.,[2,2,6,7])
            self.assertIs(sam.outputs['cond_frame_outputs'][0],anchor)
            self.assertIs(tracker.gallery,gallery)
            self.assertTrue(sam.step.call_args.kwargs['preserve_memory'])
            if accepted:
                sam.commit_memory.assert_called_once_with(4,pin=False,condition=True)
                self.assertFalse(sam.outputs['non_cond_frame_outputs'])
            else:
                sam.commit_memory.assert_not_called()
                self.assertIs(sam.outputs,banks)
                self.assertEqual(tracker.state,'SEARCHING')

    def test_anchor_top_quality_and_recent_memory_are_bounded(self):
        tracker = LiveTracker.__new__(LiveTracker)
        tracker.outputs = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        tracker.pinned_quality = {}
        tracker.memory = 16
        for index in range(60):
            score = .99 if index == 2 else .7
            out = {key: torch.tensor(1.) for key in ('maskmem_features', 'maskmem_pos_enc', 'obj_ptr',
                                                   'best_iou_score', 'object_score_logits', 'kf_ious')}
            tracker.pending_memory = (index, out, [1, 1, 5, 5], score, 5.)
            tracker.commit_memory(4)
        self.assertIn(0, tracker.outputs['cond_frame_outputs'])
        self.assertIn(2, tracker.outputs['cond_frame_outputs'])
        self.assertLessEqual(len(tracker.outputs['cond_frame_outputs']), 5)
        self.assertEqual(len(tracker.outputs['non_cond_frame_outputs']), 16)
        tracker.pending_memory = (60, out, [1, 1, 5, 5], 1., 10.)
        tracker.commit_memory(4, pin=False)
        self.assertNotIn(60, tracker.outputs['cond_frame_outputs'])
        tracker.commit_memory(4,pin=False,condition=True)
        tracker.pending_memory = (61,out,[1,1,5,5],.8,5.)
        tracker.commit_memory(4)
        self.assertIn(0,tracker.outputs['cond_frame_outputs'])
        self.assertIn(60,tracker.outputs['cond_frame_outputs'])
        self.assertEqual(tracker.active_prompt_index,60)

    def test_lab_uses_mask_not_background(self):
        image = np.zeros((40, 40, 3), np.uint8)
        mask = np.zeros((40, 40), bool)
        mask[10:20, 10:20] = True
        image[mask] = (60, 150, 40)
        first = lab_descriptor(image, mask)
        image[~mask] = 255
        np.testing.assert_array_equal(first, lab_descriptor(image, mask))

    def test_probation_has_fill_and_contour(self):
        frame = np.zeros((80, 100, 3), np.uint8)
        mask = np.zeros((80, 100), bool)
        mask[15:35, 15:35] = True
        result = dict(mask=mask, box=[15, 15, 34, 34], probation=True)
        view = overlay(frame, result)
        self.assertGreater(view[20, 20, 2], view[20, 20, 1])
        self.assertGreater(view[15, 20, 2], 200)
        result['probation'] = False
        self.assertGreater(overlay(frame, result)[20, 20, 1], 50)

    def test_overlap(self):
        self.assertEqual(overlap([1, 1, 5, 5], [1, 1, 5, 5]), 1.)
        self.assertEqual(overlap([1, 1, 5, 5], [20, 20, 25, 25]), 0.)


if __name__ == '__main__':
    unittest.main()
