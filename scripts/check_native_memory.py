"""Identity guards and unprompted, frozen-memory probation, without GPU."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from recovery.native_memory import NativeMemoryRecovery, shape_signature
from sam_memory import SAMMemoryTracker


class Tests(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((100,100),bool);self.mask[20:80,20:30]=True
        self.frame=np.zeros((100,100,3),np.uint8)
        self.anchor=object()
        self.sam=SimpleNamespace(min_iou=.5,outputs={'cond_frame_outputs':{0:self.anchor},'non_cond_frame_outputs':{}},
            snapshot_motion=Mock(return_value={'saved':True}),restore_motion=Mock(),commit_memory=Mock(),
            step=Mock(return_value=dict(mask=self.mask,box=[20,20,29,79],iou=.95,object_logit=5,dx=0,dy=0,inference_ms=1.)),
            motion_reference_stamp=0.)
        self.tracker=NativeMemoryRecovery(self.sam)
        self.tracker.gallery=[object()]
        self.tracker.anchor_area=self.tracker.recent_area=600
        self.tracker.last_good=0.
        self.tracker.state='SEARCHING'
        self.value=dict(mask=self.mask,box=[20,20,29,79],iou=.95,logit=5.,similarity=.99,
                        anchor_similarity=.99,shape_ratio=1.,ratio=1.,color=1.,score=.97)
        self.tracker.assess=Mock(return_value=self.value)
        self.tracker.anchor_confirmation=Mock(return_value=None)
        self.tracker.ambiguity=Mock(return_value=None)

    def test_color_cannot_override_original_shape_or_memory(self):
        self.value['shape_ratio']=.1
        result=self.tracker.step(self.frame,.1)
        self.assertFalse(result['valid'])
        self.assertEqual(result['transition_reason'],'selected_shape_mismatch')
        self.sam.commit_memory.assert_not_called()
        self.value.update(shape_ratio=1.,anchor_similarity=.5)
        self.assertFalse(self.tracker.step(self.frame,.2)['valid'])

    def test_freezes_memory_until_confirmed_without_reprompt(self):
        gallery=self.tracker.gallery
        for index in range(1,8):
            result=self.tracker.step(self.frame,index/30)
            self.sam.commit_memory.assert_not_called()
        result=self.tracker.step(self.frame,8/30)
        self.assertEqual(result['status'],'TRACKING')
        self.assertEqual(result['reacquired'],1)
        self.sam.commit_memory.assert_called_once()
        self.assertIs(gallery,self.tracker.gallery)
        self.assertIs(self.sam.outputs['cond_frame_outputs'][0],self.anchor)
        for call in self.sam.step.call_args_list:
            self.assertEqual(call.kwargs,{'continue_lost':True})
            self.assertEqual(len(call.args),2)

    def test_failed_probation_never_poisoned_memory(self):
        for index in range(1,4):
            self.tracker.step(self.frame,index/30)
        self.tracker.probation=1  # Final confirmation is mandatory before commit.
        self.tracker.anchor_confirmation.return_value='temporal_anchor_disagreement'
        result=self.tracker.step(self.frame,4/30)
        self.assertFalse(result['valid'])
        self.assertEqual(result['status'],'SEARCHING')
        self.sam.commit_memory.assert_not_called()
        self.sam.restore_motion.assert_called()

    def test_no_color_dependency_and_no_high_score_margin_bypass(self):
        self.value['color']=0.
        self.assertIsNone(self.tracker.identity_reason(self.value))
        self.tracker.ambiguity.return_value='ambiguous_sam_objects'
        self.assertFalse(self.tracker.step(self.frame,.1)['valid'])
        self.assertEqual(self.tracker.hits,0)

    def test_shape_rotation_invariant(self):
        self.assertAlmostEqual(shape_signature(self.mask),shape_signature(np.rot90(self.mask)),places=5)

    def test_replay_old_gate_accepts_shape_mismatch_new_gate_rejects(self):
        # Gate-level replay, not a claim that a real SAM model emitted these scores.
        old=SAMMemoryTracker(self.sam)
        old.state='SEARCHING';old.gallery=[object()]
        value=dict(self.value,shape_ratio=.1,plausible=True,recent_ratio=3.,stability=1.)
        old.assess=Mock(return_value=value)
        old.search=Mock(return_value=[])
        old.seed_candidate=Mock(return_value={'valid':True})
        old.step(self.frame,.1)
        old.step(self.frame,.2)
        old.seed_candidate.assert_called_once()
        self.tracker.assess.return_value=value
        self.assertFalse(self.tracker.step(self.frame,.1)['valid'])
        self.assertFalse(self.tracker.step(self.frame,.2)['valid'])


if __name__=='__main__':
    unittest.main()
