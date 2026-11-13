import unittest
from unittest.mock import patch
import numpy as np
from recovery.identity_scores import same_support,fuse
from recovery.continuity import select_component
from check_object_continuity import Tests as Fixture
from recovery.native_memory import NativeMemoryRecovery


class IdentityTests(unittest.TestCase):
    def test_nested_masks_are_not_distinct_identity(self):
        body=np.zeros((100,100),bool);body[10:90,20:60]=True
        face=np.zeros_like(body);face[10:30,30:50]=True
        self.assertTrue(same_support(body,face,{}))
        self.assertFalse(same_support(body,np.roll(body,50,axis=1),{}))

    def test_disconnected_supported_parts_are_preserved(self):
        parts=np.zeros((100,100),bool);parts[10:25,20:40]=True;parts[35:80,10:50]=True
        chosen,count,split=select_component(parts,parts)
        np.testing.assert_array_equal(chosen,parts)
        self.assertEqual(count,2);self.assertFalse(split)

    def test_confidence_contains_no_color(self):
        value=dict(iou=.9,logit=5,anchor_similarity=.9,score=.9,color=1)
        diag=dict(flow_mask_iou=.9,temporal_area_ratio=1,continuity_distance=1,continuity_limit=30)
        first=fuse(value,diag,{'margin':.04})[0]
        value['color']=0
        self.assertEqual(first,fuse(value,diag,{'margin':.04})[0])
        self.assertLess(fuse(value,dict(diag,flow_mask_iou=.1),{'margin':.04})[0],first)

    def test_nested_proposals_have_no_false_margin_penalty(self):
        f=Fixture();f.setUp();t=f.t
        # Use the real grouping method, independent of neural inference.
        t.ambiguity=type(t).ambiguity.__get__(t)
        body=f.mask.copy();face=body.copy();face[40:]=False
        proposals=[dict(f.value,mask=face),dict(f.value,mask=body)]
        with patch.object(NativeMemoryRecovery,'ambiguity',side_effect=lambda *args:setattr(t,'proposals',proposals)):
            reason=t.ambiguity(f.frame,dict(f.value))
        self.assertIsNone(reason);self.assertEqual(t.diag['identity_candidates'],0)

    def test_stale_probe_cannot_confirm_identity(self):
        f=Fixture();f.setUp();t=f.t
        from unittest.mock import Mock
        t.probe_worker=Mock();t.current_stamp=2.;t.identity_epoch=2
        t.probe_result=dict(epoch=1,stamp=2.)
        self.assertEqual(t.async_evidence(f.frame,f.value,'anchor_reason'),'probe_pending')
        t.probe_worker.submit.assert_called_once()

    def test_feedback_confidence_and_legacy_header(self):
        import tempfile,csv
        from pathlib import Path
        from recovery.selection import record_feedback
        from calibrate_confidence import calibrate
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'frames.csv'
            record_feedback(path,1,1.,0,True,False,.9)
            labels=path.with_suffix('.labels.csv')
            self.assertEqual(calibrate(labels)['status'],'insufficient_labeled_classes')
            with labels.open() as handle:
                self.assertEqual(next(csv.DictReader(handle))['confidence'],'0.9')
            labels.write_text('frame,time_s,attempt_id,correct,tracker_feedback_applied\n')
            record_feedback(path,2,2.,0,False,False,.2)
            with labels.open() as handle:
                self.assertTrue(all(len(row)==5 for row in csv.reader(handle)))


if __name__=='__main__':
    unittest.main()
