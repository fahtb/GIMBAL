"""Short CPU contracts for temporal identity, no user videos or model weights."""
import tempfile
from pathlib import Path
from unittest.mock import Mock
import unittest
import cv2
import numpy as np
import check_native_memory
from recovery.continuity import ObjectContinuityRecovery,select_component
from recovery.metrics import identity_metrics,truth_at
from raw_recording import RawRecorder
from video_playback import VideoSource


class Tests(unittest.TestCase):
    def setUp(self):
        fixture=check_native_memory.Tests();fixture.setUp()
        self.frame,self.mask,self.sam=fixture.frame,fixture.mask,fixture.sam
        self.sam.index=1
        self.sam.snapshot_pending_memory=Mock(return_value=(1,{},[20,20,29,79],.95))
        self.sam.store_verified_memory=Mock()
        self.sam.rebase_verified=Mock(return_value=dict(self.sam.step.return_value))
        self.sam.step.side_effect=lambda *a,**k:dict(self.sam.step.return_value)
        self.t=ObjectContinuityRecovery(self.sam)
        self.t.gallery=[object()];self.t.anchor_area=self.t.recent_area=600
        self.t.state='TRACKING';self.t.last_good=self.t.trusted_stamp=0.
        self.t.continuity_active=True
        self.t.previous_stamp=0.;self.t.previous_mask=self.mask.copy()
        self.t.previous_gray=cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        self.t.last_box=[20,20,29,79]
        self.value=fixture.value
        self.value.update(proto=object(),lab=np.zeros(1))
        self.t.assess=Mock(side_effect=lambda f,m,s,l,*a:dict(self.value,mask=m,box=__import__('sam_memory').mask_box(m),iou=s,logit=l) if m is not None and m.any() else None)
        self.t.anchor_confirmation=Mock(return_value=None)
        self.t.ambiguity=Mock(return_value=None)
        self.t.remember=Mock()

    def test_two_components_keep_original_not_lost(self):
        other=np.zeros_like(self.mask);other[20:80,70:80]=True
        self.sam.step.return_value['mask']=self.mask|other
        result=self.t.step(self.frame,.03)
        self.assertEqual(result['status'],'TRACKING_LOW')
        self.assertGreater(result['control_gain'],0)
        self.assertLess(result['control_gain'],1)
        self.assertTrue(result['valid']);np.testing.assert_array_equal(result['mask'],self.mask)
        self.sam.store_verified_memory.assert_not_called()

    def test_probe_unavailable_is_searching_not_identity_ambiguity(self):
        self.t.state='SEARCHING';self.t.continuity_active=False
        self.t.ambiguity.return_value='probe_pending'
        result=self.t.step(self.frame,1.)
        self.assertEqual(result['status'],'SEARCHING');self.assertFalse(result['valid'])
        self.sam.store_verified_memory.assert_not_called()

    def test_low_aggregate_does_not_override_hard_continuity(self):
        from unittest.mock import patch
        with patch('recovery.continuity.fuse',return_value=(.4,{})):
            result=self.t.step(self.frame,.03)
        self.assertEqual(result['status'],'TRACKING_LOW');self.assertTrue(result['valid'])
        self.sam.store_verified_memory.assert_not_called()

    def test_recovery_waits_for_elapsed_probation_then_rebases(self):
        self.t.state='SEARCHING';self.t.continuity_active=False
        for i in range(10):
            result=self.t.step(self.frame,1+i/30)
            if i<9:self.assertFalse(result['valid'])
        self.sam.rebase_verified.assert_called_once()
        self.assertEqual(result['status'],'TRACKING_LOW');self.assertTrue(result['valid'])
        self.assertTrue(result['valid']);np.testing.assert_array_equal(result['mask'],self.mask)
        self.sam.commit_memory.assert_not_called()

    def test_competing_anchor_does_not_discard_continuous_target(self):
        self.t.native_count=2
        self.t.anchor_confirmation.return_value='temporal_anchor_disagreement'
        result=self.t.step(self.frame,.03)
        self.assertEqual(result['status'],'TRACKING_LOW');self.assertTrue(result['valid'])
        self.sam.commit_memory.assert_not_called()

    def test_jump_repairs_with_positive_negative_and_freezes_memory(self):
        other=np.roll(self.mask,50,axis=1)
        self.sam.step.return_value['mask']=other
        self.sam.decode_points=Mock(return_value=(np.array([self.mask]),[.95],[5.],[1.]))
        result=self.t.step(self.frame,.03)
        self.assertEqual(result['transition_reason'],'continuity_reprompt')
        self.assertEqual(self.sam.decode_points.call_args.kwargs['labels'],[1,0])
        self.sam.commit_memory.assert_not_called()

    def test_failed_repair_coasts_then_searches_never_switches(self):
        self.sam.step.return_value['mask']=np.roll(self.mask,50,axis=1)
        self.t.repair=Mock(return_value=None)
        self.t.anchor_confirmation.return_value='temporal_anchor_disagreement'
        first=self.t.step(self.frame,.03)
        self.assertEqual(first['status'],'COAST');self.assertFalse(first['valid'])
        later=self.t.step(self.frame,.8)
        self.assertEqual(later['status'],'SEARCHING');self.assertFalse(later['valid'])
        self.sam.commit_memory.assert_not_called()

    def test_merged_connected_mask_is_not_accepted_as_target(self):
        merged=self.mask.copy();merged[30:60,25:85]=True
        self.assertFalse(self.t.continuous(merged,self.mask))

    def test_gradual_merge_cannot_become_new_area_reference(self):
        grown=self.mask.copy();grown[20:80,30:37]=True
        self.assertTrue(self.t.merged_growth(grown,self.mask))
        self.assertTrue(self.t.merged_growth(grown,grown))

    def test_coast_does_not_follow_occluder_features(self):
        from unittest.mock import patch
        self.t.state='COAST';self.t.native_velocity=np.array([30.,0.])
        with patch('cv2.goodFeaturesToTrack') as find:
            _,predicted,ok=self.t.predict(self.frame,.1)
        find.assert_not_called();self.assertFalse(ok)
        np.testing.assert_array_equal(predicted,np.roll(self.mask,3,axis=1))

    def test_ambiguous_recovery_never_bypasses_margin(self):
        self.t.state='SEARCHING';self.t.trusted_stamp=None
        self.t.ambiguity.return_value='ambiguous_sam_objects'
        for i in range(10):
            result=self.t.step(self.frame,1+i*.1)
            self.assertFalse(result['valid']);self.assertEqual(self.t.hits,0)

    def test_recovery_freezes_through_probation(self):
        self.t.state='SEARCHING';self.t.trusted_stamp=None
        for i in range(8):
            result=self.t.step(self.frame,1+i*.1)
            self.sam.commit_memory.assert_not_called()
        self.assertEqual(result['status'],'TRACKING');self.assertTrue(result['valid'])

    def test_raw_timestamp_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'raw.avi';writer=RawRecorder(path)
            for i,stamp in enumerate([100.,100.04,100.11]):
                writer.submit((i,stamp,self.frame))
            writer.close()
            source=VideoSource(path)
            try:
                stamps=[source.read()[1] for _ in range(3)]
                np.testing.assert_allclose(stamps,[0,.04,.11])
                self.assertEqual(source.index_at_time(.05),1)
            finally:
                source.close()

    def test_gt_interpolation_and_switch_event(self):
        gt=[dict(frame=i,visible=True,box=[0,0,10,10],distractors=[[20,20,30,30]]) for i in (0,15)]
        self.assertEqual(truth_at(gt,7)['box'],[0,0,10,10])
        rows=[dict(frame=i,time_s=i/30,status='TRACKING',x1=20,y1=20,x2=30,y2=30) for i in range(16)]
        metrics=identity_metrics(rows,gt)
        self.assertEqual(metrics['identity_switches'],1)
        self.assertEqual(metrics['correct_with_distractor_percent'],0)

    def test_ambiguous_search_cannot_skip_probation(self):
        self.t.state='AMBIGUOUS';self.t.continuity_active=False
        result=self.t.step(self.frame,.1)
        self.assertFalse(result['valid']);self.assertEqual(self.t.hits,1)

    def test_camera_ui_flow_output_wiring(self):
        import camera,threading,time,sys
        from types import SimpleNamespace
        from unittest.mock import patch
        frame=self.frame.copy();mask=self.mask.copy()
        class Capture:
            def __init__(self,*args):
                self.packet=(0,time.perf_counter(),frame)
                self.stop=threading.Event();self.thread=SimpleNamespace(join=Mock())
                self.error=None;self.done=False;self.measured_fps=30.
            def get(self):
                self.packet=(self.packet[0]+1,time.perf_counter(),frame)
                return self.packet
        class Worker:
            def __init__(self,capture,args):
                self.lock=threading.Lock();self.generation=1;self.ready=True;self.error=None
                self.thread=SimpleNamespace(join=Mock());self.label=Mock()
                self.result=dict(seq=0,generation=1,captured=capture.packet[1],frame=frame,mask=mask,
                                 box=[20,20,29,79],dx=-25,dy=0,valid=True,status='TRACKING',
                                 target_mode='SAM_ONLY',inference_ms=10,latency_ms=10,skipped=0)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            argv=['camera.py','--source','0','--target-mode','object','--csv',str(root/'live.csv'),
                  '--save-video',str(root/'view.avi'),'--profile',str(root/'profile.json'),
                  '--telemetry',str(root/'target.json')]
            with patch.object(sys,'argv',argv),patch.object(camera,'LatestCapture',Capture),patch.object(camera,'Worker',Worker), \
                 patch('cv2.namedWindow'),patch('cv2.setMouseCallback'),patch('cv2.imshow'), \
                 patch('cv2.waitKey',side_effect=[-1,-1,ord('q')]),patch('cv2.getWindowProperty',return_value=1),patch('cv2.destroyAllWindows'):
                camera.main()
            self.assertTrue((root/'view.avi').exists())
            self.assertTrue((root/'live.csv').exists())
            self.assertTrue((root/'profile.json').exists())


if __name__=='__main__':
    unittest.main()
