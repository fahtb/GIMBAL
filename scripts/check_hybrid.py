import unittest
import numpy as np
import cv2
from tracking_control import BoxKalman,SimulatedGimbal,safe_output
from hybrid_tracker import IdentityMemory,HybridConfig
from motion_flow import estimate_motion,partial_support

class HybridTests(unittest.TestCase):
    def test_search_crop_coordinates_and_rotation(self):
        rng=np.random.default_rng(19)
        patch=rng.integers(20,235,(30,18,3),dtype=np.uint8)
        frame=np.zeros((180,300,3),np.uint8);frame[50:80,180:198]=patch
        mask=np.zeros(frame.shape[:2],bool);mask[50:80,180:198]=True
        memory=IdentityMemory(frame,mask);cfg=HybridConfig(scales=(1.,))
        full=memory.proposals(frame,cfg)
        local=memory.proposals(frame,cfg,[180,50,198,80],local=True,radius_px=35)
        np.testing.assert_allclose(full[0]['box'],local[0]['box'],atol=1)
        rotated=np.zeros_like(frame);rotated[110:128,50:80]=np.rot90(patch)
        found=memory.proposals(rotated,cfg,rotation=1)
        np.testing.assert_allclose(found[0]['box'],[50,110,80,128],atol=1)
        rotated_mask=np.zeros_like(mask);rotated_mask[110:128,50:80]=True
        from small_target import appearance
        self.assertGreater(memory.similarity(appearance(rotated,rotated_mask)[0]),.9)
        rotated[50:80,180:198]=patch
        self.assertGreater(memory.competing_score(rotated,cfg,found[0]),.99)
    def test_partial_texture_and_distractor(self):
        rng=np.random.default_rng(17)
        frame=np.zeros((100,120,3),np.uint8)
        frame[20:80,30:90]=rng.integers(20,230,(60,60,3),dtype=np.uint8)
        mask=np.zeros(frame.shape[:2],bool);mask[20:80,30:90]=True
        partial=mask.copy();partial[:,60:]=False
        current=frame.copy();current[20:80,60:90]=0
        evidence=partial_support(frame,mask,current,partial,[30,20,90,80])
        self.assertGreater(evidence['similarity'],.99)
        self.assertAlmostEqual(evidence['visible_ratio'],.5)
        current[20:80,30:60]=rng.integers(20,230,(60,30,3),dtype=np.uint8)
        self.assertLess(partial_support(frame,mask,current,partial,[30,20,90,80])['similarity'],.9)
    def test_partial_display_does_not_restore_control(self):
        from mask_display import MaskDisplay
        from check_smooth_video import sam
        data=sam(0);data.update(target_mode='SAM_ONLY',status='UNCERTAIN',is_confirmed=False,dx=None,dy=None)
        shown=MaskDisplay().step((0,0.,data['frame']),data)
        self.assertIsNotNone(shown['mask'])
        self.assertIsNone(shown['dx']);self.assertIsNone(shown['dy'])
    def test_uncertain_counts_one_control_lock_episode(self):
        from session_audit import analyze
        rows=[dict(frame=i,time_s=i*.1,status=s,valid=v,is_confirmed=c) for i,(s,v,c) in enumerate([
            ('TRACKING',True,True),('UNCERTAIN',True,False),('REDETECTING',False,False),('RECOVERED',True,True)])]
        result=analyze(rows)
        self.assertEqual((result['losses'],result['resumes']),(1,1))
    def test_variable_dt_and_outlier(self):
        k=BoxKalman();k.reset([0,0,20,20],0)
        for t in (.1,.25,.4,.8):k.predict(t);k.update([t*20,0,t*20+20,20])
        self.assertGreater(k.x[4],5)
        self.assertGreater(k.innovation([900,900,920,920]),25)
        with self.assertRaises(ValueError):k.predict(.2)
    def test_controller_hold_limits_stale(self):
        c=SimulatedGimbal();last=0.
        for i in range(50):
            r=c.step([1,1],i*.02,True);self.assertLessEqual(abs(r['rate_x']),.6)
            self.assertLessEqual(abs(r['rate_x']-last),.041);last=r['rate_x']
        self.assertEqual(c.step(None,2.,False)['rate_x'],0)
        self.assertEqual(safe_output(dict(valid=True,is_confirmed=True,age_ms=500,gimbal_command=r))['gimbal_command']['mode'],'HOLD')
    def test_duplicate_candidates_keep_margin(self):
        rng=np.random.default_rng(4);patch=rng.integers(30,240,(24,24,3),dtype=np.uint8)
        frame=np.zeros((150,240,3),np.uint8);frame[50:74,40:64]=patch;frame[50:74,170:194]=patch
        mask=np.zeros(frame.shape[:2],bool);mask[50:74,40:64]=1
        memory=IdentityMemory(frame,mask);cfg=HybridConfig(scales=(1.,))
        ranked=memory.proposals(frame,cfg)
        self.assertGreaterEqual(len(ranked),2);self.assertLess(ranked[0]['score']-ranked[1]['score'],cfg.proposal_margin)
    def test_lk_translation(self):
        rng=np.random.default_rng(5);f=rng.integers(0,255,(120,160,3),dtype=np.uint8)
        m=np.zeros(f.shape[:2],bool);m[20:100,20:140]=1
        current=cv2.warpAffine(f,np.float32([[1,0,3],[0,1,2]]),(160,120))
        flow=estimate_motion(f,current,m);self.assertIsNotNone(flow)
        np.testing.assert_allclose(flow['matrix'][:,2],[3,2],atol=.7)
    def test_scale_aware_measurement_and_gallery(self):
        k=BoxKalman(relative_measurement=.03);k.reset([0,0,150,470],0)
        self.assertGreater(k.measurement_covariance()[3,3],k.measurement_covariance()[1,1])
        self.assertGreater(k.innovation([900,0,1050,470]),25)
        with self.assertRaises(ValueError):HybridConfig(confirmation_frames=0)
    def test_telemetry_watchdog(self):
        import tempfile,time,json
        from pathlib import Path
        from camera import TelemetryWriter
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'target.json';writer=TelemetryWriter(path)
            try:
                writer.submit(dict(valid=True,is_confirmed=True,age_ms=0,gimbal_command=dict(mode='TRACK',rate_x=.2,rate_y=0)))
                time.sleep(.4)
                self.assertEqual(json.loads(path.read_text())['gimbal_command']['mode'],'HOLD')
            finally:writer.close()
    def test_cuda_unavailable_clear_error(self):
        from unittest.mock import patch
        from source_camera_tracker import SourceCameraTracker
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA unavailable'):SourceCameraTracker()
    def test_camera_disconnect_releases(self):
        from unittest.mock import patch,MagicMock
        from camera import LatestCapture
        cap=MagicMock();cap.isOpened.return_value=True;cap.get.return_value=30.;cap.read.return_value=(False,None)
        with patch('cv2.VideoCapture',return_value=cap):
            capture=LatestCapture(0,dict(backend='dshow'));capture.start_read.set();capture.thread.join(2)
            self.assertTrue(capture.done);cap.release.assert_called()
    def test_labelled_metrics(self):
        from recovery.metrics import identity_metrics
        rows=[dict(frame=0,time_s=0,status='TRACKING',x1=0,y1=0,x2=10,y2=10,dx=0,dy=0),
              dict(frame=1,time_s=.1,status='RECOVERED',x1=20,y1=0,x2=30,y2=10,dx=0,dy=0)]
        gt=[dict(frame=0,visible=True,box=[0,0,10,10]),dict(frame=1,visible=False)]
        result=identity_metrics(rows,gt)
        self.assertEqual(result['success_iou50_percent'],100)
        self.assertEqual(result['false_locks_when_absent'],1)

if __name__=='__main__':unittest.main()
