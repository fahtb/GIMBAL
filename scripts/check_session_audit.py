import unittest
from session_audit import analyze


def row(i,state,valid,reason='',generation=0):
    return dict(frame=str(i),time_s=str(i*.1),status=state,valid=str(valid),
                transition_reason=reason,generation=str(generation),**{'pass':'0'})


class Tests(unittest.TestCase):
    def test_search_without_literal_lost_counts_loss(self):
        result=analyze([row(0,'TRACKING',True),row(1,'SEARCHING',False),row(2,'SEARCHING',False),row(3,'TRACKING',True)])
        self.assertEqual((result['losses'],result['resumes']),(1,1))
        self.assertAlmostEqual(result['observed_seconds'],.3)

    def test_manual_reselection_is_not_recovery(self):
        result=analyze([row(0,'TRACKING',True),row(1,'COAST',False),row(2,'TRACKING',True,'anchor_initialized',1)])
        self.assertEqual((result['losses'],result['resumes']),(1,0))

    def test_ambiguous_can_still_be_valid(self):
        result=analyze([row(0,'TRACKING',True),row(1,'AMBIGUOUS',True),row(2,'TRACKING',True)])
        self.assertEqual(result['losses'],0)

    def test_frame_log_search_loss_summary(self):
        import tempfile
        from pathlib import Path
        from video_playback import FrameLog
        with tempfile.TemporaryDirectory() as folder:
            log=FrameLog(Path(folder)/'test.csv')
            for i,state in enumerate(('TRACKING','SEARCHING','TRACKING')):
                valid=state=='TRACKING'
                log.write(0,(i,i*.1,None),dict(status=state,valid=valid,box=[0,0,10,10] if valid else None),.01)
            self.assertEqual((log.lost,log.recovered),(1,1))
            log.handle.close()

    def test_reduced_flow_preserves_translation(self):
        import cv2,numpy as np
        from mask_display import DenseMaskTransport
        rng=np.random.default_rng(4)
        before=rng.integers(40,210,(480,640),dtype=np.uint8)
        before=cv2.GaussianBlur(before,(5,5),0)
        after=cv2.warpAffine(before,np.array([[1,0,5],[0,1,3]],np.float32),(640,480))
        mask=np.zeros_like(before,dtype=np.float32);mask[80:400,90:500]=1
        expected=cv2.warpAffine(mask,np.array([[1,0,5],[0,1,3]],np.float32),(640,480))>.5
        actual=DenseMaskTransport(45,1.,320).warp(mask,after,before)>.5
        self.assertGreater(np.logical_and(actual,expected).sum()/np.logical_or(actual,expected).sum(),.97)


if __name__=='__main__':
    unittest.main()
