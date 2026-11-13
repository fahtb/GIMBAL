"""Geometry and appearance checks independent of GPU/model weights."""
import unittest
import numpy as np
import cv2
from small_target import region,bounds,structural_match,Settings

class SmallGeometry(unittest.TestCase):
    def test_live_frame_144_requires_emergency_recenter(self):
        from small_target import roi_edge_risk
        # Logged original camera pixels, seven frames after the last rebase.
        self.assertTrue(roi_edge_risk((256,101,438,283),[323,118,349,140],.18))
        self.assertFalse(roi_edge_risk((256,101,438,283),[332,181,364,206],.18))
    def test_regions_contain_border_targets(self):
        for box in ([0,0,6,8],[634,472,640,480],[310,230,325,245]):
            r=region((480,640),np.array(box,float),128)
            self.assertTrue(0<=r[0]<r[2]<=640 and 0<=r[1]<r[3]<=480)
            self.assertTrue(r[0]<=box[0] and r[1]<=box[1] and r[2]>=box[2] and r[3]>=box[3])
    def test_roundtrip_mask(self):
        full=np.zeros((480,640),bool);full[200:209,320:326]=1
        roi=region(full.shape,bounds(full),128);x,y,r,b=roi
        restored=np.zeros_like(full);restored[y:b,x:r]=full[y:b,x:r]
        np.testing.assert_array_equal(full,restored)
        np.testing.assert_array_equal(bounds(restored),[320,200,326,209])
    def test_display_does_not_drop_tiny_or_keep_lost_mask(self):
        from mask_display import DenseMaskTransport
        view=DenseMaskTransport(5,1.)
        frame=np.full((48,64,3),80,np.uint8);mask=np.zeros((48,64),bool);mask[20,30]=1
        sam=dict(seq=0,captured=0.,frame=frame,mask=mask,box=[30,20,30,20],small=True,valid=True,status='TRACKING')
        tracked=view.step((0,0.,frame),sam)
        self.assertTrue(tracked['valid']);self.assertEqual(int(tracked['mask'].sum()),1)
        sam.update(seq=1,captured=.03,mask=None,box=None,valid=False,status='LOST')
        lost=view.step((1,.03,frame),sam)
        self.assertIsNone(lost['mask']);self.assertIsNone(lost['dx']);self.assertFalse(lost['valid'])
    def test_texture_alignment_not_color(self):
        a=np.zeros((24,24),np.float32);cv2.line(a,(3,3),(20,20),1,3);cv2.line(a,(3,20),(20,3),1,3)
        shifted=np.roll(a,1,axis=0);wrong=np.zeros_like(a);cv2.circle(wrong,(12,12),7,1,2)
        self.assertGreater(structural_match(a.ravel(),shifted.ravel()),.9)
        self.assertLess(structural_match(a.ravel(),wrong.ravel()),Settings().recovery_appearance)

if __name__=='__main__':unittest.main()
