import unittest
from camera import camera_timing


class Tests(unittest.TestCase):
    def test_new_image_does_not_reset_old_sam_age(self):
        result=camera_timing(dict(captured=10.,sam_latency_ms=90.),(42,10.25,None),10.26)
        self.assertAlmostEqual(result['age_ms'],260.)
        self.assertAlmostEqual(result['display_latency_ms'],10.)
        self.assertEqual(result['latency_ms'],90.)
        self.assertEqual(result['captured'],10.)

    def test_waiting_for_selection_has_no_sam_latency(self):
        result=camera_timing({},(0,20.,None),20.005)
        self.assertEqual(result['latency_ms'],0.)
        self.assertAlmostEqual(result['display_latency_ms'],5.)


if __name__=='__main__':
    unittest.main()
