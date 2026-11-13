import unittest
import numpy as np
from sam_tiles import tile_windows


class Tests(unittest.TestCase):
    def test_overlapping_tiles_cover_native_frame_and_edges(self):
        cover = np.zeros((1080,1920),np.uint8)
        windows = tile_windows(cover.shape,512,.25,np.array([350,400]))
        for a,b,c,d in windows:
            self.assertTrue(0 <= a < c <= 1920 and 0 <= b < d <= 1080)
            cover[b:d,a:c] += 1
        self.assertTrue(np.all(cover > 0))
        self.assertGreater(cover.max(),1)

    def test_small_frame_is_single_tile(self):
        self.assertEqual(tile_windows((100,120),512,.25,np.array([60,50])), [[0,0,120,100]])


if __name__ == '__main__':
    unittest.main()
