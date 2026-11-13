"""Headless transport/export regression tests; no CUDA/model needed."""
import csv
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import video_playback as video


class Clock:
    now = 0.

    def tick(self):
        return self.now


class Tracker:
    def __init__(self, clock, delay):
        self.clock, self.delay = clock, delay
        self.stamps = []

    def step(self, frame, stamp, box=None):
        if box is None:
            assert stamp > self.stamps[-1]
        self.stamps.append(stamp)
        self.clock.now += self.delay
        status = {2: 'LOST', 3: 'SEARCHING', 4: 'REACQUIRE'}.get(len(self.stamps), 'TRACKING')
        return dict(status=status, box=[2, 2, 16, 16] if status == 'TRACKING' else None,
                    mask=None, dx=1, dy=2, inference_ms=10., reid_top1=.95)


class Telemetry:
    def __init__(self, path):
        pass

    def submit(self, data):
        pass

    def close(self):
        pass


class PlaybackTests(unittest.TestCase):
    def run_clip(self, mode='offline', keys=(ord('s'), ord('p')), delay=.12, **options):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10., (64, 48))
            self.assertTrue(writer.isOpened())
            for index in range(10):
                writer.write(np.full((48, 64, 3), index * 20, np.uint8))
            writer.release()
            args = SimpleNamespace(source=str(source), mode=mode, start_frame=0,
                                   max_frames=None, resize_width=32, loop=False,
                                   save_video=root / 'out.avi', csv=root / 'frames.csv', telemetry=root / 't.json')
            vars(args).update(options)
            clock = Clock()
            tracker = Tracker(clock, delay)
            actions = iter(keys)

            def wait(_):
                clock.now += .02
                self.assertLess(clock.now, 30, 'playback hung')
                return next(actions, -1)

            with ExitStack() as stack:
                for method in ('namedWindow', 'imshow', 'destroyAllWindows'):
                    stack.enter_context(patch.object(video.cv2, method))
                stack.enter_context(patch.object(video.cv2, 'getWindowProperty', return_value=1))
                stack.enter_context(patch.object(video.cv2, 'waitKey', side_effect=wait))
                stack.enter_context(patch.object(video.time, 'perf_counter', side_effect=clock.tick))
                stack.enter_context(patch.object(video, 'select_box', return_value=[2, 2, 16, 16]))
                video.run_video(args, lambda _: (tracker, None), lambda frame, _: frame.copy(), Telemetry)
            with args.csv.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            cap = cv2.VideoCapture(str(args.save_video))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(rows))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 32)
            cap.release()
            return rows, tracker.stamps

    def test_offline_every_frame_and_media_time(self):
        rows, stamps = self.run_clip(delay=2.)
        self.assertEqual([int(r['frame']) for r in rows], list(range(10)))
        self.assertEqual(stamps, [i / 10 for i in range(10)])
        self.assertEqual(rows[1]['status'], 'LOST')
        self.assertEqual(rows[4]['status'], 'TRACKING')

    def test_realtime_skips_and_keeps_csv_video_aligned(self):
        rows, stamps = self.run_clip(mode='realtime', delay=.35)
        self.assertEqual([int(r['frame']) for r in rows], list(range(10)))
        self.assertTrue(any(r['status'] == 'SKIPPED' for r in rows))
        self.assertLess(len(stamps), 10)
        self.assertTrue(all(not r['x1'] for r in rows if r['status'] == 'SKIPPED'))

    def test_start_limit_and_paused_steps(self):
        rows, stamps = self.run_clip(start_frame=3, max_frames=3,
                                    keys=(ord('s'), ord('d'), ord('d')))
        self.assertEqual([int(r['frame']) for r in rows], [3, 4, 5])
        self.assertEqual(stamps, [.3, .4, .5])

    def test_backward_requires_new_selection(self):
        rows, stamps = self.run_clip(keys=(ord('s'), ord('d'), ord('a'), ord('s'), ord('q')))
        self.assertEqual([int(r['frame']) for r in rows], [0, 1, 0, 0])
        self.assertEqual(rows[2]['status'], 'WAIT_SELECTION')
        self.assertEqual(stamps, [0., .1, 0.])

    def test_loop_resets_selection(self):
        rows, stamps = self.run_clip(start_frame=8, loop=True,
                                    keys=(ord('s'), ord('p'), -1, -1, ord('s'), ord('q')))
        self.assertEqual([int(r['frame']) for r in rows], [8, 9, 8, 8])
        self.assertEqual(rows[2]['status'], 'WAIT_SELECTION')
        self.assertEqual(stamps, [.8, .9, .8])

    def test_headless_init_frame_bbox_and_limit(self):
        rows, stamps = self.run_clip(no_display=True, init_frame=2, init_bbox=[2,2,14,14], max_frames=3)
        self.assertEqual([int(r['frame']) for r in rows], [2,3,4])
        self.assertEqual(stamps, [.2,.3,.4])
        self.assertEqual(rows[0]['selected_bbox'], '[2, 2, 14, 14]')


if __name__ == '__main__':
    unittest.main()
