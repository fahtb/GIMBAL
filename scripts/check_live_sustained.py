"""Actual CUDA SAMURAI + motion over a paced moving real image, with known translation.

This measures software latency after source publication, not camera exposure or
GUI latency. The source is synthetic camera motion of the bundled blocks image.
"""
import json
import argparse
import threading
import time
from types import SimpleNamespace
import cv2
import numpy as np
from camera import LatestCapture, Worker, ROOT
from live_motion import MotionWorker


class MovingScene(LatestCapture):
    @staticmethod
    def displacement(seq):
        return np.array([35 * np.sin(seq / 25), 20 * np.sin(seq / 37)])

    def run(self):
        while not self.start_read.wait(.05):
            if self.stop.is_set():
                return
        cap = cv2.VideoCapture(self.source)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            self.error = 'Cannot read source image'
            return
        start = time.perf_counter()
        try:
            for seq in range(210):
                if self.stop.is_set():
                    break
                self.stop.wait(max(0., start + seq / 30 - time.perf_counter()))
                dx, dy = self.displacement(seq)
                shifted = cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]),
                                         (frame.shape[1], frame.shape[0]))
                if seq >= 180:  # Full occlusion must stop, without waiting for SAM.
                    shifted[:] = 0
                with self.lock:
                    self.packet = (seq, time.perf_counter(), shifted)
        finally:
            self.done = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam-delay-ms', type=float, default=0,
                        help='test-only minimum SAM duration, to reproduce a slower GPU asynchronously')
    options = parser.parse_args()
    if options.sam_delay_ms:
        import camera
        original_step = camera.LiveTracker.step
        def delayed_step(self, *args, **kwargs):
            start = time.perf_counter()
            result = original_step(self, *args, **kwargs)
            time.sleep(max(0, options.sam_delay_ms / 1000 - (time.perf_counter() - start)))
            result['inference_ms'] = (time.perf_counter() - start) * 1000
            return result
        camera.LiveTracker.step = delayed_step
    args = SimpleNamespace(checkpoint=ROOT / 'sam2/checkpoints/sam2.1_hiera_tiny.pt',
                           memory=24, min_iou=.5, max_gap=1., image_size=512)
    capture = MovingScene(str(ROOT / 'sam2/demo/data/gallery/03_blocks.mp4'))
    sam = Worker(capture, args)
    motion = MotionWorker(capture, sam)
    rows, errors = [], []
    try:
        deadline = time.perf_counter() + 180
        while not sam.ready or capture.get() is None:
            assert not sam.error and not capture.error, (sam.error, capture.error)
            assert time.perf_counter() < deadline
            time.sleep(.005)
        first = capture.get()
        d = capture.displacement(first[0])
        sam.select(first, (np.array([483, 122, 583, 232]) + np.tile(d, 2)).tolist())
        last_seq = -1
        while not capture.done:
            assert not sam.error and not motion.error, (sam.error, motion.error)
            with motion.lock:
                r = motion.result
            if r and r['seq'] != last_seq:
                rows.append({k: v for k, v in r.items() if k not in ('mask', 'frame')})
                last_seq = r['seq']
                if r['valid'] and r['seq'] < 180:
                    truth_center = np.array([533., 177.]) + capture.displacement(r['seq'])
                    center = (np.array(r['box'][:2]) + r['box'][2:]) / 2
                    errors.append(float(np.linalg.norm(center - truth_center)))
            time.sleep(.002)
        good = [r for r in rows if r['valid'] and r['seq'] < 180]
        report = dict(valid_samples=len(good), total_samples=len(rows),
                      test_only_minimum_sam_duration_ms=options.sam_delay_ms,
                      first_loss=next((dict(seq=r['seq'], reason=r['status']) for r in rows if r['status'].startswith('LOST')), None),
                      median_update_fps=float(np.median([r['tracking_fps'] for r in good])) if good else 0,
                      mean_live_latency_ms=float(np.mean([r['latency_ms'] for r in good])) if good else None,
                      p95_live_latency_ms=float(np.percentile([r['latency_ms'] for r in good], 95)) if good else None,
                      mean_flow_ms=float(np.mean([r['flow_ms'] for r in good])) if good else None,
                      mean_sam_ms=float(np.mean([r['sam_inference_ms'] for r in good])) if good else None,
                      mean_center_error_px=float(np.mean(errors)) if errors else None,
                      p95_center_error_px=float(np.percentile(errors, 95)) if errors else None,
                      rows=rows)
        suffix = f'_{int(options.sam_delay_ms)}ms' if options.sam_delay_ms else ''
        (ROOT / f'test_results/live_sustained_report{suffix}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)
        assert len(good) >= 100, [(r['seq'], r['status']) for r in rows[::10]]
        assert report['p95_center_error_px'] < 8
        assert not any(r['valid'] for r in rows if r['seq'] >= 183)
        assert all(r['history_frames'] <= 45 for r in rows)
        print('Sustained paced 30 FPS source + actual CUDA SAMURAI, target position, occlusion and bounded history PASS')
    finally:
        capture.stop.set()
        for worker in (capture, sam, motion):
            worker.thread.join(5)


if __name__ == '__main__':
    main()
