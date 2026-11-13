"""Paced video test of actual SAMURAI CUDA + live motion workers (no GUI)."""
import json
import time
from types import SimpleNamespace
import numpy as np
from camera import LatestCapture, Worker, ROOT
from live_motion import MotionWorker


def main():
    args = SimpleNamespace(checkpoint=ROOT / 'sam2/checkpoints/sam2.1_hiera_tiny.pt',
                           memory=24, min_iou=.5, max_gap=1., image_size=512)
    capture = LatestCapture(str(ROOT / 'sam2/demo/data/gallery/03_blocks.mp4'))
    sam = Worker(capture, args)
    motion = MotionWorker(capture, sam)
    rows = []
    try:
        deadline = time.perf_counter() + 180
        while not sam.ready or capture.get() is None:
            assert not sam.error and not capture.error, (sam.error, capture.error)
            assert time.perf_counter() < deadline
            time.sleep(.01)
        sam.select(capture.get(), [483, 122, 583, 232])
        start, last_seq = time.perf_counter(), -1
        while time.perf_counter() - start < 8 and not capture.done:
            assert not sam.error and not motion.error, (sam.error, motion.error)
            with motion.lock:
                result = motion.result
            if result and result['seq'] != last_seq:
                rows.append({k: v for k, v in result.items() if k not in ('mask', 'frame')})
                last_seq = result['seq']
            time.sleep(.002)
        good = [r for r in rows if r['valid']]
        assert len(good) >= 5, [(r['seq'], r['status']) for r in rows[:50]]
        assert all(r['history_frames'] <= 45 for r in rows)
        report = dict(tests='actual SAMURAI + latest-frame motion workers + bounded history PASS',
                      valid_samples=len(good), total_samples=len(rows),
                      median_live_fps=float(np.median([r['tracking_fps'] for r in good])),
                      mean_live_latency_ms=float(np.mean([r['latency_ms'] for r in good])),
                      p95_live_latency_ms=float(np.percentile([r['latency_ms'] for r in good], 95)),
                      mean_flow_ms=float(np.mean([r['flow_ms'] for r in good])),
                      mean_sam_inference_ms=float(np.mean([r['sam_inference_ms'] for r in good])),
                      rows=rows)
        (ROOT / 'test_results/live_pipeline_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)
    finally:
        capture.stop.set()
        for worker in (capture, sam, motion):
            worker.thread.join(5)


if __name__ == '__main__':
    main()
