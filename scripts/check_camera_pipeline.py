"""Headless check of the actual capture + inference workers on a paced video."""
import json
import argparse
import time
from types import SimpleNamespace
import numpy as np
from camera import LatestCapture, Worker, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=str(ROOT / 'sam2/demo/data/gallery/03_blocks.mp4'))
    parser.add_argument('--box', nargs=4, type=float, default=[483, 122, 583, 232])
    parser.add_argument('--reid', action='store_true', help='Enable person ReID; use a person video/box')
    options = parser.parse_args()
    args = SimpleNamespace(checkpoint=ROOT / 'sam2/checkpoints/sam2.1_hiera_tiny.pt',
                           memory=24, min_iou=.5, max_gap=1., image_size=512, no_reid=not options.reid)
    capture = LatestCapture(options.source)
    worker = Worker(capture, args)
    rows = []
    try:
        deadline = time.perf_counter() + 180
        while not worker.ready or capture.get() is None:
            assert not worker.error, worker.error
            assert not capture.error, capture.error
            assert time.perf_counter() < deadline, 'Startup timed out'
            time.sleep(.01)
        first = capture.get()
        worker.select(first, options.box)
        last_seq = -1
        start = time.perf_counter()
        while time.perf_counter() - start < 8 and not capture.done:
            assert not worker.error, worker.error
            with worker.lock:
                result = worker.result
            if result and result['seq'] != last_seq:
                rows.append({k: v for k, v in result.items() if k not in ('mask', 'frame')})
                last_seq = result['seq']
            time.sleep(.01)
        assert rows and any(r['box'] is not None for r in rows)
        active = [r for r in rows if r['inference_ms'] > 0]
        assert any(r['skipped'] > 0 for r in active), 'Expected frame drops on paced 30 FPS source'
        assert all(b['seq'] > a['seq'] for a, b in zip(rows, rows[1:]))
        report = dict(tests='paced video capture, CUDA worker, latest frame drops, monotonic sequence PASS',
                      source=options.source, variant='SAMURAI + person ReID' if options.reid else 'restored SAM-only CUDA graph',
                      image_size=args.image_size,
                      active_results=len(active),
                      mean_latency_ms=float(np.mean([r['latency_ms'] for r in active])),
                      mean_inference_ms=float(np.mean([r['inference_ms'] for r in active])),
                      skipped_during_inference=sum(r['skipped'] for r in active), rows=rows)
        suffix = '_reid' if options.reid else ''
        (ROOT / f'test_results/camera_pipeline_report_{args.image_size}{suffix}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if result:
            from camera import draw
            import cv2
            cv2.imwrite(str(ROOT / f'test_results/restored_pipeline_last{suffix}.jpg'), draw(result['frame'], result))
        print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)
    finally:
        capture.stop.set()
        capture.thread.join(5)
        worker.thread.join(10)


if __name__ == '__main__':
    main()
