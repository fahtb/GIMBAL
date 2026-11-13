"""Compare execution overhead without changing SAMURAI outputs/configuration."""
import json
from contextlib import nullcontext
import cv2
import numpy as np
import torch
from live_tracker import LiveTracker, ROOT


def main():
    cap = cv2.VideoCapture(str(ROOT / 'sam2/demo/data/gallery/03_blocks.mp4'))
    ok, frame = cap.read()
    cap.release()
    assert ok
    tracker = LiveTracker(ROOT / 'sam2/checkpoints/sam2.1_hiera_tiny.pt', image_size=512)
    tracker.warmup()
    report = []
    reference = None
    for threads, persistent in ((4, False), (4, True), (1, True), (2, True)):
        torch.set_num_threads(threads)
        times, masks = [], []
        context = torch.autocast('cuda', dtype=torch.float16) if persistent else nullcontext()
        with torch.inference_mode(), context:
            for i in range(42):
                result = tracker.step(frame, i / 30, [483, 122, 583, 232] if i == 0 else None)
                assert result['box'] is not None
                if i >= 20:
                    times.append(result['inference_ms'])
                masks.append(result['mask'].copy())
        if reference is None:
            reference = masks
        agrees = all(np.array_equal(a, b) for a, b in zip(reference, masks))
        row = dict(threads=threads, persistent_autocast=persistent,
                   mean_ms=float(np.mean(times)), p95_ms=float(np.percentile(times, 95)),
                   masks_identical_to_baseline=agrees)
        report.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / 'test_results/sam_execution_benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
