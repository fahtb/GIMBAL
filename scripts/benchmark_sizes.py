"""Compare real SAMURAI encoder resolutions on identical frames and GPU."""
import gc
import json
import cv2
import numpy as np
import torch
from live_tracker import LiveTracker, ROOT


def main():
    reports = []
    references = {}
    for size in (1024, 768, 512):
        tracker = LiveTracker(ROOT / 'sam2/checkpoints/sam2.1_hiera_tiny.pt', image_size=size)
        tracker.warmup()
        cap = cv2.VideoCapture(str(ROOT / 'sam2/demo/data/gallery/03_blocks.mp4'))
        times, agreements, rows = [], [], []
        for i in range(45):
            ok, frame = cap.read()
            assert ok
            result = tracker.step(frame, i / 30, [483, 122, 583, 232] if i == 0 else None)
            if i >= 5 and result['inference_ms']:
                times.append(result['inference_ms'])
            if size == 1024 and result['mask'] is not None:
                references[i] = result['mask'].copy()
            elif i in references and result['mask'] is not None:
                a, b = references[i], result['mask']
                agreements.append(float(np.logical_and(a, b).sum() / np.logical_or(a, b).sum()))
            rows.append({k: v for k, v in result.items() if k != 'mask'})
            if i == 30:
                from camera import draw
                cv2.imwrite(str(ROOT / f'test_results/size_{size}_frame30.jpg'), draw(frame, result))
        cap.release()
        report = dict(image_size=size, mean_ms=float(np.mean(times)), fps=1000 / float(np.mean(times)),
                      p95_ms=float(np.percentile(times, 95)),
                      tracking_frames=sum(r['box'] is not None for r in rows),
                      mean_mask_iou_vs_1024=float(np.mean(agreements)) if agreements else None,
                      rows=rows)
        reports.append(report)
        print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)
        del tracker
        gc.collect()
        torch.cuda.empty_cache()
    (ROOT / 'test_results/resolution_benchmark.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
