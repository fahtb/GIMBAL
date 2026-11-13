"""Actual CUDA SAMURAI + YOLO test on recorded video with injected full occlusion."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from live_tracker import LiveTracker, ROOT
from person_reid import ReIDConfig, ReIDTracker, YOLOPersonDetector


def main():
    cfg = ReIDConfig.load(ROOT / 'configs/reid.json')
    cfg.csv_path = 'test_results/reid_gpu_events.csv'
    tracker = LiveTracker(ROOT / 'sam2/checkpoints/sam2.1_hiera_tiny.pt', image_size=512, cuda_graph=True)
    tracker.warmup()
    detector = YOLOPersonDetector(cfg)
    manager = ReIDTracker(tracker, detector, cfg)
    cap = cv2.VideoCapture(str(ROOT / 'sam2/demo/data/gallery/05_default_juggle.mp4'))
    rows = []
    torch.cuda.reset_peak_memory_stats()
    try:
        for index in range(120):
            ok, frame = cap.read()
            if not ok:
                break
            if 25 <= index < 40:
                frame[:] = 0  # Known synthetic full occlusion, not a real camera test.
            result = manager.step(frame, index / 30., [590, 40, 850, 680] if index == 0 else None)
            rows.append({k: v for k, v in result.items() if k != 'mask'})
            if index % 10 == 0:
                print(index, result['status'], round(result['reid_top1'], 3),
                      round(result['pipeline_ms'], 1), flush=True)
            if result['reacquired'] > 0 and result['valid']:
                from camera import draw
                cv2.imwrite(str(ROOT / 'test_results/reid_recovered.jpg'), draw(frame, result))
    finally:
        cap.release()
    report = dict(source='05_default_juggle.mp4; black frames 25..39 injected; source timestamps 30 FPS',
                  camera_verified=False, recovered=manager.successful,
                  suspected_wrong=manager.suspected_wrong,
                  mean_tracking_pipeline_ms=float(np.mean([r['pipeline_ms'] for r in rows if r['status'] == 'TRACKING'])),
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20, rows=rows)
    path = ROOT / 'test_results/reid_gpu_report.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
    assert any(r['status'] == 'LOST' for r in rows), 'Occlusion did not trigger loss'
    assert manager.successful >= 1, 'Person was not automatically recovered'
    assert all(not r['valid'] and r['dx'] is None for r in rows if r['status'] != 'TRACKING')
    assert len(manager.gallery) <= cfg.gallery_size
    print('CUDA ReID integration PASS', flush=True)


if __name__ == '__main__':
    main()
