"""20 reproducible synthetic exits/returns from a video frame; NOT 20 live trials."""
import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
import csv
import cv2
import numpy as np
from live_tracker import ROOT, LiveTracker


def load_baseline():
    spec = importlib.util.spec_from_file_location('baseline_reid', ROOT / 'backups/pre_reid_v2/person_reid.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['before', 'after'], required=True)
    args = p.parse_args()
    if args.variant == 'before':
        module = load_baseline()
        from reid_diagnostics import instrument_baseline
        manager_type = instrument_baseline(module.ReIDTracker)
        cfg = module.ReIDConfig.load(ROOT / 'backups/pre_reid_v2/reid.json')
    else:
        import person_reid as module
        manager_type = module.ReIDTracker
        cfg = module.ReIDConfig.load(ROOT / 'configs/reid.json')
    out = ROOT / 'test_results' / f'reid_cycles_{args.variant}'
    out.mkdir(parents=True, exist_ok=True)
    cfg.csv_path = str(out / 'events.csv')
    if hasattr(cfg, 'search_csv'):
        cfg.search_csv = str(out / 'frames.csv')
    sam = LiveTracker(ROOT / 'sam2/checkpoints/sam2.1_hiera_tiny.pt', image_size=512, cuda_graph=True)
    sam.warmup()
    detector = module.YOLOPersonDetector(cfg)
    cap = cv2.VideoCapture(str(ROOT / 'sam2/demo/data/gallery/05_default_juggle.mp4'))
    ok, base = cap.read()
    cap.release()
    assert ok
    h, w = base.shape[:2]
    # Manually specified visible person box in this fixed, reproducible source image.
    target = np.array([590, 40, 850, 680], dtype=float)
    trials = []
    for trial in range(20):
        manager = manager_type(sam, detector, cfg)
        stamp = trial * 20.
        manager.step(base, stamp, target.tolist())
        for j in range(1, 5):
            manager.step(base, stamp+j*.1)
        # Translate the whole source fully outside the field of view, then return.
        gone = cv2.warpAffine(base, np.float32([[1, 0, w], [0, 1, 0]]), (w, h))
        for j in range(5, 10):
            manager.step(gone, stamp+j*.1)
        scale = 1. if trial < 10 else 1.8 + (trial % 5)*.1
        tx = (trial % 5-2)*25 if trial < 10 else w/2 - 720*scale
        ty = 0 if trial < 10 else -30*scale
        returned = cv2.warpAffine(base, np.float32([[scale, 0, tx], [0, scale, ty]]), (w, h))
        truth = target * scale + np.array([tx, ty, tx, ty])
        truth[[0, 2]] = np.clip(truth[[0, 2]], 0, w-1)
        truth[[1, 3]] = np.clip(truth[[1, 3]], 0, h-1)
        entry = stamp+1.
        wall_entry = time.perf_counter()
        provisional = confirmed = None
        wrong = False
        for j in range(10, 70):  # six source seconds after return, right-censored failures
            result = manager.step(returned, stamp+j*.1)
            if result.get('valid') and provisional is None:
                provisional = stamp+j*.1-entry
            if result['status'] == 'TRACKING' and not result.get('provisional', False):
                confirmed = stamp+j*.1-entry
                wrong = module.iou(result['box'], truth) < .4
                break
        row = dict(trial=trial+1, session=manager.session, crop='full' if trial<10 else 'bottom_truncated',
                   confirmed_seconds=confirmed, provisional_seconds=provisional,
                   wall_seconds=time.perf_counter()-wall_entry, wrong=wrong,
                   success=confirmed is not None and not wrong,
                   first_candidate_seconds=None if manager.first_candidate is None else manager.first_candidate-entry,
                   gallery_size=len(manager.gallery))
        trials.append(row)
        print(args.variant, row, flush=True)
    times = [r['confirmed_seconds'] for r in trials if r['success']]
    report = dict(variant=args.variant, protocol='20 synthetic translated exits/returns, one fixed real video image; 10 full, 10 enlarged bottom-truncated; 10 Hz source timeline, no pacing; 6s observation after return',
                  trials=trials, success_rate=sum(r['success'] for r in trials)/20,
                  wrong_among_accepted=sum(r['wrong'] for r in trials)/max(1, sum(r['confirmed_seconds'] is not None for r in trials)),
                  median_seconds=float(np.median(times)) if times else None,
                  p90_seconds=float(np.percentile(times,90)) if times else None)
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='trials'}, indent=2))


if __name__ == '__main__':
    main()
