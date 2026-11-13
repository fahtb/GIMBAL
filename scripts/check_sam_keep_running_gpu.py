"""Controlled occlusion diagnostic; not evidence for natural reappearance.

Reuses one source frame for 10 visible, 5 black and 15 restored frames.
Writes per-frame raw SAM observations for stopped and keep-running modes.
"""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from live_tracker import LiveTracker, ROOT
from sam_only import SAMOnlyTracker


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--init-bbox', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--memory-recovery', action='store_true', help='compare stopped diagnostic to SAM memory recovery')
    args = p.parse_args()
    cap = cv2.VideoCapture(args.source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError('Cannot read first source frame')
    frame = cv2.resize(frame, (960, round(frame.shape[0]*960/frame.shape[1])))
    x, y, w, h = map(float, args.init_bbox.split(','))
    sam = LiveTracker(ROOT / 'sam2/checkpoints/sam2.1_hiera_tiny.pt', image_size=512, cuda_graph=True)
    sam.warmup()
    rows, summary = [], []
    for keep in (False, True):
        tracker = SAMOnlyTracker(sam, keep_running=keep)
        if keep and args.memory_recovery:
            from sam_memory import SAMMemoryTracker
            tracker = SAMMemoryTracker(sam)
        for index in range(30):
            image = np.zeros_like(frame) if 10 <= index < 15 else frame
            result = tracker.step(image, index/30, [x, y, x+w, y+h] if index == 0 else None)
            rows.append(dict(keep_running=keep, frame=index,
                             phase='black' if 10 <= index < 15 else 'restored' if index >= 15 else 'visible',
                             **{key: result.get(key) for key in ('status', 'predictor_called',
                                'object_score_logits', 'predicted_iou', 'mask_area', 'sam_observation_valid', 'sam_reason')}))
        restored = [r for r in rows if r['keep_running'] == keep and r['phase'] == 'restored']
        summary.append(dict(keep_running=keep, calls_restored=sum(bool(r['predictor_called']) for r in restored),
                            reacquired=getattr(tracker, 'reacquired', 0),
                            positive_object_frames=sum((r['object_score_logits'] or 0) > 0 for r in restored),
                            valid_observations=sum(bool(r['sam_observation_valid']) for r in restored)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
