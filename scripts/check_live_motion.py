"""Known-motion, delayed-anchor, occlusion and bounded-history regression tests.

Synthetic anchors are ground-truth masks, NOT model predictions. A separate
check_live_pipeline.py measures the complete CUDA SAMURAI + motion pipeline.
"""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from live_motion import MotionTracker, box_iou


def scene(i):
    rng = np.random.default_rng(42)
    texture = rng.integers(30, 225, (70, 60, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    frame = np.full((240, 320, 3), 30, np.uint8)
    # Similar distractor, distinct location; it must not become the target.
    frame[150:220, 240:300] = texture
    x, y = 25 + i * 2, 45 + round(5 * np.sin(i / 9))
    mask = np.zeros((240, 320), bool)
    if not 65 <= i < 72:
        frame[y:y + 70, x:x + 60] = texture
        mask[y:y + 70, x:x + 60] = True
    return frame, mask, [x, y, x + 60, y + 70]


def anchor(i, generation=1):
    frame, mask, box = scene(i)
    return dict(frame=frame, mask=mask, box=box if mask.any() else None, seq=i,
                captured=i / 30, inference_ms=400., generation=generation)


def main():
    cv2.setNumThreads(1)
    tracker = MotionTracker()
    tracker.reset(1)
    errors, old_errors, timings, ious = [], [], [], []
    loss_at = None
    for i in range(80):
        frame, _, truth = scene(i)
        sam = anchor(((i - 12) // 6) * 6) if i >= 12 else None
        r = tracker.step((i, i / 30, frame), sam)
        assert r['history_frames'] <= 45
        if 12 <= i < 65:
            assert r['valid'], (i, r['status'])
            center = (np.array(r['box'][:2]) + r['box'][2:]) / 2
            true_center = (np.array(truth[:2]) + truth[2:]) / 2
            old_center = (np.array(sam['box'][:2]) + sam['box'][2:]) / 2
            errors.append(float(np.linalg.norm(center - true_center)))
            old_errors.append(float(np.linalg.norm(old_center - true_center)))
            timings.append(r['flow_ms'])
            ious.append(box_iou(r['box'], truth))
        if i >= 65 and not r['valid'] and loss_at is None:
            loss_at = i
        if i >= 72:
            assert not r['valid'], 'Must not reacquire automatically'
    assert loss_at is not None and loss_at <= 67, loss_at
    assert np.mean(errors) < 3 and np.mean(ious) > .9
    tracker.reset(2)
    frame, _, _ = scene(80)
    rejected = tracker.step((80, 80 / 30, frame), anchor(80, 1))
    assert not rejected['valid'], 'Old selection must be ignored'
    restored = tracker.step((80, 80 / 30, frame), anchor(80, 2))
    assert restored['valid']
    # Long capture gap is unsafe even when an old anchor still exists.
    assert not tracker.step((100, 100 / 30, scene(80)[0]))['valid']
    # Expire stale semantic confirmation while texture is still trackable.
    tracker = MotionTracker(max_sam_age=.3)
    tracker.reset(1)
    for i in range(12):
        expired = tracker.step((i, i / 30, scene(i)[0]), anchor(0) if i == 0 else None)
    assert not expired['valid']
    # A confident but substantially enlarged SAM box must not pull the live
    # target onto an adjacent object. This reproduces a measured model drift.
    tracker = MotionTracker()
    tracker.reset(1)
    for i in range(6):
        tracker.step((i, i / 30, scene(i)[0]), anchor(0) if i == 0 else None)
    wrong = anchor(6)
    wrong['box'] = [wrong['box'][0] - 25, wrong['box'][1] - 25,
                    wrong['box'][2] + 25, wrong['box'][3] + 25]
    rejected = tracker.step((6, 6 / 30, scene(6)[0]), wrong)
    assert not rejected['valid'], 'Reject SAM drift instead of moving the live target'
    report = dict(tests='delayed-anchor alignment, similar distractor, occlusion, loss latch, reselection, gap, expiry, SAM disagreement, bounded history PASS',
                  simulated_sam_delay_ms=400, mean_center_error_px=float(np.mean(errors)),
                  held_old_box_mean_error_px=float(np.mean(old_errors)),
                  mean_box_iou=float(np.mean(ious)), mean_flow_ms=float(np.mean(timings)),
                  p95_flow_ms=float(np.percentile(timings, 95)), occlusion_at=65, loss_at=loss_at)
    root = Path(__file__).resolve().parents[1]
    (root / 'test_results/live_motion_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
