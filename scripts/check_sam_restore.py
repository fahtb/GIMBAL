"""Interleaved A/B speed + output equivalence for the restored SAM-only tracker."""
import json
import time
import cv2
import numpy as np
import torch
from live_tracker import LiveTracker, ROOT


def main():
    trackers = [LiveTracker(ROOT / 'sam2/checkpoints/sam2.1_hiera_tiny.pt', image_size=512, cuda_graph=bool(i))
                for i in range(2)]
    assert trackers[1].cuda_graph_enabled, 'CUDA graph failed; inspect startup error'
    trackers[0].model.fast_mask_boxes = False
    trackers[0].model.cache_tracking_prompts = False
    trackers[0].cache_memory_scores = False
    for t in trackers:
        t.warmup()
    # Exact bounds for empty, disconnected, full and singleton masks.
    masks = torch.full((4, 512, 512), -1., device='cuda')
    masks[1, 20:490, 30:460] = 1
    masks[1, 2, 501] = 1
    masks[2] = 1
    masks[3, 511, 0] = 1
    assert trackers[0].model._samurai_mask_boxes(masks) == trackers[1].model._samurai_mask_boxes(masks)
    box_times = [[], []]
    for i in range(40):
        for j in (i % 2, 1 - i % 2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            trackers[j].model._samurai_mask_boxes(masks)
            torch.cuda.synchronize()
            if i >= 5:
                box_times[j].append((time.perf_counter() - start) * 1000)
    cap = cv2.VideoCapture(str(ROOT / 'sam2/demo/data/gallery/03_blocks.mp4'))
    ok, first = cap.read()
    assert ok
    times = [[], []]
    comparisons = 0
    for phase in ('static', 'video'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(45):
            if phase == 'video':
                ok, frame = cap.read()
                assert ok
            else:
                frame = first
            results = [None, None]
            for j in (i % 2, 1 - i % 2):
                results[j] = trackers[j].step(frame, i / 30, [483, 122, 583, 232] if i == 0 else None)
                if i >= 15 and results[j]['inference_ms'] > 0:
                    times[j].append(results[j]['inference_ms'])
            a, b = results
            assert a['status'] == b['status'] and a['box'] == b['box'], (phase, i)
            assert np.array_equal(a['mask'], b['mask']), (phase, i, 'mask changed')
            assert a['memory'] == b['memory'] <= 24
            comparisons += 1
    cap.release()
    report = dict(tests='exact bbox edge cases; interleaved static/video masks, boxes, loss status and bounded memory PASS',
                  image_size=512, frames_compared=comparisons, cuda_graph_enabled=trackers[1].cuda_graph_enabled,
                  baseline_bbox_ms=float(np.mean(box_times[0])), optimized_bbox_ms=float(np.mean(box_times[1])),
                  baseline_inference_ms=float(np.mean(times[0])), optimized_inference_ms=float(np.mean(times[1])),
                  optimized_p95_ms=float(np.percentile(times[1], 95)))
    report['memory_graph_replays'] = getattr(trackers[1].model.memory_attention, 'hits', 0)
    report['decoder_graph_replays'] = getattr(trackers[1].model.sam_mask_decoder.forward, 'hits', 0)
    assert report['decoder_graph_replays'] > 0, 'Decoder graph was never exercised'
    assert report['memory_graph_replays'] > 0, 'Full-memory graph was never exercised'
    report['speedup'] = report['baseline_inference_ms'] / report['optimized_inference_ms']
    (ROOT / 'test_results/sam_restore_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
