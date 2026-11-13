"""Actual SAM + class/DINO search and full-frame generic retrieval."""
import argparse
import csv
import json
from pathlib import Path
import time

import cv2

from camera import create_tracker, draw, TelemetryWriter, ROOT
from video_playback import run_video
from object_tracking import ObjectConfig
from object_search import ObjectSearch
from person_reid import iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dino-only', action='store_true')
    parser.add_argument('--unknown', action='store_true', help='hide COCO class to exercise DINO recovery end to end')
    args = parser.parse_args()
    cap = cv2.VideoCapture(str(ROOT/'videos/300651.mp4'))
    ok, frame = cap.read()
    cap.release()
    assert ok
    box = [1910,865,2480,1170]
    if args.dino_only:
        cfg = ObjectConfig.load(ROOT/'configs/object.json')
        worker = ObjectSearch(None, cfg, (frame,box), None)
        try:
            tick = time.perf_counter()
            worker.submit(frame, 0., box)
            deadline = tick + 120
            while worker.get() is None:
                assert time.perf_counter() < deadline, 'DINO search timeout'
                time.sleep(.01)
            result = worker.get()
            report = {k:v for k,v in result.items() if k != 'frame'}
            report['iou_to_selected'] = iou(result['box'],box) if result['box'] else 0.
            (ROOT/'test_results/object_dino_search.json').write_text(json.dumps(report,indent=2))
            print(report, flush=True)
            assert result['reason'] == 'qualified', report
            assert report['iou_to_selected'] >= .3, report
        finally:
            worker.close()
        print('DINOv2-small full-frame unknown-class retrieval PASS')
        return
    # Fixed image with black occlusion: tests reacquisition, not drone accuracy.
    path = ROOT/'test_results/object_occlusion.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30., (frame.shape[1],frame.shape[0]))
    assert writer.isOpened()
    for index in range(30):
        writer.write(frame*0 if 5 <= index < 9 else frame)
    writer.release()
    from types import SimpleNamespace
    options = SimpleNamespace(source=str(path), start_frame=0, init_frame=0, init_bbox=[1910,865,570,305],
        mode='offline', max_frames=30, resize_width=None, loop=False, no_display=True,
        save_video=ROOT/'test_results/object_occlusion_result.mp4', csv=ROOT/'test_results/object_occlusion.csv',
        telemetry=ROOT/'test_results/object_occlusion_target.json', checkpoint=ROOT/'sam2/checkpoints/sam2.1_hiera_tiny.pt',
        memory=24, min_iou=.5, max_gap=1., image_size=512, no_cuda_graph=False, no_reid=False,
        reid_config=ROOT/'configs/reid.json', object_config=ROOT/'configs/object.json', target_mode='auto')
    factory = create_tracker
    if args.unknown:
        options.csv = ROOT/'test_results/object_unknown_occlusion.csv'
        options.save_video = ROOT/'test_results/object_unknown_occlusion_result.mp4'
        def factory(options):
            tracker, detector = create_tracker(options)
            detector.detect_all = lambda *a, **k: []
            return tracker, detector
    run_video(options, factory, draw, TelemetryWriter)
    with options.csv.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    assert any(row['status'] == 'LOST' for row in rows)
    assert any(row['transition_reason'] == 'object_reacquired' for row in rows), 'Object never recovered'
    assert rows[-1]['status'] == 'TRACKING'
    print(f'CUDA SAM + asynchronous {"DINO" if args.unknown else "COCO"} search + same-frame verification PASS')


if __name__ == '__main__':
    main()
