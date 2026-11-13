"""Real capture/inference/detector threads on a paced synthetic exit/return clip."""
import json
import threading
import time
from types import SimpleNamespace
import cv2
import numpy as np
from camera import ROOT, LatestCapture, Worker


def main():
    cap = cv2.VideoCapture(str(ROOT/'sam2/demo/data/gallery/05_default_juggle.mp4'))
    ok, frame = cap.read()
    cap.release()
    assert ok
    h,w = frame.shape[:2]
    path = ROOT/'test_results/reid_threaded_exit.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (w,h))
    assert writer.isOpened()
    for index in range(150):
        writer.write(np.zeros_like(frame) if 40 <= index < 70 else frame)
    writer.release()
    cfg = json.loads((ROOT/'configs/reid.json').read_text())
    cfg.update(csv_path='test_results/threaded_events.csv',search_csv='test_results/threaded_frames.csv')
    cfg_path=ROOT/'test_results/threaded_config.json'
    cfg_path.write_text(json.dumps(cfg))
    args=SimpleNamespace(checkpoint=ROOT/'sam2/checkpoints/sam2.1_hiera_tiny.pt',
                         memory=24,min_iou=.5,max_gap=1.,image_size=512,reid_config=cfg_path)
    capture=LatestCapture(str(path))
    worker=Worker(capture,args)
    rows=[]
    labeled=False
    try:
        deadline=time.perf_counter()+90
        while not worker.ready or capture.get() is None:
            assert not worker.error,worker.error
            assert time.perf_counter()<deadline
            time.sleep(.01)
        worker.select(capture.get(),[590,40,850,680])
        last=-1
        while not capture.done:
            assert not worker.error,worker.error
            with worker.lock:
                result=worker.result
            if result and result['seq']>last:
                rows.append({k:v for k,v in result.items() if k not in ('frame','mask')})
                last=result['seq']
                if result['reacquired'] and not labeled:
                    worker.label(True)
                    labeled=True
            time.sleep(.005)
        states={r['status'] for r in rows}
        assert {'TRACKING','LOST','SEARCHING','REACQUIRE'} <= states,states
        assert any(r['reacquired']>=1 and r['confirmed_correct']==1 for r in rows)
        assert all(b['seq']>a['seq'] for a,b in zip(rows,rows[1:]))
        assert any(r['provisional'] and r['control_gain']==.25 for r in rows)
        report=dict(test='paced actual worker + dedicated person-search + C label',
                    camera_verified=False, states=sorted(states),
                    mean_latency_ms=float(np.mean([r['latency_ms'] for r in rows])),rows=rows)
        (ROOT/'test_results/reid_threaded_report.json').write_text(json.dumps(report,indent=2))
        print({k:v for k,v in report.items() if k!='rows'})
    finally:
        capture.stop.set()
        capture.thread.join(3)
        worker.thread.join(5)
    assert not any(t.name=='person-search' and t.is_alive() for t in threading.enumerate())
    print('Threaded search, same-frame detections, provisional gain, label routing, shutdown PASS')


if __name__=='__main__':
    main()
