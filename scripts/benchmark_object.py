"""Serial, reproducible legacy/auto comparisons on drone and person clips."""
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def summarize(path):
    with path.open(newline='', encoding='utf-8') as handle:
        rows = [r for r in csv.DictReader(handle) if r['status'] != 'SKIPPED']
    tracking = sum(r['status'] == 'TRACKING' for r in rows)
    lost = recovered = 0
    for before, after in zip(rows, rows[1:]):
        lost += before['status'] == 'TRACKING' and after['status'].startswith('LOST')
        recovered += before['status'] in ('LOST','SEARCHING','REACQUIRE') and after['status'] == 'TRACKING'
    seconds = sum(float(r['latency_ms']) / 1000 for r in rows)
    return dict(frames=len(rows), tracking_percent=100*tracking/max(1,len(rows)), lost=lost,
                reacquired=recovered, processing_fps=len(rows)/max(1e-9,seconds), csv=str(path))


def main():
    output = ROOT/'test_results/object_benchmark'
    output.mkdir(parents=True, exist_ok=True)
    fixture = ROOT/'test_results/reid_threaded_exit.avi'
    if not fixture.exists():
        import cv2
        cap = cv2.VideoCapture(str(ROOT/'sam2/demo/data/gallery/05_default_juggle.mp4'))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError('Missing person regression source video')
        writer = cv2.VideoWriter(str(fixture), cv2.VideoWriter_fourcc(*'MJPG'), 30.,
                                 (frame.shape[1],frame.shape[0]))
        if not writer.isOpened():
            raise RuntimeError('Cannot create person regression fixture')
        try:
            for index in range(150):
                writer.write(frame*0 if 40 <= index < 70 else frame)
        finally:
            writer.release()
    report = []
    cases = [('drone', 'videos/300651.mp4', '1910,865,570,305', 988),
             ('person_occlusion', 'test_results/reid_threaded_exit.avi', '590,40,260,640', 150)]
    for name, source, box, count in cases:
        for mode in ('person','auto'):
            path = output/f'{name}_{mode}.csv'
            command = [sys.executable, str(ROOT/'scripts/camera.py'), '--source', str(ROOT/source),
                       '--mode','offline','--init-frame','0','--init-bbox',box,'--no-display',
                       '--max-frames',str(count),'--target-mode',mode,'--csv',str(path),
                       '--telemetry',str(output/f'{name}_{mode}_target.json')]
            print(f'BENCHMARK {name} {mode}', flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            data = dict(clip=name, mode=mode, command=command, **summarize(path))
            report.append(data)
            (output/'summary.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
            print(json.dumps({k:v for k,v in data.items() if k not in ('command','csv')}), flush=True)
    # Compare exact frame outputs for person regression, ignoring timing.
    with (output/'person_occlusion_person.csv').open() as a, (output/'person_occlusion_auto.csv').open() as b:
        baseline, current = list(csv.DictReader(a)), list(csv.DictReader(b))
    keys = ('status','x1','y1','x2','y2','iou','mask_area','transition_reason')
    assert len(baseline) == len(current)
    assert all(all(a[k] == b[k] for k in keys) for a,b in zip(baseline,current)), 'PERSON behavior changed'
    assert report[1]['tracking_percent'] >= 80
    assert report[-1]['reacquired'] >= 1
    print('Drone threshold and exact PERSON regression PASS', flush=True)


if __name__ == '__main__':
    main()
