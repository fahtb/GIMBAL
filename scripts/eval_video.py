"""Headless evaluation of the source camera predictor; no legacy A/B policies."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def summarize(path, case):
    rows = [row for row in csv.DictReader(path.open(encoding='utf-8')) if row['status'] != 'SKIPPED']
    visible = case.get('visible_ranges')
    considered = [row for row in rows if not visible or any(a <= int(row['frame']) <= b for a,b in visible)]
    accepted = ('TRACKING','TRACKING_LOW', 'SMALL', 'RECOVERED')
    def mean(name):
        values = [float(row[name]) for row in rows if row.get(name) not in ('', None)]
        return statistics.mean(values) if values else None
    from session_audit import analyze
    audit=analyze(rows)
    lost=audit['losses']
    sam_times = [1000/float(row['sam_fps']) for row in rows if float(row.get('sam_fps') or 0) > 0]
    result = dict(frames=len(rows), tracking_percent=100*sum(row['status'] in accepted for row in considered)/max(1,len(considered)),
                  visibility_annotated=bool(visible), lost=lost,
                  reacquired=max([int(row.get('reacquired') or 0) for row in rows] or [0]),
                  live_fps=mean('live_fps'), sam_compute_fps=1000/statistics.mean(sam_times) if sam_times else 0,
                  processing_fps=1000/mean('latency_ms') if mean('latency_ms') else None,
                  age_ms=mean('age_ms'), false_reacquisitions=None, recovery_success_rate=None,
                  lock_resumes=audit['resumes'],events=audit['events'])
    # Reappearance entries explicitly annotate the interval and the true box at
    # the measured accepted frame. Without identity annotations, report only
    # candidate timing, never invent an accuracy/success/false-positive number.
    events = case.get('reappearances') or []
    timings = []
    for event in events:
        match = next((row for row in rows if int(row['frame']) >= event['frame'] and row['status'] in accepted), None)
        timings.append(dict(frame=event['frame'], accepted_frame=int(match['frame']) if match else None,
                            delay_s=float(match['time_s'])-event['time_s'] if match else None,
                            identity_verified=False))
    result['reappearance_candidates'] = timings
    import numpy as np
    result['valid_xy_percent']=100*sum(r.get('dx') not in ('',None) and r.get('dy') not in ('',None) for r in considered)/max(1,len(considered))
    from collections import Counter
    result['state_frames']=dict(Counter(r['status'] for r in rows))
    result['reason_frames']=dict(Counter(r.get('transition_reason','') for r in rows))
    result['state_time_seconds']={}
    for a,b in zip(rows,rows[1:]):
        state=a['status'];dt=max(0.,float(b['time_s'])-float(a['time_s']))
        result['state_time_seconds'][state]=result['state_time_seconds'].get(state,0.)+dt
    for field in ('confidence','pipeline_ms','age_ms','latency_ms','gpu_core_ms','flow_mask_iou','continuity_distance'):
        values=[float(r[field]) for r in rows if r.get(field) not in ('',None)]
        result[field+'_distribution']=dict(median=float(np.median(values)),p95=float(np.percentile(values,95))) if values else None
    if case.get('gt'):
        from recovery.metrics import identity_metrics
        gt=case['gt']
        if isinstance(gt,str):
            gt=json.loads((ROOT/gt).read_text(encoding='utf-8'))
        result.update(identity_metrics(rows,gt))
    labels = case.get('reacquisition_labels') or {}
    if labels:
        result['false_reacquisitions'] = sum(not correct for correct in labels.values())
        result['labelled_reacquisitions'] = len(labels)
        result['false_reacquisition_percent'] = 100*result['false_reacquisitions']/len(labels)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', type=Path, default=Path(__file__).with_name('eval_cases.json'))
    p.add_argument('--case', help='one case name; default all')
    p.add_argument('--variants', default='source', choices=('source',), help='legacy variants removed from main tracking')
    p.add_argument('--recoveries', default='source', choices=('source',))
    p.add_argument('--mode', choices=('offline','realtime'), default='offline')
    p.add_argument('--small-target',choices=('auto','off'),default='auto')
    p.add_argument('--engine',choices=('baseline','enhanced','both'),default='baseline')
    p.add_argument('--max-frames', type=int)
    p.add_argument('--output', type=Path, default=ROOT/'runtime/sam_eval')
    args = p.parse_args()
    cases = json.loads(args.cases.read_text())
    selected = [case for case in cases if args.case is None or case['name'] == args.case]
    if not selected:
        p.error('Unknown case')
    variants = ['baseline','enhanced'] if args.engine=='both' else [args.engine]
    recoveries = ['source']
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for case in selected:
        from itertools import product
        for variant,recovery in product(variants,recoveries):
            stem = args.output/f'{case["name"]}_{recovery}_v{variant}_{args.mode}_{args.small_target}'
            path = stem.with_suffix('.csv')
            cmd = [sys.executable, str(ROOT/'scripts/camera.py'), '--source', str(ROOT/case['video']),
                   '--mode', args.mode, '--no-display', '--resize-width', str(case['resize_width']),
                   '--init-frame', str(case['init_frame']), '--init-bbox', ','.join(map(str,case['init_bbox'])),
                   '--max-frames', str(args.max_frames or case['max_frames']), '--csv', str(path),
                   '--profile', str(stem.with_suffix('.profile.json')),
                   '--telemetry', str(stem.with_suffix('.telemetry.json'))]
            cmd += ['--sam-only','--small-target',args.small_target,'--engine',variant]
            with stem.with_suffix('.log').open('w', encoding='utf-8') as log:
                process = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            result = dict(case=case['name'], variant=variant,recovery=recovery,exit_code=process.returncode, csv=str(path))
            if process.returncode == 0:
                result.update(summarize(path, case))
                profile_path = stem.with_suffix('.profile.json')
                if profile_path.exists() and args.mode == 'realtime':
                    result['live_fps'] = json.loads(profile_path.read_text())['live_fps']
                    # Frame/display-loop latency is not the inference rate.
                    result['processing_fps']=None
                    worker_path=path.with_suffix('.sam.csv')
                    if worker_path.exists():
                        import numpy as np
                        worker_rows=list(csv.DictReader(worker_path.open(encoding='utf-8')))
                        result['worker_updates']=len(worker_rows)
                        result['worker_timing']={}
                        for key in ('queue_ms','worker_compute_ms','sam_latency_ms'):
                            values=[float(r[key]) for r in worker_rows if r.get(key) not in ('',None)]
                            result['worker_timing'][key]=dict(p50=float(np.median(values)),p95=float(np.percentile(values,95))) if values else None
                elif args.mode == 'offline':
                    result['live_fps'] = None
            report.append(result)
            (args.output/'report.json').write_text(json.dumps(report, indent=2))
            print(json.dumps({k:result.get(k) for k in ('case','variant','exit_code','tracking_percent','reacquired','live_fps')}), flush=True)


if __name__ == '__main__':
    main()
