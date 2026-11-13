"""Reproducible comparison; right-censored failures are not silently averaged away."""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
reports={}
summary={}
for variant in ('before','after'):
    folder=ROOT/'test_results'/f'reid_cycles_{variant}'
    report=json.loads((folder/'report.json').read_text())
    reports[variant]=report
    ids={r['session'] for r in report['trials']}
    csv_name='events_frames.csv' if variant=='before' else 'frames.csv'
    rows=[r for r in csv.DictReader((folder/csv_name).open()) if r['session'] in ids]
    detector=[float(r['detector_ms']) for r in rows if r['new_detection']=='True']
    events=[r for r in csv.DictReader((folder/'events.csv').open()) if r['session'] in ids]
    phases=[]
    for trial in report['trials']:
        group=[e for e in events if e['session']==trial['session']]
        first=next((float(e['source_time']) for e in group if e['event']=='first_candidate'),None)
        accepted=next((float(e['source_time']) for e in group if e['event'] in ('provisional','reacquire_candidate')),None)
        confirmed=next((float(e['source_time']) for e in group if e['event']=='reacquired'),None)
        if confirmed is not None:
            phases.append([accepted-first,confirmed-accepted])
    summary[variant]=dict(success_rate=report['success_rate'], median_success=report['median_seconds'],
        p90_success=report['p90_seconds'], detector_median_ms=float(np.median(detector)),
        detector_p90_ms=float(np.percentile(detector,90)), reasons=dict(Counter(r['reason'] for r in rows)),
        median_candidate_to_accept_seconds=float(np.median(np.asarray(phases)[:,0])),
        median_accept_to_confirm_seconds=float(np.median(np.asarray(phases)[:,1])))
lines=['# ReID before / after — controlled synthetic replay', '',
       '20 trials per version, same single-person video image. Ten full-body returns; ten enlarged bottom-truncated returns. Whole frame translated off-screen to simulate exit. Each trial starts with manual selection. No physical webcam trials or real distractor identities.', '',
       'Time below is SOURCE TIMELINE at 10 Hz, not wall-clock latency. Six seconds observation after return; failures are right-censored. Median/p90 of successful recoveries exclude failures.', '',
       '| Metric | Before | After |','|---|---:|---:|',
       f'| Success | {reports["before"]["success_rate"]:.0%} | {reports["after"]["success_rate"]:.0%} |',
       f'| Median confirmed, successful trials | {reports["before"]["median_seconds"]} s | {reports["after"]["median_seconds"]} s |',
       f'| p90 confirmed, successful trials | {reports["before"]["p90_seconds"]} s | {reports["after"]["p90_seconds"]} s |',
       '| p90 across ALL trials (censored) | '+ ' | '.join('>6 s' if r['success_rate'] < .9 else f'{r["p90_seconds"]:.3f} s' for r in reports.values())+' |',
       '| Wrong among accepted in this one-person fixture | '+ ' | '.join(str(sum(t['wrong'] for t in r['trials']))+'/'+str(sum(t['confirmed_seconds'] is not None for t in r['trials'])) for r in reports.values())+' |',
       '| Confirmed identity-switch rate with real distractors | NOT MEASURED | NOT MEASURED |','',
       'The wrong proxy checks overlap with the known transformed manual target box (IoU >=0.4), not human identity annotation. It cannot establish real-world false-reidentification rate.', '',
       '| Trial | Crop | Before confirmed (s) | After confirmed (s) |', '|---:|---|---:|---:|']
for a,b in zip(reports['before']['trials'],reports['after']['trials']):
    old='>6 (failed)' if a['confirmed_seconds'] is None else f'{a["confirmed_seconds"]:.3f}'
    new='>6 (failed)' if b['confirmed_seconds'] is None else f'{b["confirmed_seconds"]:.3f}'
    lines.append(f'| {a["trial"]} | {a["crop"]} | {old} | {new} |')
lines += ['', 'Stage measurements and rejection counts:', '', '```json', json.dumps(summary,indent=2), '```', '',
          'Per-trial reports include actual unpaced wall_seconds and exact session IDs to select relevant CSV rows across repeated runs. Provisional delay 0 source seconds means accepted on the first returning source frame; it still requires detector and SAM computation.']
(ROOT/'test_results/REID_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(ROOT/'test_results/reid_comparison.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
