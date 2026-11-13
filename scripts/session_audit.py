"""Audit observed state dwell times and lock events; never infer identity accuracy."""
import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


def read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def locked(row):
    if row.get('is_confirmed') not in ('',None) and str(row['is_confirmed']).lower() not in ('true','1'):
        return False
    if row.get('valid') not in ('',None):
        return str(row['valid']).lower() in ('true','1')
    return row.get('status') in ('TRACKING','TRACKING_LOW','SMALL','AMBIGUOUS') and row.get('x1') not in ('',None)


def analyze(rows):
    durations=defaultdict(float);counts=Counter();events=[]
    prior=None;had_lock=False;missing=False;start=None
    for row in rows:
        stamp=float(row['time_s']);state=row['status']
        counts[state]+=1
        if start is None:
            start=stamp
        reason=row.get('transition_reason') or ''
        manual=reason=='anchor_initialized' or row.get('selected_bbox') not in ('',None)
        if manual and prior and reason=='anchor_initialized' and prior.get('transition_reason')==reason and row.get('sam_seq') not in ('',None) and row.get('sam_seq')==prior.get('sam_seq'):
            manual=False
        reset=prior is not None and (stamp<float(prior['time_s']) or row.get('pass')!=prior.get('pass') or
              (row.get('generation') not in ('',None) and row.get('generation')!=prior.get('generation')))
        if prior is not None and not reset:
            durations[prior['status']]+=max(0.,stamp-float(prior['time_s']))
        def event(kind):
            events.append(dict(kind=kind,frame=int(row['frame']),time_s=stamp,relative_s=stamp-start,
                               before=prior['status'] if prior else '',after=state,reason=reason,
                               generation=row.get('generation',''),attempt_id=row.get('attempt_id','')))
        if reset or manual:
            had_lock=missing=False
            if reset:
                event('segment_boundary')
            if manual:
                event('manual_selection')
        if prior is None or state!=prior['status'] or reset:
            event('state_transition')
        if state!='SKIPPED':
            if locked(row):
                if missing and not manual and not reset:
                    event('lock_resumed')
                had_lock=True;missing=False
            elif had_lock and not missing:
                event('lock_lost');missing=True
        prior=row
    total=sum(durations.values())
    return dict(observed_seconds=total,states=[dict(state=s,frames=counts[s],seconds=durations[s],
                time_percent=100*durations[s]/total if total else 0.) for s in sorted(counts)],
                losses=sum(e['kind']=='lock_lost' for e in events),
                resumes=sum(e['kind']=='lock_resumed' for e in events),events=events,
                notes=['Durations hold each observed state until the next timestamp; final-frame duration unknown.',
                       'Lock loss/resumption is not a ground-truth identity or occlusion label.',
                       'Legacy CSV validity inferred from status and box where valid field is absent.'])


def audit(path,verbose=True):
    rows=read_rows(path)
    report=analyze(rows)
    sam_path=path.with_suffix('.sam.csv')
    sam=analyze(read_rows(sam_path)) if sam_path.exists() else None
    selections=read_rows(path.with_suffix('.selections.csv'))
    manual_frames={e['frame'] for e in (sam or report)['events'] if e['kind']=='manual_selection'}
    for item in selections:
        item['matches_initialization_in_this_csv']=int(item['frame']) in manual_frames
        item['reason']='not_recorded; user intent cannot be reconstructed'
    report=dict(display=report,sam=sam,selections=selections)
    out=path.with_suffix('.audit.json');out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    events=[dict(stream=stream,**event) for stream in ('display','sam') if report[stream]
            for event in report[stream]['events']]
    if events:
        with path.with_suffix('.events.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,events[0].keys());writer.writeheader();writer.writerows(events)
    states=report['display']['states']
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{80+len(states)*38}">',
         '<rect width="100%" height="100%" fill="white"/><g font-family="Arial" font-size="14">',
         '<text x="20" y="25">Observed display state durations (not tracking accuracy)</text>']
    for i,item in enumerate(states):
        y=55+i*38
        svg.append(f'<text x="15" y="{y+15}">{escape(item["state"])}</text><rect x="205" y="{y}" width="{item["time_percent"]*4.5:.1f}" height="23" fill="#347bc1"/><text x="675" y="{y+15}">{item["seconds"]:.2f}s ({item["time_percent"]:.1f}%)</text>')
    svg.append('</g></svg>');path.with_suffix('.states.svg').write_text(''.join(svg),encoding='utf-8')
    if verbose:
        print(json.dumps(dict(report=str(out),display_states=states,sam_losses=(sam or report['display'])['losses'],
                              sam_resumes=(sam or report['display'])['resumes'],selections=selections),ensure_ascii=False))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv',type=Path);args=p.parse_args();audit(args.csv)
