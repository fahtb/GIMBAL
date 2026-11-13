"""Propose (never apply) a confidence cutoff from explicit C/W score labels."""
import argparse,csv,json
from pathlib import Path
import numpy as np


def calibrate(path):
    with path.open(encoding='utf-8') as handle:
        rows=list(csv.DictReader(handle))
    correct=[float(r['confidence']) for r in rows if r.get('confidence') and r['correct']=='1']
    wrong=[float(r['confidence']) for r in rows if r.get('confidence') and r['correct']=='0']
    report=dict(correct_samples=len(correct),wrong_samples=len(wrong),recommended_threshold=None,
                status='insufficient_labeled_classes',note='Event labels are sparse; validate any proposal on held-out clips.')
    for name,values in (('C',correct),('W',wrong)):
        report[name]=dict(p10=float(np.percentile(values,10)),median=float(np.median(values)),p90=float(np.percentile(values,90))) if values else None
    if min(len(correct),len(wrong))>=5:
        candidates=sorted(set(correct+wrong+[0.,1.]))
        best=min(candidates,key=lambda t:3*np.mean(np.array(wrong)>=t)+np.mean(np.array(correct)<t))
        report.update(status='proposal_only',recommended_threshold=best)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('labels',type=Path);args=p.parse_args()
    print(json.dumps(calibrate(args.labels),indent=2))
