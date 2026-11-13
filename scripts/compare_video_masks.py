"""Compare only matching source frames, not mismatched playback positions."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(reference, candidate):
    a,b = np.load(reference),np.load(candidate)
    ar,br = json.loads(str(a['rows'])),json.loads(str(b['rows']))
    amasks,bmasks = a['masks'],b['masks']
    lookup = {r['frame']:i for i,r in enumerate(ar)}
    if not np.array_equal(a['shape'],b['shape']):
        raise ValueError('Audit image dimensions differ')
    count = int(np.prod(a['shape']))
    ious,errors = [],[]
    for j,row in enumerate(br):
        if row['frame'] not in lookup:
            continue
        i = lookup[row['frame']]
        if ar[i]['box'] is None or row['box'] is None:
            continue
        am = np.unpackbits(amasks[i])[:count].astype(bool)
        bm = np.unpackbits(bmasks[j])[:count].astype(bool)
        ious.append(float(np.count_nonzero(am & bm)/max(1,np.count_nonzero(am | bm))))
        box1,box2 = np.array(ar[i]['box']),np.array(row['box'])
        errors.append(float(np.linalg.norm((box1[:2]+box1[2:]-box2[:2]-box2[2:])/2)))
    data = dict(reference_frames=len(ar),candidate_frames=len(br),matched_valid_frames=len(ious),
                candidate_tracking_percent=100*sum(r['status']=='TRACKING' for r in br)/max(1,len(br)),
                mean_iou=float(np.mean(ious)) if ious else None,p05_iou=float(np.percentile(ious,5)) if ious else None,
                mean_center_error_px=float(np.mean(errors)) if errors else None,
                p95_center_error_px=float(np.percentile(errors,95)) if errors else None,
                mean_age_ms=float(np.mean([r.get('age_ms',0.) for r in br])))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reference',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = compare(args.reference,args.candidate)
    print(json.dumps(result,indent=2))
    if args.output:
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
