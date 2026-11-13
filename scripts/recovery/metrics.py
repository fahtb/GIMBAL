"""Sparse XYXY ground truth evaluation. Unknown visibility is never a success."""
import bisect
import numpy as np
from sam_memory import overlap


def truth_at(gt,frame):
    frames=[int(v['frame']) for v in gt]
    pos=bisect.bisect_left(frames,frame)
    if pos<len(gt) and frames[pos]==frame:
        return gt[pos]
    if pos==0 or pos==len(gt):
        return None
    a,b=gt[pos-1],gt[pos]
    if a['visible']!=b['visible']:
        return None  # Require explicit occlusion boundaries, not invented visibility.
    value=dict(a)
    if a['visible']:
        t=(frame-a['frame'])/(b['frame']-a['frame'])
        value['box']=(np.array(a['box'])*(1-t)+np.array(b['box'])*t).tolist()
    if len(a.get('distractors',[]))==len(b.get('distractors',[])):
        t=(frame-a['frame'])/(b['frame']-a['frame'])
        value['distractors']=[(np.array(x)*(1-t)+np.array(y)*t).tolist() for x,y in zip(a.get('distractors',[]),b.get('distractors',[]))]
    else:
        value['distractors']=[]
    return value


def identity_metrics(rows,gt):
    gt=sorted(gt,key=lambda v:v['frame'])
    lost=switches=correct=distractor_frames=labelled=0
    previous_correct=previous_wrong=False
    previous_visible=None
    return_time=None
    return_events=0
    delays=[]
    visible_count=valid_visible=ambiguous_visible=right_visible=0
    ious=[];errors=[];absent_frames=false_absent=0
    for row in rows:
        truth=truth_at(gt,int(row['frame']))
        if truth is None:
            previous_correct=previous_wrong=False
            continue
        labelled+=1
        box=[float(row[k]) for k in ('x1','y1','x2','y2')] if row.get('x1') not in ('',None) else None
        locked=row['status'] in ('TRACKING','TRACKING_LOW','SMALL','AMBIGUOUS','RECOVERED') and box is not None
        target=overlap(box,truth.get('box')) if truth['visible'] else 0.
        other=max([overlap(box,b) for b in truth.get('distractors',[])] or [0.])
        right=locked and truth['visible'] and target>=.3 and target>other
        wrong=locked and other>=.3 and other>target
        if truth['visible']:
            ious.append(float(target) if locked else 0.)
            if locked and truth.get('box') is not None:
                errors.append(float(np.linalg.norm((np.array(box[:2])+box[2:])/2-(np.array(truth['box'][:2])+truth['box'][2:])/2)))
            visible_count+=1
            valid_visible+=row.get('dx') not in ('',None) and row.get('dy') not in ('',None)
            ambiguous_visible+=row['status']=='AMBIGUOUS'
            right_visible+=bool(right)
        else:
            absent_frames+=1;false_absent+=bool(locked)
        switches+=bool(wrong and not previous_wrong)
        lost+=bool(truth['visible'] and previous_correct and not right and not wrong)
        if truth['visible'] and truth.get('distractors'):
            distractor_frames+=1;correct+=bool(right)
        if truth['visible'] and previous_visible is False:
            return_time=float(row['time_s'])
            return_events+=1
        if return_time is not None and right:
            delays.append(float(row['time_s'])-return_time);return_time=None
        previous_correct,previous_wrong,previous_visible=right,wrong,truth['visible']
    return dict(gt_frames=labelled,unnecessary_lost=lost,identity_switches=switches,
                mean_visible_iou=float(np.mean(ious)) if ious else None,
                mean_accepted_center_error_px=float(np.mean(errors)) if errors else None,
                success_iou50_percent=100*float(np.mean(np.array(ious)>=.5)) if ious else None,
                false_locks_when_absent=false_absent,annotated_absent_frames=absent_frames,
                correct_visible_percent=100*right_visible/visible_count if visible_count else None,
                ambiguous_visible_percent=100*ambiguous_visible/visible_count if visible_count else None,
                valid_visible_xy_percent=100*valid_visible/visible_count if visible_count else None,
                correct_with_distractor_percent=100*correct/distractor_frames if distractor_frames else None,
                recovery_median_s=float(np.median(delays)) if delays else None,
                recovery_p90_s=float(np.percentile(delays,90)) if delays else None,
                verified_recoveries=len(delays),return_events=return_events,
                recovery_success_percent=100*len(delays)/return_events if return_events else None,
                unrecovered_return=return_time is not None)
