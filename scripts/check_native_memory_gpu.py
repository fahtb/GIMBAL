"""GPU A/B on procedural same-color distractor; no input video is opened."""
import csv
import json
import time
import cv2
import numpy as np
import torch
from live_tracker import LiveTracker, ROOT
from sam_memory import SAMMemoryTracker
from recovery.native_memory import NativeMemoryRecovery


def scene(target=True, near=False):
    frame=np.full((480,640,3),(45,58,62),np.uint8)
    for x in range(0,640,20):
        cv2.line(frame,(x,0),(x,479),(62,70,76),2)
    # Thin white target versus a broad white distractor; color is identical.
    target_mask=np.zeros((480,640),np.uint8)
    cv2.rectangle(target_mask,(146,120),(160,240),1,-1)
    if target:
        frame[target_mask.astype(bool)]=(230,230,230)
        cv2.rectangle(frame,(149,126),(156,132),(80,100,85),-1)
    distractor=np.zeros_like(target_mask)
    polygon=np.array([[395,200],[415,200],[425,228],[465,236],[475,249],[470,260],[390,260],[386,249]],np.int32)
    if near:
        polygon -= np.array([250,70],np.int32)
    cv2.fillPoly(distractor,[polygon],1)
    frame[distractor.astype(bool)]=(230,230,230)
    return frame,target_mask.astype(bool),distractor.astype(bool)


def main():
    sam=LiveTracker(ROOT/'sam2/checkpoints/sam2.1_hiera_tiny.pt',image_size=512,cuda_graph=True)
    sam.warmup()
    rows=[]
    summary=[]
    for case,name,cls in [(case,name,cls) for case in ('separate','location_swap')
                         for name,cls in (('current',SAMMemoryTracker),('friend',NativeMemoryRecovery))]:
        manager=cls(sam)
        first_return=None
        wrong_frames=0
        wrong_attempts=set()
        for index in range(60):
            visible=index<10 or index>=35
            image,truth,distractor=scene(visible,near=case=='location_swap' and 14<=index<35)
            result=manager.step(image,index/30.,[142,116,165,245] if index==0 else None)
            mask=result.get('mask')
            wrong=bool(result.get('valid') and mask is not None and (mask&distractor).sum()>mask.sum()*.5)
            correct=bool(visible and result.get('valid') and mask is not None and (mask&truth).sum()/max(1,(mask|truth).sum())>.5)
            wrong_frames+=wrong
            if wrong:
                wrong_attempts.add(result.get('attempt_id',0))
            if index>=35 and correct and not result.get('probation') and first_return is None:
                first_return=(index-35)/30.
            row=dict(case=case,policy=name,frame=index,phase='visible' if index<10 else 'distractor_only' if index<35 else 'return',
                     wrong=wrong,correct=correct,**{k:result.get(k) for k in
                     ('status','transition_reason','reid_top1','lab_similarity','anchor_similarity','shape_ratio',
                      'anchor_mask_iou','identity_area_ratio','search_top2','reacquired','attempt_id','pipeline_ms')})
            rows.append(row)
        summary.append(dict(case=case,policy=name,wrong_frames=wrong_frames,wrong_attempts=len(wrong_attempts),
                            return_seconds=first_return,reacquired=manager.reacquired,
                            mean_pipeline_ms=float(np.mean([r['pipeline_ms'] for r in rows if r['policy']==name and r['case']==case]))))
        print(summary[-1],flush=True)
    out=ROOT/'test_results/native_memory'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'same_color.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,rows[0].keys());writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    cv2.imwrite(str(out/'fixture.png'),scene()[0])
    assert all(s['wrong_frames']==0 for s in summary if s['policy']=='friend'),'New policy accepted same-color distractor'
    assert all(s['return_seconds'] is not None for s in summary if s['policy']=='friend'),'New policy did not resume the original object'
    print('Native memory GPU A/B PASS (procedural fixture, not user camera accuracy)')


if __name__=='__main__':
    main()
