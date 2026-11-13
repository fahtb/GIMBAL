"""Explicit human disambiguation; candidate masks stay tied to their SAM frame."""
import csv
from pathlib import Path
import cv2
import numpy as np

def mask_box(mask):
    y,x=np.nonzero(mask)
    return [int(x.min()),int(y.min()),int(x.max()),int(y.max())] if len(x) else None



def candidate_box(result,index):
    masks=(result or {}).get('search_masks',[])
    return mask_box(masks[index]) if 0<=index<len(masks) else None


def record_candidate(path,frame,index,box):
    from video_playback import record_selection
    record_selection(path,frame,box)
    path=Path(path).with_suffix('.choices.csv')
    exists=path.exists()
    with path.open('a',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle)
        if not exists:
            writer.writerow(('frame','candidate','reason','x1','y1','x2','y2'))
        writer.writerow((frame,index+1,'manual_identity_confirmation',*box))


def choose_worker(worker,index,path,snapshot=None):
    # Use the exact displayed snapshot, not whichever result finishes at keypress.
    result=snapshot
    box=candidate_box(result,index)
    if box is None:
        return False
    seq=result.get('candidate_seq',result['seq'])
    stamp=result.get('candidate_stamp',result['captured'])
    frame=result.get('candidate_frame',result['frame'])
    record_candidate(path,seq,index,box)
    worker.select((seq,stamp,frame),box)
    return True


def draw_candidates(view,result,hud_bottom=0):
    import numpy as np
    colors=((0,180,255),(255,140,40),(190,70,230),(60,220,160))
    for index,mask in enumerate(result.get('search_masks',[])[:9]):
        if mask.shape!=view.shape[:2]:
            continue
        contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        color=colors[index%len(colors)]
        cv2.drawContours(view,contours,-1,color,1)
        safe=mask.astype('uint8').copy()
        safe[:min(safe.shape[0],hud_bottom+18)]=0
        safe[:,:12]=0;safe[:,-12:]=0;safe[-12:]=0
        if safe.any():
            distance=cv2.distanceTransform(safe,cv2.DIST_L2,3)
            y,x=np.unravel_index(distance.argmax(),distance.shape)
            cv2.putText(view,str(index+1),(int(x)-5,int(y)+5),0,.65,(0,0,0),4)
            cv2.putText(view,str(index+1),(int(x)-5,int(y)+5),0,.65,color,2)


def record_feedback(path,frame,stamp,attempt,correct,applied,confidence=None):
    path=Path(path).with_suffix('.labels.csv')
    path.parent.mkdir(parents=True,exist_ok=True)
    exists=path.exists()
    legacy=False
    if exists:
        with path.open(encoding='utf-8') as handle:
            legacy='confidence' not in next(csv.reader(handle),[])
    with path.open('a',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle)
        if not exists:
            writer.writerow(('frame','time_s','attempt_id','correct','tracker_feedback_applied','confidence'))
        values=(frame,stamp,attempt,int(correct),int(bool(applied)))
        writer.writerow(values if legacy else (*values,confidence))
