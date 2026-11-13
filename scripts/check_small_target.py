"""Same-frame A/B small-target tests. Synthetic GT is not natural-video accuracy."""
import json,time,argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from source_camera_tracker import SourceCameraTracker,ROOT
from small_target import SmallTargetTracker


WIDTH,HEIGHT=640,480

def scene(case,i):
    f=np.full((HEIGHT,WIDTH,3),45,np.uint8);gt=np.zeros((HEIGHT,WIDTH),bool);distractor=np.zeros_like(gt)
    size=60
    if case=='scale': size=round(60-54*(1-abs(i-23)/23))
    if case in ('lateral','occlusion','exit','same_color'): size=12
    x=200+(i*4 if case=='lateral' else 0);y=220
    visible=not(case in ('occlusion','exit','same_color') and 15<=i<29)
    if case=='exit' and i>=29: x=420
    def paint(x,y,size,wrong=False):
        texture=np.full((48,48,3),(90,190,220),np.uint8)
        if wrong:
            cv2.circle(texture,(24,24),16,(25,25,25),4)
        else:
            cv2.line(texture,(6,6),(40,40),(15,15,15),5)
            cv2.line(texture,(6,40),(40,6),(15,15,15),5)
        f[y:y+size,x:x+size]=cv2.resize(texture,(size,size),interpolation=cv2.INTER_AREA)
    if visible:
        paint(x,y,size);gt[y:y+size,x:x+size]=True
    if case=='same_color' and i>=10:
        paint(250,220,14,True);distractor[220:234,250:264]=True
    return f,gt,distractor,[x,y,x+size-1,y+size-1]


def main():
    global WIDTH,HEIGHT
    parser=argparse.ArgumentParser();parser.add_argument('--wide',action='store_true');args=parser.parse_args()
    if args.wide:WIDTH,HEIGHT=1280,720
    cv2.setNumThreads(4)
    base=SourceCameraTracker();base.warmup()
    reports=[]
    for case in ('near','scale','lateral','occlusion','exit','same_color'):
        for mode in ('off','auto'):
            base.reset();tracker=base if mode=='off' else SmallTargetTracker(base)
            rows=[];times=[]
            for i in range(47):
                f,gt,distractor,box=scene(case,i)
                start=time.perf_counter();r=tracker.step(f,i/30,box if i==0 else None);ms=(time.perf_counter()-start)*1000
                mask=r.get('mask');mask=mask if mask is not None else np.zeros_like(gt)
                assert mask.shape==gt.shape
                intersection=(mask&gt).sum();union=(mask|gt).sum();iou=float(intersection/max(1,union))
                wrong=bool(mask.any() and (not gt.any() or (mask&distractor).sum()>intersection))
                rows.append(dict(frame=i,visible=bool(gt.any()),iou=iou,wrong=wrong,status=r['status'],
                                 roi=r.get('crop_box'),pixels=r.get('model_target_pixels'),appearance=r.get('appearance_score'),reason=r.get('transition_reason')))
                if i>0: times.append(ms)
                if mode=='auto' and r['status']!='TRACKING': assert r['mask'] is None and r['dx'] is None
            visible=[v for v in rows if v['visible']]
            report=dict(case=case,mode=mode,correct_visible_percent=100*sum(v['iou']>=.5 for v in visible)/len(visible),
                        wrong_frames=sum(v['wrong'] for v in rows),mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),
                        max_allocated_mb=torch.cuda.max_memory_allocated()/2**20,frames=rows)
            reports.append(report);print({k:v for k,v in report.items() if k!='frames'},flush=True)
    out=ROOT/'test_results/small_target';out.mkdir(parents=True,exist_ok=True)
    (out/('ab_wide.json' if args.wide else 'ab.json')).write_text(json.dumps(reports,indent=2))
    base.close()

if __name__=='__main__': main()
