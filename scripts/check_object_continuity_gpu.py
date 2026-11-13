"""Small SAM GPU A-D procedural A/B. No existing/user video is opened."""
import csv
import json
import time
import cv2
import numpy as np
import torch
from live_tracker import LiveTracker,ROOT
from recovery.native_memory import NativeMemoryRecovery
from recovery.continuity import ObjectContinuityRecovery
from video_playback import FrameLog
from recovery.metrics import identity_metrics


def scene(case,index):
    frame=np.full((240,320,3),50,np.uint8)
    for x in range(0,320,16):
        cv2.line(frame,(x,0),(x,239),(60,65,70),1)
    visible=case!='B' or index<4 or index>=14
    tx=80+min(index,10)*2 if case=='C' else 80
    target=np.zeros((240,320),np.uint8)
    cv2.rectangle(target,(tx,70),(tx+14,170),1,-1)
    distractor=np.zeros_like(target)
    present=index>=4 and (case!='A' or index<14)
    if visible:
        frame[target>0]=(230,230,230)
        for y in range(82,160,12):
            cv2.line(frame,(tx+4,y),(tx+10,y),(55,65,80),2)
    if present:
        dx=max(65,160-index*6) if case=='C' else 103 if case=='D' else 175
        cv2.rectangle(distractor,(dx,80),(dx+24,160),1,-1)
        frame[distractor>0]=(230,230,230)
        for y in range(85,155,12):
            cv2.line(frame,(dx+3,y),(dx+20,y),(95,75,65),1)
    from sam_memory import mask_box
    truth=dict(frame=index,visible=visible,box=mask_box(target),distractors=[mask_box(distractor)] if present else [])
    return frame,truth


def main():
    out=ROOT/'test_results/object_continuity';out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(4)
    print(dict(torch=torch.__version__,cuda=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0)),flush=True)
    sam=LiveTracker(ROOT/'sam2/checkpoints/sam2.1_hiera_tiny.pt',image_size=512,cuda_graph=True)
    sam.warmup()
    reports=[]
    for case in ('A','B','C','D'):
        for name,cls in (('before',NativeMemoryRecovery),('after',ObjectContinuityRecovery)):
            tracker=cls(sam);path=out/f'{case}_{name}.csv';log=FrameLog(path)
            truth=[];timings=[]
            try:
                for i in range(22):
                    frame,gt=scene(case,i);truth.append(gt)
                    start=time.perf_counter()
                    result=tracker.step(frame,i/30,[78,68,96,172] if i==0 else None)
                    elapsed=time.perf_counter()-start;timings.append(elapsed)
                    log.write(0,(i,i/30,frame),result,elapsed)
            finally:
                log.handle.close()
            with path.open(encoding='utf-8') as handle:
                rows=list(csv.DictReader(handle))
            report=dict(case=case,policy=name,**identity_metrics(rows,truth),
                        tracking_percent=100*sum(r['status'] in ('TRACKING','SMALL','AMBIGUOUS') and bool(r['x1']) for r in rows)/len(rows),
                        sam_pipeline_fps=1/float(np.mean(timings[1:])),live_fps=None,
                        peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2)
            reports.append(report)
            (out/'report.json').write_text(json.dumps(reports,indent=2))
            print(json.dumps(report),flush=True)
    # Also exercise signed point prompts on the actual SAM decoder.
    frame,_=scene('A',5)
    sam.step(frame,2.,[78,68,96,172])
    masks,*_=sam.decode_points([[87,110],[185,110]],frame.shape[:2],labels=[1,0])
    assert masks.shape==(1,240,320)
    print('Signed SAM prompt smoke PASS; fixture metrics are not natural-camera accuracy.',flush=True)


if __name__=='__main__':
    main()
