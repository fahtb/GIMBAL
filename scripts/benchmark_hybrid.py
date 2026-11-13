"""Bounded real-video A/B performance benchmark. No GT => no accuracy claim."""
import argparse,json,time,statistics
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import torch
from source_camera_tracker import SourceCameraTracker
from small_target import SmallTargetTracker
from hybrid_tracker import HybridTracker

ROOT=Path(__file__).resolve().parents[1]

def distribution(values):
    return dict(p50=float(np.median(values)),p95=float(np.percentile(values,95)),mean=statistics.mean(values)) if values else None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',default='person')
    p.add_argument('--frames',type=int,default=80)
    p.add_argument('--image-size',type=int,choices=(512,1024),default=512)
    p.add_argument('--precision',choices=('bf16','fp16'),default='bf16')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    cases=json.loads((ROOT/'scripts/eval_cases.json').read_text())
    case=next(c for c in cases if c['name']==args.case)
    cv2.setNumThreads(4)
    cap=cv2.VideoCapture(str(ROOT/case['video']))
    if not cap.isOpened():raise RuntimeError('Video cannot be opened')
    fps=cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES,case['init_frame'])
    frames=[]
    try:
        for _ in range(args.frames):
            ok,f=cap.read()
            if not ok:break
            scale=case['resize_width']/f.shape[1]
            frames.append(cv2.resize(f,None,fx=scale,fy=scale))
    finally:cap.release()
    if not frames:raise RuntimeError('No decoded frames')
    x,y,w,h=case['init_bbox'];prompt=[x,y,x+w,y+h]
    started=time.perf_counter()
    base=SourceCameraTracker(image_size=args.image_size,precision=args.precision)
    base.warmup()
    warmup=time.perf_counter()-started
    report=dict(case=args.case,frames=len(frames),source_fps=fps,device=torch.cuda.get_device_name(),
                torch=torch.__version__,image_size=args.image_size,precision=str(base.dtype),
                cold_load_warmup_s=warmup,accuracy='not measured: no ground truth',results=[])
    for engine in ('baseline','enhanced'):
        tracker=(SmallTargetTracker(base,ROOT/'configs/small_target.json') if engine=='baseline' else
                 HybridTracker(base,ROOT/'configs/small_target.json',ROOT/'configs/hybrid.json'))
        torch.cuda.reset_peak_memory_stats()
        rows=[];memory=[]
        for i,f in enumerate(frames):
            tick=time.perf_counter()
            result=tracker.step(f,(case['init_frame']+i)/fps,prompt if i==0 else None)
            torch.cuda.synchronize()
            elapsed=(time.perf_counter()-tick)*1000
            rows.append(dict(frame=i,state=result['status'],valid=result.get('valid',False),
                             reason=result.get('transition_reason'),wall_ms=elapsed,sam_ms=result.get('inference_ms',0.),
                             memory_frames=result.get('recent_memory_count'),target_id=result.get('target_id'),
                             box=result.get('box'),kalman_innovation=result.get('kalman_innovation'),
                             flow_inlier_ratio=result.get('flow_inlier_ratio'),appearance=result.get('appearance_score')))
            if i%20==0 or i==len(frames)-1:memory.append(dict(frame=i,allocated_mb=torch.cuda.memory_allocated()/1048576))
        steady=rows[1:]
        entry=dict(engine=engine,steady_frames=len(steady),pipeline_ms=distribution([r['wall_ms'] for r in steady]),
                   sam_ms=distribution([r['sam_ms'] for r in steady]),state_frames=dict(Counter(r['state'] for r in rows)),
                   reason_frames=dict(Counter(r['reason'] for r in rows)),memory_samples=memory,
                   peak_allocated_mb=torch.cuda.max_memory_allocated()/1048576,
                   peak_reserved_mb=torch.cuda.max_memory_reserved()/1048576,rows=rows)
        entry['processing_hz']=1000/entry['pipeline_ms']['mean']
        report['results'].append(entry)
        tracker.close()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in entry.items() if k not in ('rows','memory_samples','reason_frames')}),flush=True)

if __name__=='__main__':main()
