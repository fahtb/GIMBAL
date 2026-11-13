"""Wall-clock CUDA soak using generated images; records bounded-memory evidence."""
import argparse,csv,json,time
from types import SimpleNamespace
import numpy as np
import torch
from camera import create_tracker
from live_tracker import ROOT
from check_object_continuity_gpu import scene


def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=600)
    p.add_argument('--output',type=str,default=str(ROOT/'test_results/quarantine/soak.json'))
    args=p.parse_args()
    opts=SimpleNamespace(checkpoint=ROOT/'sam2/checkpoints/sam2.1_hiera_tiny.pt',memory=24,min_iou=.5,
        max_gap=1.,image_size=512,sam_only=True,target_mode='object',recovery='friend')
    wrapper,_=create_tracker(opts);t=wrapper.tracker
    frame,_=scene('D',5)
    start=time.perf_counter();last=start;count=0;samples=[];states={};times=[]
    try:
        while time.perf_counter()-start<args.seconds:
            now=time.perf_counter()
            r=t.step(frame,now,[78,68,96,172] if count==0 else None)
            count+=1;states[r['status']]=states.get(r['status'],0)+1
            times.append((time.perf_counter()-now)*1000)
            bank=t.sam.outputs
            assert len(bank['non_cond_frame_outputs'])<=24
            assert len(bank['cond_frame_outputs'])<=2*int(t.cfg['pinned_frames'])+2
            assert len(t.quarantine.entries)<=t.cfg['quarantine_frames']
            assert t.sam.anchor_index in bank['cond_frame_outputs']
            if now-last>=10 or count==1:
                samples.append(dict(seconds=now-start,allocated_mb=torch.cuda.memory_allocated()/2**20,
                    reserved_mb=torch.cuda.memory_reserved()/2**20,cond=len(bank['cond_frame_outputs']),
                    recent=len(bank['non_cond_frame_outputs']),quarantine=len(t.quarantine.entries),frames=count))
                last=now
            if now-start>30 and count%100==0:
                from pathlib import Path
                Path(args.output).parent.mkdir(parents=True,exist_ok=True)
                Path(args.output).with_suffix('.progress.json').write_text(json.dumps(samples[-1]))
    finally:
        wrapper.close()
    from pathlib import Path
    report=dict(elapsed_s=time.perf_counter()-start,frames=count,states=states,samples=samples,
                compute_mean_ms=float(np.mean(times[10:])),compute_p95_ms=float(np.percentile(times[10:],95)),
                longest_step_ms=max(times),stall_steps=sum(t>5000 for t in times),
                uninterrupted=not any(t>5000 for t in times))
    steady=[x for x in samples if x['seconds']>=60]
    report['allocated_growth_after_60s_mb']=steady[-1]['allocated_mb']-steady[0]['allocated_mb'] if steady else None
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='samples'}))


if __name__=='__main__':main()
