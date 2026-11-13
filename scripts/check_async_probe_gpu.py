"""Short isolated SAM-probe A/B on generated frames; no input video opened."""
import json,time
from types import SimpleNamespace
import numpy as np
import torch
from camera import create_tracker
from live_tracker import ROOT
from recovery.continuity import ObjectContinuityRecovery
from check_object_continuity_gpu import scene


def main():
    args=SimpleNamespace(checkpoint=ROOT/'sam2/checkpoints/sam2.1_hiera_tiny.pt',memory=24,
        min_iou=.5,max_gap=1.,image_size=512,sam_only=True,target_mode='object',recovery='friend',timing_details=True)
    wrapped,_=create_tracker(args);after=wrapped.tracker
    before=ObjectContinuityRecovery(after.sam)
    report=[]
    accuracy=[]
    try:
        # Verifier must reuse supplied features and must not call its encoder.
        def forbidden(*a,**k):
            raise AssertionError('Probe attempted a second image encoder')
        after.probe_worker.tracker.sam.backbone=forbidden
        for name,tracker in (('sync',before),('async',after)):
            times=[];cores=[];valid=0;states={}
            for i in range(24):
                frame,_=scene('D',i)
                tick=time.perf_counter()
                r=tracker.step(frame,i/30,[78,68,96,172] if i==0 else None)
                times.append((time.perf_counter()-tick)*1000)
                cores.append(r.get('sam_core_wall_ms',0));valid+=bool(r.get('valid'))
                states[r['status']]=states.get(r['status'],0)+1
            report.append(dict(policy=name,valid_frames=valid,total=24,states=states,
                pipeline_mean_ms=float(np.mean(times[1:])),pipeline_p95_ms=float(np.percentile(times[1:],95)),
                sam_core_p95_ms=float(np.percentile(cores[1:],95)),peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2))
            print(json.dumps(report[-1]),flush=True)
        after.probe_worker.poll()
        from recovery.metrics import identity_metrics
        for case in ('A','B','C','D'):
            rows=[];truth=[]
            for i in range(22):
                frame,gt=scene(case,i);truth.append(gt)
                r=after.step(frame,i/30,[78,68,96,172] if i==0 else None)
                row=dict(frame=i,time_s=i/30,status=r['status'],dx=r.get('dx'),dy=r.get('dy'))
                row.update(zip(('x1','y1','x2','y2'),r.get('box') or [None]*4))
                rows.append(row)
            accuracy.append(dict(case=case,**identity_metrics(rows,truth)))
            print(json.dumps(accuracy[-1]),flush=True)
    finally:
        wrapped.close()
    out=ROOT/'test_results/identity_confidence';out.mkdir(parents=True,exist_ok=True)
    (out/'async_probe.json').write_text(json.dumps(report,indent=2))
    (out/'procedural_identity.json').write_text(json.dumps(accuracy,indent=2))
    assert report[1]['valid_frames']>=20,'Async verifier discarded too much continuous evidence'


if __name__=='__main__':
    main()
