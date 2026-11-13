"""Same generated inputs for resize/JPEG/precision A/B; no user video opened."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from camera import create_tracker
from live_tracker import ROOT
from check_object_continuity_gpu import scene
from recovery.native_memory import mask_agreement


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'test_results/quarantine/preprocess.json')
    args=p.parse_args();reports=[]
    for resize,jpeg,precision in (('linear',0,'fp16'),('bicubic',0,'fp16'),('linear',90,'fp16'),('linear',0,'bf16')):
        options=SimpleNamespace(checkpoint=ROOT/'sam2/checkpoints/sam2.1_hiera_tiny.pt',memory=24,min_iou=.5,max_gap=1.,image_size=512,
            sam_only=True,target_mode='object',recovery='friend',sam_resize=resize,sam_jpeg_quality=jpeg,sam_precision=precision)
        wrapped,_=create_tracker(options);times=[];agreements=[];valid=0;previous=None
        try:
            for i in range(32):
                frame,_=scene('D',i)
                rng=np.random.default_rng(i)
                frame=np.clip(frame.astype(float)+rng.normal(0,2,frame.shape),0,255).astype(np.uint8)
                tick=time.perf_counter();r=wrapped.tracker.step(frame,i/15,[78,68,96,172] if i==0 else None)
                times.append((time.perf_counter()-tick)*1000);valid+=bool(r.get('valid'))
                mask=r.get('mask')
                if mask is not None and previous is not None:agreements.append(mask_agreement(previous,mask))
                previous=mask
        finally:wrapped.close()
        report=dict(resize=resize,jpeg=jpeg,precision=precision,valid_frames=valid,total=32,
            pipeline_mean_ms=float(np.mean(times[3:])),pipeline_p95_ms=float(np.percentile(times[3:],95)),
            consecutive_mask_iou=float(np.mean(agreements)) if agreements else None)
        reports.append(report);print(json.dumps(report),flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(reports,indent=2))


if __name__=='__main__':main()
