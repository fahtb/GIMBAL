"""Source port integration checks; generated fixture is not an accuracy benchmark."""
import argparse, hashlib, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from camera import create_tracker, LatestCapture, Worker, ROOT


def frame(i=0):
    out=np.full((240,320,3),45,np.uint8)
    x=70+min(i,30)
    cv2.rectangle(out,(x,60),(x+45,180),(50,180,230),-1)
    for y in range(70,180,15): cv2.line(out,(x+6,y),(x+38,y),(15,40,70),2)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--camera',action='store_true');args=p.parse_args()
    out=ROOT/'test_results/source_camera';out.mkdir(parents=True,exist_ok=True)
    opts=SimpleNamespace(checkpoint=ROOT/'sam2/checkpoints/sam2.1_hiera_tiny.pt',image_size=512,
                         sam_only=True,small_target='off',source='0',csv=out/'camera.csv',sam_precision='bf16')
    if args.camera:
        cap=LatestCapture(0,dict(backend='msmf',width=640,height=480,fps=30,mjpg=False))
        worker=Worker(cap,opts)
        def wait(predicate, seconds=120):
            end=time.perf_counter()+seconds
            while not predicate():
                if worker.error or cap.error: raise RuntimeError(worker.error or cap.error)
                if time.perf_counter()>end: raise TimeoutError('Camera worker timeout')
                time.sleep(.02)
        try:
            wait(lambda:worker.ready and cap.get() is not None)
            pkt=cap.get();h,w=pkt[2].shape[:2];worker.select(pkt,[w*.35,h*.25,w*.65,h*.8])
            wait(lambda:worker.result is not None)
            start=time.perf_counter();seq=cap.get()[0];time.sleep(3)
            result=worker.result;fps=(cap.get()[0]-seq)/(time.perf_counter()-start)
            worker.clear();wait(lambda:worker.command is None)
            worker.select(cap.get(),[w*.4,h*.3,w*.6,h*.7]);wait(lambda:worker.result is not None and worker.result['generation']==worker.generation)
            report=dict(camera_fps=fps,sam_ms=result['inference_ms'],generation=worker.generation,
                        mask_shape=list(worker.result['mask'].shape) if worker.result['mask'] is not None else None)
        finally:
            cap.stop.set();cap.thread.join(10);worker.thread.join(30)
        assert not cap.thread.is_alive() and not worker.thread.is_alive()
        reopened=cv2.VideoCapture(0,cv2.CAP_MSMF);report['released_reopened']=reopened.isOpened();reopened.release()
        (out/'camera_report.json').write_text(json.dumps(report,indent=2));print(report);return
    tracker,_=create_tracker(opts)
    times=[]
    try:
        for i in range(55):
            r=tracker.step(frame(i),i/30,[68,58,118,182] if i==0 else None)
            assert r['valid'] and r['mask'].shape==(240,320)
            if i>3: times.append(r['inference_ms'])
        assert tracker.predictor.frame_idx==54
        state=tracker.predictor.condition_state
        assert len(state['images'])==1
        assert len(state['output_dict']['non_cond_frame_outputs'])<=tracker.predictor.num_maskmem
        tracker.step(frame(),2.,[68,58,118,182]);assert tracker.predictor.frame_idx==0
        tracker.step(frame(1),2.03);assert tracker.predictor.frame_idx==1
        tracker.reset();assert tracker.step(frame(),3.)['status']=='WAIT_SELECTION'
        assert tracker.step(frame(),3.1,[90,120,90,120])['valid']
        assert tracker.label(True,3.1) and tracker.label(False,3.1)
        forbidden=['live_tracker','sam_memory','sam_crop','recovery.continuity','recovery.native_memory','person_reid']
        assert not any(x in sys.modules for x in forbidden),[x for x in forbidden if x in sys.modules]
        vendor=ROOT/'vendor/source_camera';manifest=json.loads((vendor/'SOURCE_MANIFEST.json').read_text())
        for name,digest in manifest['files'].items():
            assert hashlib.sha256((vendor/name).read_bytes()).hexdigest()==digest,name
        writer=cv2.VideoWriter(str(out/'generated.avi'),cv2.VideoWriter_fourcc(*'MJPG'),30,(320,240))
        assert writer.isOpened()
        for i in range(120): writer.write(frame(i))
        writer.release()
        report=dict(frames=55,mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),
                    cuda=torch.cuda.get_device_name(),torch=torch.__version__,memory_frames=tracker.predictor.num_maskmem,
                    source_hashes_verified=len(manifest['files']),old_modules_loaded=False,reset_box_point_feedback=True,
                    max_allocated_mb=torch.cuda.max_memory_allocated()/2**20)
        (out/'smoke.json').write_text(json.dumps(report,indent=2));print(report)
    finally: tracker.close()

if __name__=='__main__': main()
