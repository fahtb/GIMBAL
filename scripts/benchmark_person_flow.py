"""Compare non-rigid mask transport at a fixed three-source-frame SAM delay."""
import json
from pathlib import Path
import time
import cv2
import numpy as np


def main():
    cv2.setNumThreads(4)
    archive=np.load('test_results/smooth_natural_reference.npz')
    masks=np.unpackbits(archive['masks'],axis=1)[:,:540*960].reshape(-1,540,960).astype('float32')
    cap=cv2.VideoCapture('test_results/person_natural_120.avi')
    grays=[]
    for i in range(120):
        ok,frame=cap.read()
        assert ok
        grays.append(cv2.cvtColor(cv2.resize(frame,(960,540),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)[:,300:750].copy())
    cap.release()
    yy,xx=np.mgrid[:540,300:750].astype(np.float32)
    report={}
    for kind in ('dis_fast_roi','dis_medium_roi','farneback_roi'):
        engine=cv2.DISOpticalFlow_create(1 if kind=='dis_fast_roi' else 2)
        flows,times=[],[]
        for i in range(1,120):
            start=time.perf_counter()
            flow=(cv2.calcOpticalFlowFarneback(grays[i],grays[i-1],None,.5,3,21,3,5,1.2,0)
                  if kind=='farneback_roi' else engine.calc(grays[i],grays[i-1],None))
            times.append((time.perf_counter()-start)*1000)
            flows.append((xx+flow[:,:,0],yy+flow[:,:,1]))
        scores=[]
        for i in range(120):
            start=max(0,i-3)
            mask=masks[start].copy()
            for k in range(start,i):
                out=np.zeros_like(mask)
                out[:,300:750]=cv2.remap(mask,*flows[k],cv2.INTER_LINEAR)
                mask=out
            predicted,truth=mask>=.5,masks[i]>0
            scores.append(float((predicted & truth).sum()/(predicted | truth).sum()))
        report[kind]=dict(mean_iou=float(np.mean(scores)),flow_mean_ms=float(np.mean(times)))
        print(kind,report[kind],flush=True)
    Path('test_results/person_flow_methods.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
