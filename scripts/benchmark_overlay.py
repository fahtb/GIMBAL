"""ROI blend comparison including both PCIe transfers in the GPU measurement."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch


def main():
    cv2.setNumThreads(4)
    image=np.random.default_rng(5).integers(0,256,(150,240,3),dtype=np.uint8)
    mask=np.zeros(image.shape[:2],np.uint8)
    cv2.ellipse(mask,(120,75),(100,60),0,0,360,255,-1)
    tint=np.empty_like(image);tint[:]=(0,254,0)
    result={}
    for backend in ('cpu','gpu'):
        times=[]
        for i in range(120):
            start=time.perf_counter()
            if backend=='cpu':
                out=image.copy()
                blend=cv2.addWeighted(image,.65,tint,.35,0)
                cv2.copyTo(blend,mask,out)
            else:
                source=torch.from_numpy(image).to('cuda')
                m=torch.from_numpy(mask).to('cuda')>0
                blend=(source.float()*.65+source.new_tensor([0,254,0]).float()*.35).round().clamp(0,255).byte()
                out=torch.where(m[:,:,None],blend,source).cpu().numpy()
            if i>=20: times.append((time.perf_counter()-start)*1000)
        result[backend]=dict(mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)))
    print(json.dumps(result,indent=2))
    Path('test_results/overlay_cpu_gpu.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__': main()
