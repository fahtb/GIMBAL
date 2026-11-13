"""GPU lifecycle and global recovery smoke; synthetic scenes check logic only."""
import json,time
import cv2,numpy as np,torch
from source_camera_tracker import SourceCameraTracker,ROOT
from hybrid_tracker import HybridTracker
from check_small_target import scene

cv2.setNumThreads(4)
base=SourceCameraTracker();base.warmup();tracker=HybridTracker(base)
rows=[];initial_id=None
for i in range(65):
    frame,gt,_,box=scene('occlusion',0)
    if 8<=i<22:frame[:]=45;gt[:]=False
    if i>=22:frame=np.roll(frame,180,axis=1);gt=np.roll(gt,180,axis=1)
    start=time.perf_counter();r=tracker.step(frame,i/15,box if i==0 else None)
    if i==0:initial_id=r['target_id']
    assert r['target_id']==initial_id
    if not r['valid']:assert r['gimbal_command']['mode']=='HOLD'
    rows.append(dict(frame=i,state=r['status'],reason=r.get('transition_reason'),valid=r['valid'],ms=(time.perf_counter()-start)*1000,top1=r.get('proposal_top1'),top2=r.get('proposal_top2')))
for i in range(5):
    frame,_,_,box=scene('near',0);r=tracker.step(frame,10+i,box)
    assert r['target_id']!=initial_id;initial_id=r['target_id']
    assert base.predictor.frame_idx==0
tracker.close()
out=ROOT/'test_results/hybrid';out.mkdir(parents=True,exist_ok=True)
report=dict(rows=rows,recovered=[r['frame'] for r in rows if r['state']=='RECOVERED'],peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20)
(out/'gpu_smoke.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='rows'})
assert report['recovered'],'global recovery failed; inspect report'
