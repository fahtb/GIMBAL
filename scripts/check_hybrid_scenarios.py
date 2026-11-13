"""Synthetic stress/lifecycle checks only; never a substitute for annotated video."""
import json,time,argparse
import cv2,numpy as np,torch
from source_camera_tracker import SourceCameraTracker,ROOT
from small_target import SmallTargetTracker
from hybrid_tracker import HybridTracker
from check_small_target import scene

def sample(case,i):
    if case in ('occlusion','exit','same_color','scale'):return scene(case,i)
    f,gt,distractor,box=scene('near',0)
    if case in ('partial','partial_long') and 12<=i<(25 if case=='partial' else 40):
        f[220:280,230:260]=45;gt[220:280,230:260]=False
    if case=='camera':
        rng=np.random.default_rng(10)
        background=rng.integers(20,70,f.shape,dtype=np.uint8);background[gt]=f[gt];f=background
    shift=2*i if case=='slow' else (round(80*np.sin(i*.23)) if case=='fast' else round(40*np.sin(i*.12)))
    if case in ('partial','partial_long'):shift=0
    f=np.roll(f,shift,axis=1);gt=np.roll(gt,shift,axis=1)
    box=np.array(box)+[shift,0,shift,0]
    return f,gt,distractor,box.tolist()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',default='slow,fast,partial,occlusion,exit,same_color,camera,scale')
    parser.add_argument('--skip-soak',action='store_true')
    parser.add_argument('--output',default='test_results/hybrid/scenarios.json')
    args=parser.parse_args()
    cv2.setNumThreads(4);base=SourceCameraTracker();base.warmup()
    reports=[];out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    for case in args.cases.split(','):
        for engine in ('baseline','enhanced'):
            tracker=SmallTargetTracker(base) if engine=='baseline' else HybridTracker(base)
            rows=[];target_id=None;partial_seed=None
            for i in range(47):
                f,gt,distractor,box=sample(case,i);tick=time.perf_counter()
                old_seed=getattr(tracker,'seed_frame',None)
                if i==12:partial_seed=old_seed
                old_gallery=len(tracker.identity.views) if engine=='enhanced' and tracker.identity else 0
                result=tracker.step(f,i/15,box if i==0 else None)
                mask=result.get('mask');mask=np.zeros_like(gt) if mask is None else mask
                inter=int((mask&gt).sum());union=int((mask|gt).sum())
                if engine=='enhanced':
                    if i==0:target_id=result['target_id']
                    assert result['target_id']==target_id
                    if not result['valid']:assert result['gimbal_command']['mode']=='HOLD' and result['dx'] is None
                    if result.get('partial_observation'):
                        assert tracker.seed_frame is old_seed
                        assert len(tracker.identity.views)==old_gallery
                        assert not result['is_confirmed'] and result['gimbal_command']['mode']=='HOLD'
                        assert base.predictor.frame_idx not in base.predictor.condition_state['output_dict']['non_cond_frame_outputs']
                        assert i/15-tracker.last_stamp<=tracker.hybrid.partial_timeout
                    if case=='partial_long' and 12<=i<40:
                        assert tracker.seed_frame is partial_seed,'partial mask contaminated trusted seed after timeout'
                rows.append(dict(frame=i,visible=bool(gt.any()),iou=inter/max(1,union),
                                 wrong=bool(mask.any() and (not gt.any() or (mask&distractor).sum()>inter)),
                                 state=result['status'],reason=result.get('transition_reason'),wall_ms=(time.perf_counter()-tick)*1000))
            reports.append(dict(case=case,engine=engine,correct_logic_frames=sum(r['visible'] and r['iou']>=.5 for r in rows),
                                visible_frames=sum(r['visible'] for r in rows),wrong_logic_frames=sum(r['wrong'] for r in rows),rows=rows))
            print({k:v for k,v in reports[-1].items() if k!='rows'},flush=True)
            tracker.close();out.write_text(json.dumps(reports,indent=2))
    if args.skip_soak:return
    tracker=HybridTracker(base);memory=[]
    for i in range(300):
        f,_,_,box=scene('near',0);r=tracker.step(f,i/15,box if i==0 else None)
        assert r['valid']
        bank=base.predictor.condition_state['output_dict']
        assert len(bank['non_cond_frame_outputs'])<=base.predictor.num_maskmem
        if i%50==0 or i==299:memory.append(dict(frame=i,allocated_mb=torch.cuda.memory_allocated()/1048576,recent=len(bank['non_cond_frame_outputs'])))
    f=np.zeros((240,320,3),np.uint8);r=tracker.step(f,21.)
    assert not r['valid'] and r['target_id'] is None and r['gimbal_command']['mode']=='HOLD'
    tracker.close();(out.parent/'soak.json').write_text(json.dumps(dict(memory_samples=memory,resolution_change='HOLD and reselect'),indent=2))

if __name__=='__main__':main()
