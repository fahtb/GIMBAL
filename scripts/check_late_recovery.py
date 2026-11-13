"""Late-return GPU regression against the committed pre-fix tracker."""
import sys,types,subprocess,json
from pathlib import Path
import cv2
import numpy as np
from source_camera_tracker import SourceCameraTracker,ROOT
from small_target import SmallTargetTracker
from check_small_target import scene


def main():
    old=types.ModuleType('previous_small_tracker');sys.modules[old.__name__]=old
    code=subprocess.check_output(['git','-c','safe.directory=D:/Sam','show','2503b9c:scripts/small_target.py'],cwd=ROOT,text=True)
    exec(compile(code,'previous_small_tracker','exec'),old.__dict__)
    base=SourceCameraTracker();base.warmup();reports=[]
    for case in ('return','smaller_return','distractor_only'):
        for name,cls in [('before',old.SmallTargetTracker),('after',SmallTargetTracker)]:
            tracker=cls(base);rows=[]
            for i in range(40):
                frame,gt,distractor,box=scene('occlusion',0)
                # One frame per second tests long loss without wall-clock sleeps.
                if 6<=i<20 or (case=='distractor_only' and i>=6):
                    frame[:]=45;gt[:]=False
                if case=='smaller_return' and i>=20:
                    patch=frame[220:232,200:212].copy();frame[:]=45;gt[:]=False
                    frame[220:226,200:206]=cv2.resize(patch,(6,6),interpolation=cv2.INTER_AREA);gt[220:226,200:206]=True
                if case=='distractor_only' and i>=20:
                    cv2.rectangle(frame,(200,220),(211,231),(90,190,220),-1)
                    cv2.circle(frame,(206,226),4,(25,25,25),1)
                r=tracker.step(frame,float(i),box if i==0 else None)
                mask=r.get('mask');iou=0 if mask is None else float((mask&gt).sum()/max(1,(mask|gt).sum()))
                rows.append(dict(frame=i,state=r['status'],reason=r.get('transition_reason'),iou=iou,valid=r.get('valid'),similarity=r.get('appearance_score')))
                assert len(getattr(tracker,'gallery',[]))<=5
            report=dict(case=case,mode=name,recovered=next((x['frame'] for x in rows if x['frame']>=20 and x['iou']>=.5),None),
                        false_lock=sum(bool(x['valid']) for x in rows[6:] if case=='distractor_only'),rows=rows)
            reports.append(report);print({k:v for k,v in report.items() if k!='rows'},flush=True)
    out=ROOT/'test_results/small_target';out.mkdir(parents=True,exist_ok=True)
    (out/'late_recovery.json').write_text(json.dumps(reports,indent=2))
    base.close()
    for r in reports:
        if r['mode']=='after':
            assert r['false_lock']==0
            if r['case']!='distractor_only': assert r['recovered'] is not None,r

if __name__=='__main__':main()
