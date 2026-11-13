"""Small-target ROI epochs around source SAM2. One model; no cross-ROI memory reuse.

Only observed, accepted SAM masks seed a new coordinate system. Rejected frames
are removed from the recent memory bank; original appearance anchor is immutable.
No learned detector, color identity test or fabricated confidence is used.
"""
import json
import time
from dataclasses import dataclass, fields
from pathlib import Path
import cv2
import numpy as np

@dataclass
class Settings:
    trigger_model_px: float = 64.
    context: float = 3.
    min_side: int = 110
    edge_fraction: float = .18
    resize_ratio: float = 1.7
    min_epoch_frames: int = 5
    motion_factor: float = 4.0
    min_motion_px: float = 24.
    max_predict_seconds: float = .8
    appearance_min: float = .25
    recovery_appearance: float = .65
    texture_min: float = 6.
    recovery_frames: int = 2
    lost_after: int = 3
    search_stride: int = 3
    expand_factor: float = 3.0
    timeout: float = 8.
    auto_reacquire: bool = True
    slow_search_interval: float = .20
    recovery_scale_rate: float = .7
    recovery_scale_limit: float = 8.
    gallery_size: int = 6
    gallery_update_similarity: float = .8
    area_min_ratio: float = .35
    area_max_ratio: float = 2.8

    @classmethod
    def load(cls,path=None):
        data=json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else {}
        unknown=set(data)-{f.name for f in fields(cls)}
        if unknown: raise ValueError(f'Unknown small target settings: {unknown}')
        cfg=cls(**data)
        for key in ('min_side','min_epoch_frames','recovery_frames','lost_after','search_stride'):
            if getattr(cfg,key)<1: raise ValueError(f'{key} must be positive')
        if cfg.gallery_size<2 or cfg.slow_search_interval<=0 or cfg.recovery_scale_rate<0 or cfg.recovery_scale_limit<1:
            raise ValueError('Invalid automatic recovery settings')
        if not 0<=cfg.gallery_update_similarity<=1:
            raise ValueError('Invalid gallery update similarity')
        if cfg.context<2 or cfg.expand_factor<=1 or cfg.resize_ratio<=1 or cfg.timeout<=0:
            raise ValueError('Invalid ROI scale/timeout')
        if not 0<=cfg.appearance_min<=cfg.recovery_appearance<=1 or not 0<cfg.edge_fraction<.5:
            raise ValueError('Invalid appearance/edge settings')
        return cfg


def bounds(mask):
    y,x=np.nonzero(mask)
    return np.array([x.min(),y.min(),x.max()+1,y.max()+1],float) if len(x) else None


def appearance(frame,mask):
    b=bounds(mask)
    if b is None: return None,0.
    x,y,r,b=map(int,b)
    gray=cv2.cvtColor(frame[y:b,x:r],cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(24,24),interpolation=cv2.INTER_AREA).astype(np.float32)
    texture=float(gray.std())
    gray-=gray.mean()
    return gray.ravel()/max(float(np.linalg.norm(gray)),1e-6),texture


def structural_match(a,b):
    # Tiny masks gain/lose boundary pixels. Search small translation/scale
    # alignment rather than lowering the identity acceptance threshold.
    target=a.reshape(24,24).astype(np.float32)
    best=-1.
    for size in (20,24,28):
        reference=cv2.resize(b.reshape(24,24),(size,size))
        padded=cv2.copyMakeBorder(reference,6,6,6,6,cv2.BORDER_CONSTANT,value=0)
        best=max(best,float(cv2.matchTemplate(padded,target,cv2.TM_CCOEFF_NORMED).max()))
    return best


def region(shape,box,side):
    h,w=shape[:2];cx,cy=(box[:2]+box[2:])/2
    rw,rh=min(w,max(16,int(side))),min(h,max(16,int(side)))
    x=int(np.clip(round(cx-rw/2),0,w-rw));y=int(np.clip(round(cy-rh/2),0,h-rh))
    return (x,y,x+rw,y+rh)


def roi_edge_risk(roi,covered,edge_fraction):
    """Safety recenter has no epoch cooldown; covered includes motion prediction."""
    x,y,r,b=roi
    return min(covered[0]-x,r-covered[2],covered[1]-y,b-covered[3]) < min(r-x,b-y)*edge_fraction


class SmallTargetTracker:
    target_mode='SAM_ONLY'
    def __init__(self,base,config=None):
        self.base=base;self.cfg=Settings.load(config);self.reset()

    def reset(self):
        self.base.reset();self.state='WAIT_SELECTION';self.roi=None
        self.seed_frame=self.seed_mask=self.last_box=self.anchor=None
        self.velocity=np.zeros(2);self.last_stamp=0.;self.bad=self.hits=self.epoch=0
        self.search_step=0;self.reacquired=0;self.pending_box=None
        self.gallery=[];self.search_cycle=0;self.last_search_stamp=-float('inf')

    def label(self,*args): return self.base.label(*args)
    def close(self): self.reset()

    def choose_roi(self,shape,box,expand=1.):
        cfg=self.cfg;h,w=shape[:2];wh=box[2:]-box[:2]
        pixels=wh*np.array([self.base.predictor.image_size/w,self.base.predictor.image_size/h])
        if min(pixels)>=cfg.trigger_model_px and expand==1:
            return (0,0,w,h)
        return region(shape,box,max(cfg.min_side,max(wh)*cfg.context)*expand)

    def seed_epoch(self,roi):
        x,y,r,b=roi
        mask=self.seed_mask[y:b,x:r]
        bb=bounds(mask)
        if bb is None: raise ValueError('New ROI excludes verified seed')
        self.base.step(self.seed_frame[y:b,x:r],self.last_stamp,
                       [bb[0],bb[1],bb[2]-1,bb[3]-1],mask_prompt=mask)
        self.roi=roi;self.epoch=0

    def assess(self,frame,mask,stamp):
        box=bounds(mask)
        if box is None: return False,'empty_mask',None,None
        dt=max(.001,stamp-self.last_stamp)
        center=(box[:2]+box[2:])/2
        oldcenter=(self.last_box[:2]+self.last_box[2:])/2
        predicted=oldcenter+self.velocity*min(dt,self.cfg.max_predict_seconds)
        distance=float(np.linalg.norm(center-predicted))
        extent=float(np.linalg.norm(self.last_box[2:]-self.last_box[:2]))
        limit=max(self.cfg.min_motion_px,extent*self.cfg.motion_factor)
        if distance>limit: return False,'motion_gate',box,None
        ratio=float(mask.sum()/max(1,self.seed_mask.sum()))
        wh=self.last_box[2:]-self.last_box[:2]
        boundary_low=max(1.,float(np.prod(np.maximum(1,wh-2))))/max(1,self.seed_mask.sum())
        boundary_high=float(np.prod(wh+2))/max(1,self.seed_mask.sum())
        # Bounded scale uncertainty grows with actual elapsed time after loss.
        growth=min(self.cfg.recovery_scale_limit,1+max(0.,dt-.1)*self.cfg.recovery_scale_rate)**2 if self.bad else 1.
        if not min(self.cfg.area_min_ratio,boundary_low)/growth<=ratio<=max(self.cfg.area_max_ratio,boundary_high)*growth:
            return False,'area_gate',box,None
        descriptor,texture=appearance(frame,mask)
        similarity=max(structural_match(descriptor,ref) for ref in [self.anchor,*self.gallery]) if descriptor is not None else -1.
        recovering=self.bad>0
        threshold=self.cfg.recovery_appearance if recovering else self.cfg.appearance_min
        if recovering and (texture<self.cfg.texture_min or self.anchor_texture<self.cfg.texture_min):
            return False,'insufficient_texture_reselect',box,similarity
        if min(texture,self.anchor_texture)>=self.cfg.texture_min and similarity<threshold:
            return False,'appearance_gate',box,similarity
        return True,'continuity_and_structure',box,similarity

    def step(self,frame,stamp,box=None):
        start=time.perf_counter();before=self.state;similarity=None;epoch_changed=False
        if box is not None:
            self.reset()
            prompt=np.array(box,float)
            result=self.base.step(frame,stamp,box)
            initial=(0,0,frame.shape[1],frame.shape[0])
            # Prefer the established full-frame prompt. Crop prompting is only a
            # fallback for an empty small-object selection, explicitly supplied by user.
            if result.get('mask') is None and prompt[2]>prompt[0] and prompt[3]>prompt[1]:
                initial=self.choose_roi(frame.shape,prompt)
                x,y,r,b=initial
                result=self.base.step(frame[y:b,x:r],stamp,(prompt-np.array([x,y,x,y])).tolist())
                if result.get('mask') is not None:
                    full=np.zeros(frame.shape[:2],bool);full[y:b,x:r]=result['mask'];result['mask']=full
            if result.get('mask') is not None:
                full=result['mask']
                self.anchor,self.anchor_texture=appearance(frame,full)
                self.gallery=[]
                self.seed_frame=frame.copy();self.seed_mask=full.copy()
                self.last_box=bounds(full);self.last_stamp=stamp
                self.roi=initial;self.state='TRACKING'
                desired=self.choose_roi(frame.shape,self.last_box)
                if desired!=initial:self.seed_epoch(desired);epoch_changed=True
            else: self.state='LOST'
            reason='manual_selection' if self.state=='TRACKING' else 'initial_mask_empty_reselect'
        elif self.seed_frame is None or frame.shape!=self.seed_frame.shape:
            self.reset()
            return dict(status='LOST',mask=None,box=None,dx=None,dy=None,valid=False,
                        inference_ms=0.,predictor_called=False,target_mode=self.target_mode,
                        transition_reason='select_target')
        else:
            cfg=self.cfg;elapsed=stamp-self.last_stamp
            if self.bad and elapsed>cfg.timeout:
                self.state='LOST'
                if not cfg.auto_reacquire:
                    return self.result(dict(inference_ms=0.,predictor_called=False),frame,stamp,before,
                                       'timeout_reselect',None,start,False)
                if not self.hits and stamp-self.last_search_stamp<cfg.slow_search_interval:
                    return self.result(dict(inference_ms=0.,predictor_called=False),frame,stamp,before,
                                       'search_wait',None,start,False)
            self.epoch+=1;self.search_step+=1
            if self.bad:
                self.last_search_stamp=stamp
            if self.bad and not self.hits and (self.bad==1 or self.search_step%cfg.search_stride==0):
                # Broad context alone shrinks a distant object back below SAM's
                # resolution. Alternate broad and detail searches indefinitely.
                factors=(.6,.35,1.,cfg.expand_factor,cfg.expand_factor**2,100.)
                factor=factors[self.search_cycle%len(factors)];self.search_cycle+=1
                wh=self.last_box[2:]-self.last_box[:2]
                side=max(cfg.min_side,max(wh)+32,max(wh)*cfg.context*factor)
                desired=region(frame.shape,self.last_box,side)
                self.seed_epoch(desired);epoch_changed=True
            elif not self.bad:
                shift=self.velocity*min(max(0.,elapsed),cfg.max_predict_seconds)
                predicted=self.last_box+np.tile(shift,2)
                covered=np.r_[np.minimum(self.last_box[:2],predicted[:2]),np.maximum(self.last_box[2:],predicted[2:])]
                desired=self.choose_roi(frame.shape,covered)
                x,y,r,b=self.roi;bb=self.last_box
                near=roi_edge_risk(self.roi,covered,cfg.edge_fraction)
                ratio=(desired[2]-desired[0])/(r-x)
                if desired!=self.roi and (near or (self.epoch>=cfg.min_epoch_frames and (ratio<1/cfg.resize_ratio or ratio>cfg.resize_ratio))):
                    self.seed_epoch(desired);epoch_changed=True
            x,y,r,b=self.roi
            bank=self.base.predictor.condition_state['output_dict']['non_cond_frame_outputs']
            previous=dict(bank)
            result=self.base.step(frame[y:b,x:r],stamp)
            local=result.get('mask');mask=np.zeros(frame.shape[:2],bool)
            if local is not None: mask[y:b,x:r]=local
            good,reason,newbox,similarity=self.assess(frame,mask,stamp)
            if good and self.bad:
                # Require the returning proposal itself to stay spatially consistent.
                if self.pending_box is not None and np.linalg.norm(newbox-self.pending_box)>max(cfg.min_motion_px,np.linalg.norm(newbox[2:]-newbox[:2])):
                    self.hits=0
                self.pending_box=newbox;self.hits+=1
                good=self.hits>=cfg.recovery_frames
                if not good: reason='recovery_probation'
            if good:
                if self.bad: self.reacquired+=1
                dt=max(.001,stamp-self.last_stamp)
                self.velocity=.5*self.velocity+.5*((newbox[:2]+newbox[2:]-self.last_box[:2]-self.last_box[2:])/2/dt)
                descriptor,texture=appearance(frame,mask)
                # Immutable original anchor + bounded, verified recent views.
                # Never learn a proposal during uncertainty/probation.
                if similarity is not None and cfg.gallery_update_similarity<=similarity<.97 and texture>=cfg.texture_min:
                    self.gallery.append(descriptor.copy())
                    self.gallery=self.gallery[-max(1,cfg.gallery_size-1):]
                self.seed_frame=frame.copy();self.seed_mask=mask.copy();self.last_box=newbox;self.last_stamp=stamp
                self.bad=self.hits=0;self.search_cycle=0;self.pending_box=None;self.state='TRACKING';result['mask']=mask
            else:
                # track() writes its own memory; quarantine by restoring only recent bank.
                # Frame counter still advances causally; missing memory entries are allowed.
                bank=self.base.predictor.condition_state['output_dict']['non_cond_frame_outputs']
                bank.clear();bank.update(previous)
                self.bad+=1
                if reason!='recovery_probation': self.hits=0;self.pending_box=None
                self.state='LOST' if self.bad>=cfg.lost_after else 'UNCERTAIN'
                result['mask']=None
        return self.result(result,frame,stamp,before,reason,similarity,start,epoch_changed)

    def result(self,result,frame,stamp,before,reason,similarity,start,changed):
        mask=result.get('mask') if self.state=='TRACKING' else None
        box=bounds(mask) if mask is not None else None
        h,w=frame.shape[:2];roi=self.roi or (0,0,w,h)
        target=(box[2:]-box[:2]).tolist() if box is not None else None
        model_px=([target[0]*self.base.predictor.image_size/(roi[2]-roi[0]),
                   target[1]*self.base.predictor.image_size/(roi[3]-roi[1])] if target else None)
        result.update(status=self.state,transition=f'{before}->{self.state}',transition_reason=reason,
            mask=mask,box=[int(box[0]),int(box[1]),int(box[2]-1),int(box[3]-1)] if box is not None else None,
            dx=(box[0]+box[2]-1-w)/2 if box is not None else None,
            dy=(box[1]+box[3]-1-h)/2 if box is not None else None,
            valid=box is not None,control_gain=1. if box is not None else 0.,small=True,
            crop_box=list(roi),target_pixels=target,model_target_pixels=model_px,
            appearance_score=similarity,appearance_samples=1+len(self.gallery),roi_reinitialized=changed,source_width=w,source_height=h,
            reacquired=self.reacquired,pipeline_ms=(time.perf_counter()-start)*1000,
            captured=stamp,recovery_policy='small_target_roi',target_mode=self.target_mode)
        return result


def debug_hud(view,result):
    """Only observational values; source does not expose identity confidence."""
    roi=result.get('crop_box')
    if roi:
        x,y,r,b=map(int,roi);cv2.rectangle(view,(x,y),(r-1,b-1),(255,180,0),1)
    pixels=result.get('target_pixels');model=result.get('model_target_pixels')
    lines=[f'{result.get("status","WAIT_SELECTION")} | {result.get("transition_reason","")}',
           f'target px {pixels} | model px {None if model is None else [round(v,1) for v in model]}',
           f'appearance {result.get("appearance_score")} | identity confidence: N/A']
    if result.get('engine')=='enhanced':
        # Coordinates always refer to the source frame; scale only for this view.
        sx=view.shape[1]/result.get('source_width',view.shape[1]);sy=view.shape[0]/result.get('source_height',view.shape[0])
        for key,color in (('kalman_box',(255,180,0)),('search_box',(180,0,255)),('candidate_box',(0,180,255))):
            bb=result.get(key)
            if bb is not None:
                a,b,c,d=[round(v*(sx if i%2==0 else sy)) for i,v in enumerate(bb)]
                cv2.rectangle(view,(a,b),(c,d),color,1)
        def number(key):
            v=result.get(key)
            return '-' if v is None else f'{v:.2f}'
        lines += [f'ID {result.get("target_id")} | gallery {result.get("identity_gallery")} | {result.get("recommended_control_mode")}',
                  f'SAM stability {number("tracking_quality")} | top1/top2 {number("proposal_top1")}/{number("proposal_top2")}',
                  f'SAM {number("inference_ms")} ms | age {number("age_ms")} ms | VRAM {number("gpu_allocated_mb")} MiB',
                  f'K/I/O: {result.get("features")} | blue: prediction only']
    color=(0,0,255) if result.get('status') in ('LOST','REDETECTING') else (0,220,255)
    for i,line in enumerate(lines):cv2.putText(view,line,(10,48+i*23),0,.45,color,1)
    return view
