"""OBJECT-only temporal identity gate. SAM is the only learned component.

Prediction/repair never enters SAM memory. Color is diagnostic, not identity.
Coordinates are those of the current SAM crop; recenter resets flow history.
"""
import time
import cv2
import numpy as np
import torch
from sam_memory import mask_box, overlap
from .native_memory import NativeMemoryRecovery, mask_agreement, shape_signature
from .identity_scores import relation,same_support,granularity_score,fuse


def center(mask):
    y,x=np.nonzero(mask)
    return np.array([x.mean(),y.mean()]) if len(x) else np.zeros(2)


def select_component(mask, predicted):
    """Do not discard a still-visible target when SAM merges distant objects."""
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    ids=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=8]
    if len(ids)<=1 or predicted is None:
        return mask,len(ids),False
    best=max(ids,key=lambda i:int(np.logical_and(labels==i,predicted).sum()))
    supported=[i for i in ids if np.logical_and(labels==i,predicted).sum()/max(1,min(stats[i,cv2.CC_STAT_AREA],predicted.sum()))>=.5]
    selected=np.isin(labels,supported or [best])
    return selected,len(ids),len(supported or [best])<len(ids)


class ObjectContinuityRecovery(NativeMemoryRecovery):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.previous_gray=self.previous_mask=None
        self.previous_stamp=self.trusted_stamp=None
        self.trial_mask=None
        self.diag={}
        self.proposals=[]
        self.ambiguous_latched=False
        self.continuity_active=False
        self.last_anchor_check=None
        self.probe_worker=None
        self.probe_result=None
        self.identity_epoch=0
        self.high_hits=0
        self.original_box=None
        self.distractors=[]
        from .quarantine import MemoryQuarantine
        self.quarantine=MemoryQuarantine()
        self.high_since=self.trial_since=None

    def ambiguity(self,frame,value):
        if self.probe_worker is not None:
            return self.async_evidence(frame,value,'ambiguity_reason')
        super().ambiguity(frame,value)
        raw=list(self.proposals)
        anchor=getattr(self,'anchor_observation',None)
        reference=value
        if anchor and self.search_valid(anchor) and same_support(anchor['mask'],value.get('mask'),self.cfg):
            reference=anchor
            raw.append(anchor)
        groups=[]
        for item in sorted(raw,key=lambda v:granularity_score(v,reference,self.cfg) if reference.get('mask') is not None else v['score'],reverse=True):
            if not any(same_support(item['mask'],old['mask'],self.cfg) for old in groups):
                groups.append(item)
        others=[v for v in groups if not same_support(v['mask'],reference.get('mask'),self.cfg)]
        self.distractors=[v['mask'].copy() for v in others[:4]]
        second=max([v['score'] for v in others] or [0.])
        value['second_score']=second
        self.proposals=groups
        self.diag.update(proposal_count=len(raw),identity_candidates=len(others),
                         chosen_proposal=next((i for i,v in enumerate(raw) if same_support(v['mask'],reference.get('mask'),self.cfg)),-1),
                         proposal_signals=[dict(score=v['score'],area=int(v['mask'].sum()),
                           **relation(v['mask'],value.get('mask'))) for v in raw])
        return 'ambiguous_sam_objects' if second and value['score']-second<self.cfg['margin'] else None

    def anchor_confirmation(self,frame,value):
        if self.probe_worker is not None:
            return self.async_evidence(frame,value,'anchor_reason')
        return super().anchor_confirmation(frame,value)

    def async_evidence(self,frame,value,key):
        evidence=self.probe_result
        diagnostic_key='anchor_probe_reason' if key=='anchor_reason' else 'identity_probe_reason'
        if evidence is not None:
            self.diag.update(probe_age_ms=max(0.,self.current_stamp-evidence['stamp'])*1000,
                             probe_ms=evidence.get('elapsed_ms'))
        if evidence is None or evidence['epoch']!=self.identity_epoch or self.current_stamp-evidence['stamp']>self.cfg.get('probe_max_age_s',.4):
            self.diag[diagnostic_key]='probe_pending'
            self.diag['probe_wait_reason']='not_ready' if evidence is None else 'epoch_changed' if evidence['epoch']!=self.identity_epoch else 'expired'
            self.probe_worker.submit(self,frame,value,self.current_stamp)
            return 'probe_pending'
        # Evidence must concern this same continuing mask, not merely the epoch.
        if not same_support(evidence['mask'],value.get('mask'),self.cfg):
            self.diag[diagnostic_key]='probe_support_changed'
            self.diag['probe_wait_reason']='mask_support_changed'
            self.probe_worker.submit(self,frame,value,self.current_stamp)
            return 'probe_support_changed'
        value['second_score']=evidence['second_score']
        value['anchor_mask_iou']=evidence.get('anchor_mask_iou')
        self.diag.update(evidence['diag'],probe_age_ms=max(0.,self.current_stamp-evidence['stamp'])*1000,
                         probe_ms=evidence['elapsed_ms'],anchor_probe_reason=evidence['anchor_reason'],
                         identity_probe_reason=evidence['ambiguity_reason'])
        # Old hypotheses can only veto, never initialize today's target.
        self.probe_worker.submit(self,frame,value,self.current_stamp)
        return evidence[key]

    def predict(self,frame,stamp):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if self.previous_mask is None or self.previous_gray.shape!=gray.shape:
            return gray,None,False
        dt=max(0.,stamp-self.previous_stamp)
        self.diag['continuity_dt_s']=dt
        if self.motion_stamp is not None:
            kdt=max(.001,min(1.,stamp-self.motion_stamp))
            self.motion.transitionMatrix=np.array([[1,0,kdt,0],[0,1,0,kdt],[0,0,1,0],[0,0,0,1]],np.float32)
            self.kalman_expected=self.motion.predict()[:2,0].copy()
            self.motion_stamp=stamp
        shift=self.native_velocity*min(dt,self.cfg.get('continuity_coast_s',.35))
        flow_ok=False
        # During occlusion, LK features belong to the occluder. Never let those
        # reverse the target prediction; coast with the last trusted velocity.
        points=(None if self.state=='COAST' else cv2.goodFeaturesToTrack(self.previous_gray,60,.01,3,
                                     mask=self.previous_mask.astype(np.uint8)*255))
        if points is not None and len(points)>=4 and dt<=self.cfg.get('continuity_max_gap_s',.6):
            moved,ok,_=cv2.calcOpticalFlowPyrLK(self.previous_gray,gray,points,None,winSize=(21,21),maxLevel=3)
            if moved is not None:
                back,ok2,_=cv2.calcOpticalFlowPyrLK(gray,self.previous_gray,moved,None,winSize=(21,21),maxLevel=3)
                if back is not None:
                    good=(ok.ravel()>0)&(ok2.ravel()>0)&(np.linalg.norm(back-points,axis=2).ravel()<1.5)
                    if good.sum()>=4:
                        delta=(moved-points)[good,0]
                        shift=np.median(delta,axis=0)
                        flow_ok=bool(np.median(np.linalg.norm(delta-shift,axis=1))<4)
        predicted=cv2.warpAffine(self.previous_mask.astype(np.uint8),
                                np.array([[1,0,shift[0]],[0,1,shift[1]]],np.float32),
                                (gray.shape[1],gray.shape[0]),flags=cv2.INTER_NEAREST)>0
        return gray,predicted,flow_ok

    def continuous(self,mask,predicted):
        if mask is None or predicted is None or not mask.any() or not predicted.any():
            return False
        iou=mask_agreement(mask,predicted)
        box=mask_box(predicted)
        dt=max(.001,min(self.cfg.get('continuity_max_gap_s',.6),self.diag.get('continuity_dt_s',1/30)))
        scale=np.linalg.norm(np.subtract(box[2:],box[:2]))
        limit=max(8.,scale*self.cfg.get('continuity_center_scale',.6))+np.linalg.norm(self.native_velocity)*dt
        distance=float(np.linalg.norm(center(mask)-center(predicted)))
        area=float(mask.sum()/max(1,predicted.sum()))
        self.diag.update(flow_mask_iou=iou,continuity_distance=distance,continuity_limit=float(limit),
                         temporal_area_ratio=area,continuity_dt_s=dt)
        # Residual geometry is measured after time-scaled motion prediction.
        # Tighten at short dt; never loosen beyond existing safety envelopes.
        tau=max(self.cfg.get('continuity_iou',.25),1-self.cfg.get('continuity_iou_decay_s',22.5)*dt)
        area_limit=min(self.cfg.get('continuity_area_jump',2.),np.exp(self.cfg.get('continuity_area_rate_s',20.8)*dt))
        self.diag.update(continuity_tau=tau,continuity_area_limit=float(area_limit))
        return iou>=tau and distance<=limit and 1/area_limit<=area<=area_limit

    def merged_growth(self,mask,predicted):
        if predicted is None or not predicted.any():
            return False
        # Compare both against transported shape AND the last clean memory area;
        # otherwise a fused mask can grow gradually and become its own reference.
        reference=min(float(predicted.sum()),float(self.recent_area or predicted.sum()))
        return mask.sum()>reference*self.cfg.get('continuity_merge_growth',1.35)

    def repair(self,frame,predicted,distractor):
        """Positive foreground + negative distractor; current cached encoder only."""
        if predicted is None or predicted.sum()<16:
            return None
        dist=cv2.distanceTransform(predicted.astype(np.uint8),cv2.DIST_L2,3)
        y,x=np.unravel_index(dist.argmax(),dist.shape)
        points,labels=[[float(x),float(y)]],[1]
        negative=np.logical_and(distractor,~cv2.dilate(predicted.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool))
        for other in self.distractors:
            if other.shape==negative.shape:
                negative|=other & distractor & ~cv2.dilate(predicted.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        if negative.any():
            nd=cv2.distanceTransform(negative.astype(np.uint8),cv2.DIST_L2,3)
            y,x=np.unravel_index(nd.argmax(),nd.shape)
            points.append([float(x),float(y)]);labels.append(0)
        masks,scores,logits,stable=self.sam.decode_points(points,frame.shape[:2],labels=labels)
        value=self.assess(frame,masks[0],scores[0],logits[0],stable[0])
        if value and self.identity_reason(value) is None and self.continuous(value['mask'],predicted) and not self.merged_growth(value['mask'],predicted):
            return value
        return None

    def accepted(self,frame,stamp,value,gray,commit=False):
        self.continuity_active=True
        if self.last_good is not None and stamp>self.last_good and self.last_box is not None:
            old=np.array(self.last_box,float);new=np.array(value['box'],float)
            velocity=((new[:2]+new[2:]-old[:2]-old[2:])/2)/(stamp-self.last_good)
            self.native_velocity=.5*self.native_velocity+.5*velocity
        self.last_good,self.last_box=stamp,value['box']
        self.trusted_stamp=stamp
        self.previous_gray,self.previous_mask,self.previous_stamp=gray,value['mask'].copy(),stamp
        self.sam.motion_reference_stamp=stamp
        self.update_motion(value['box'],stamp)
        self.motion_stamp=stamp
        committed=self.quarantine.observe(self.sam,stamp,commit,self.cfg)
        if committed:
            self.recent_area=.9*self.recent_area+.1*value['mask'].sum()
            if value['iou']>=self.cfg['memory_iou'] and value['logit']>=self.cfg['memory_logit']:
                self.remember(value['proto'],value['lab'],frame,value['mask'])
        return committed

    @torch.inference_mode()
    def step(self,frame,stamp,box=None):
        tick,before=time.perf_counter(),self.state
        self.diag={}
        self.current_stamp=stamp
        if self.probe_worker is not None:
            self.probe_result=self.probe_worker.poll()
            self.distractors=[]
            proof=self.probe_result
            if proof is not None and proof['epoch']==self.identity_epoch and stamp-proof['stamp']<=self.cfg.get('probe_max_age_s',.4):
                # For prompting, only use distractors still overlapping the fresh
                # SAM observation; stale positions must not suppress the target.
                self.distractors=proof.get('distractors',[])
        if box is not None:
            self.quarantine.clear();self.high_since=self.trial_since=None
            self.identity_epoch+=1;self.probe_result=None;self.high_hits=0
            self.distractors=[]
            self.original_box=list(box)
            self.previous_mask=self.previous_gray=None
            self.trusted_stamp=None
            self.continuity_active=False
            self.last_anchor_check=None
            self.proposals=[];self.ambiguous_latched=False;self.trial_mask=None
            result=super().step(frame,stamp,box)
            if result.get('valid'):
                self.previous_mask=result['mask'].copy()
                self.previous_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                self.previous_stamp=self.trusted_stamp=stamp
                self.continuity_active=True
            result['target_mode']='SAM_ONLY'
            return result
        gray,predicted,flow_ok=self.predict(frame,stamp)
        self.diag['continuity_flow_valid']=flow_ok
        motion=self.sam.snapshot_motion()
        result=self.sam.step(frame,stamp,continue_lost=True)
        self.native_count+=1
        raw=result.get('mask')
        raw=np.zeros(frame.shape[:2],bool) if raw is None else raw
        self.diag['raw_mask_area']=int(raw.sum())
        local=self.continuity_active and before in ('TRACKING','TRACKING_LOW','AMBIGUOUS','COAST') and self.trusted_stamp is not None and stamp-self.trusted_stamp<=self.cfg.get('continuity_max_gap_s',.6)
        chosen,components,split=select_component(raw,predicted if local else None)
        self.diag.update(mask_components=components,component_selected=split)
        value=self.assess(frame,chosen,result.get('iou',0),result.get('object_logit',-1))
        if value is not None:
            self.motion_reason(value,frame,stamp)
            if getattr(self,'kalman_expected',None) is not None:
                self.diag.update(kalman_innovation=float(np.linalg.norm(center(chosen)-self.kalman_expected)),
                                 kalman_limit=max(8.,float(np.linalg.norm(np.subtract(value['box'][2:],value['box'][:2])))*.6))
        if self.probe_worker is not None and (not local or self.probe_result is None):
            self.probe_worker.submit(self,frame,value,stamp)
        linked=local and self.continuous(chosen,predicted)
        repaired=False
        reason=self.identity_reason(value)
        weak_anchor=bool(linked and flow_ok and reason=='selected_anchor_mismatch' and
                         value['anchor_similarity']>=self.cfg['track_similarity'])
        if weak_anchor:
            reason=None  # Existing TRACK threshold only, never recovery threshold.
        fused=bool(local and self.merged_growth(chosen,predicted))
        self.diag['suspected_merge']=fused
        if fused:
            linked=False
            reason='connected_mask_growth'
        if local and (not linked or reason):
            candidate=self.repair(frame,predicted,raw)
            if candidate is not None:
                value=candidate;linked=True;reason=None;repaired=True
        confidence,parts=fuse(value,self.diag,self.cfg)
        self.diag.update(confidence=confidence,confidence_parts=parts)
        # Fused confidence includes distractor margin. A low margin must lower
        # control gain, not invalidate independently established continuity.
        interval=self.cfg.get('native_memory_interval',3)
        healthy=(before=='TRACKING' and linked and not (split or repaired or weak_anchor or fused or reason)
                 and flow_ok and value['iou']>=.9 and self.diag.get('flow_mask_iou',0)>=.8 and self.native_count>=12)
        if healthy:
            interval=self.cfg.get('stable_anchor_interval',interval)
        period=self.cfg.get('stable_probe_s',.3) if healthy else self.cfg.get('active_probe_s',.1)
        verify=(self.last_anchor_check is None or stamp-self.last_anchor_check>=period)
        if linked and reason is None:
            # Anchor-only attention may prefer the distractor. That is ambiguity,
            # not permission to discard the continuously observed target.
            ambiguous=split or repaired or weak_anchor or self.ambiguous_latched
            if verify and not repaired:
                self.last_anchor_check=stamp
                ambiguous=bool(split or weak_anchor or self.anchor_confirmation(frame,value) or self.ambiguity(frame,value))
            self.ambiguous_latched=ambiguous
            confidence,parts=fuse(value,self.diag,self.cfg)
            self.diag.update(confidence=confidence,confidence_parts=parts,identity_ambiguous=ambiguous)
            high=confidence>=self.cfg.get('confidence_high',.85)
            self.high_hits=self.high_hits+1 if high and not ambiguous else 0
            self.high_since=(stamp if self.high_since is None else self.high_since) if high and not ambiguous else None
            stable_high=(before=='TRACKING' and confidence>=self.cfg.get('confidence_hold',.8)) or (self.high_since is not None and stamp-self.high_since>=self.cfg.get('confidence_high_s',.067))
            self.state='TRACKING' if stable_high and not ambiguous else 'TRACKING_LOW'
            overlap_risk=any(other.shape==value['mask'].shape and np.logical_and(other,value['mask']).any() for other in self.distractors)
            self.diag['memory_distractor_overlap']=overlap_risk
            commit=self.state=='TRACKING' and high and not ambiguous and not overlap_risk and not split and not repaired and value['iou']>=self.cfg['memory_iou'] and value['logit']>=self.cfg['memory_logit']
            if not commit:
                self.sam.restore_motion(motion)
            commit=self.accepted(frame,stamp,value,gray,commit)
            self.hits=self.probation=0
            return self.finish(result,before,'continuity_reprompt' if repaired else 'continuity_component' if split else 'continuity_kept',tick,value,commit)
        self.sam.restore_motion(motion)
        self.quarantine.clear();self.high_since=None
        # Bounded visualization-only coast; never controls a gimbal or writes memory.
        if local and predicted is not None and predicted.any() and stamp-self.trusted_stamp<=self.cfg.get('continuity_coast_s',.35):
            self.previous_gray,self.previous_mask,self.previous_stamp=gray,predicted,stamp
            self.state='COAST'
            value=self.assess(frame,predicted,0.,-1.)
            return self.finish(result,before,reason or 'continuity_violation',tick,value,False)
        self.state='SEARCHING'
        self.high_hits=0
        self.continuity_active=False
        # Enumerate SAM encoder/decoder hypotheses even after an empty temporal
        # mask. Rank without color; none is allowed to initialize identity memory.
        reference=value or {'box':self.last_box or [0,0,0,0],'score':0.}
        probe_ambiguity=self.ambiguity(frame,reference)
        candidates=([value] if value is not None and self.search_valid(value) else [])+(list(self.proposals) if self.probe_worker is None else [])
        unique=[]
        for item in sorted(candidates,key=lambda v:v['score'],reverse=True):
            if not any(same_support(item['mask'],old['mask'],self.cfg) for old in unique):
                unique.append(item)
        value=unique[0] if unique else None
        anchor_reason=self.identity_reason(value)
        if value is not None and anchor_reason is None:
            anchor_reason=self.motion_reason(value,frame,stamp) or self.anchor_confirmation(frame,value)
        second=unique[1]['score'] if len(unique)>1 else 0.
        if value:
            value['second_score']=second
        ambiguous=probe_ambiguity or ('ambiguous_sam_objects' if second and value['score']-second<self.cfg['margin'] else None)
        # Preserve the independent ambiguity probe's rejection (including tests
        # or custom SAM adapters that do not expose proposal arrays).
        if reference.get('second_score',0) and reference['score']-reference['second_score']<self.cfg['margin']:
            ambiguous='ambiguous_sam_objects'
        if ambiguous:
            anchor_reason=ambiguous
        if ambiguous and ambiguous not in ('probe_pending','probe_support_changed'):
            self.state='AMBIGUOUS';self.ambiguous_latched=True
            anchor_reason=ambiguous
        if anchor_reason is None and value is not None:
            consistent=self.trial_mask is not None and mask_agreement(value['mask'],self.trial_mask)>=self.cfg.get('continuity_iou',.25)
            self.hits=self.hits+1 if consistent else 1
            if not consistent or self.trial_since is None:
                self.trial_since=stamp
            self.trial_mask=value['mask'].copy()
            held=stamp-self.trial_since
            if held>=self.cfg.get('recovery_confirm_s',.1):
                self.state='TRACKING_PROBATION'
                self.probation=max(0.,self.cfg.get('recovery_probation_s',.167)+self.cfg.get('recovery_confirm_s',.1)-held)
                if before!='TRACKING_PROBATION':
                    self.attempt_id+=1
                if not self.probation:
                    clean=self.sam.rebase_verified(frame,stamp,value['mask'])
                    clean_value=self.assess(frame,clean.get('mask'),clean.get('iou',0),clean.get('object_logit',-1))
                    self.identity_epoch+=1;self.probe_result=None;self.distractors=[]
                    if clean_value is None or self.identity_reason(clean_value) or mask_agreement(clean_value['mask'],value['mask'])<self.cfg.get('native_mask_agreement',.55):
                        self.state='SEARCHING';self.hits=0;self.trial_since=None
                        return self.finish(result,before,'clean_rebase_rejected',tick,None,False)
                    value=clean_value;result.update(mask=value['mask'],box=value['box'])
                    self.state='TRACKING_LOW';self.reacquired+=1;self.ambiguous_latched=False
                    self.previous_mask=self.previous_gray=None
                    self.motion_stamp=None;self.native_velocity[:]=0
                    self.accepted(frame,stamp,value,gray,False)
            reason='recovery_probation' if self.state=='TRACKING_PROBATION' else 'recovery_confirmed' if self.state in ('TRACKING','TRACKING_LOW') else 'recovery_streak'
        else:
            self.hits=self.probation=0;self.trial_mask=None;self.trial_since=None
            reason=anchor_reason or 'empty_mask'
        return self.finish(result,before,reason,tick,value,False)

    def finish(self,result,before,reason,tick,value,commit):
        self.diag.update(quarantine_size=len(self.quarantine.entries),sam_frame_idx=self.sam.index,
                         conditioning_count=len(self.sam.outputs['cond_frame_outputs']),
                         recent_memory_count=len(self.sam.outputs['non_cond_frame_outputs']))
        active=self.state in ('TRACKING','TRACKING_LOW','COAST') or (self.state=='AMBIGUOUS' and reason.startswith('continuity'))
        result=super().decorate(result,before,reason,tick,value)
        if active and value:
            x,y,u,v=value['box'];h,w=value['mask'].shape
            result.update(mask=value['mask'],box=value['box'],dx=(x+u-w)/2,dy=(y+v-h)/2,valid=self.state!='COAST')
        elif self.state=='TRACKING_PROBATION':
            result.update(mask=None,box=None,dx=None,dy=None,valid=False)
        result.update(self.diag)
        result.update(mask_area=int(value['mask'].sum()) if value else 0,ambiguous=self.state=='AMBIGUOUS',
                      memory_frozen=not commit,control_gain=0. if not result.get('valid') else 1.,
                      estimated=self.state=='COAST',recovery_policy='friend-object-continuity')
        if self.state=='TRACKING_LOW':
            result['control_gain']=self.cfg.get('low_control_gain',.35)
            result['ambiguous']=bool(self.ambiguous_latched)
        if self.state=='COAST':
            result.update(dx=None,dy=None)
        if self.state in ('SEARCHING','AMBIGUOUS','TRACKING_PROBATION'):
            candidates=([value] if value else [])+list(self.proposals)
            unique=[]
            for item in candidates:
                if not any(same_support(item['mask'],old['mask'],self.cfg) for old in unique):
                    unique.append(item)
            result['search_masks']=[v['mask'] for v in unique[:9]]
            result['candidate_scores']=[v['score'] for v in unique[:9]]
        proof=self.probe_result
        if (self.probe_worker is not None and not result.get('valid') and not getattr(self,'small_mode',False)
                and proof is not None and proof['epoch']==self.identity_epoch
                and self.current_stamp-proof['stamp']<=self.cfg.get('probe_max_age_s',.4)):
            result.update(candidate_frame=proof['candidate_frame'],candidate_stamp=proof['stamp'],
                          search_masks=proof['candidate_masks'])
        return result

    def close(self):
        if self.probe_worker is not None:
            self.probe_worker.close()

    def recenter_crop(self,frame,stamp,box):
        self.quarantine.clear()
        self.identity_epoch+=1;self.probe_result=None
        verifier=self.probe_worker
        self.probe_worker=None
        try:
            result=super().recenter_crop(frame,stamp,box)
        finally:
            self.probe_worker=verifier
        self.previous_mask=self.previous_gray=None
        self.continuity_active=False
        if result.get('valid'):
            self.previous_mask=result['mask'].copy()
            self.previous_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            self.previous_stamp=self.trusted_stamp=stamp
            self.continuity_active=True
        return result

    def seed_candidate(self,frame,stamp,best):
        self.quarantine.clear()
        # Rare crop-coordinate transitions retain their original synchronous
        # same-tile verification; background evidence cannot cross crop origins.
        verifier=self.probe_worker;self.probe_worker=None
        try:
            result=super().seed_candidate(frame,stamp,best)
        finally:
            self.probe_worker=verifier
        if result is not None:
            self.identity_epoch+=1;self.probe_result=None
            self.previous_mask=self.previous_gray=None
            self.trusted_stamp=None
            self.continuity_active=False
            self.trial_mask=best['mask'].copy()
            self.hits=0
        return result

    def label(self,correct,stamp,attempt_id=None):
        accepted=super().label(correct,stamp,attempt_id)
        if accepted and not correct:
            self.quarantine.clear();self.trial_since=self.high_since=None
            self.identity_epoch+=1;self.probe_result=None;self.distractors=[]
            self.previous_mask=self.previous_gray=None
            self.trusted_stamp=None
            self.continuity_active=False
            self.trial_mask=None
        return accepted
