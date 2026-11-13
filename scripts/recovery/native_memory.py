"""Native-memory recovery, based on the continuous propagation behavior in
D:\\Gimbal\\scripts\\demo.py (friend author's name not supplied).
Only that entry behavior is attributed to the friend. The identity checks below
are local additions; no uninspected SAM2 internals or weights were copied.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from sam_memory import SAMMemoryTracker, overlap


def shape_signature(mask):
    y,x = np.nonzero(mask)
    if len(x)<16:
        return None
    # PCA aspect is invariant to in-plane rotation (e.g. a rotating pen).
    points = np.column_stack((x,y)).astype(np.float32)
    if len(points)>4096:
        points=points[::max(1,len(points)//4096)]
    eigen=np.linalg.eigvalsh(np.cov(points.T))
    return float(np.sqrt(max(eigen[0],.1)/max(eigen[1],.1)))


def mask_agreement(a,b):
    if a is None or b is None or a.shape != b.shape:
        return 0.
    return float(np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum()))


class NativeMemoryRecovery(SAMMemoryTracker):
    """Freeze identity during loss; resume only unprompted memory-confirmed masks."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_anchor = None
        self.native_count = 0
        self.native_rollback = None
        self.native_velocity = np.zeros(2)

    def assess(self, frame, mask, score, logit, stability=1.):
        value=super().assess(frame,mask,score,logit,stability)
        if value is not None:
            value['anchor_similarity'] = (float(torch.stack([(a*b).sum() for a,b in
                zip(value['proto'],self.gallery[0])]).mean()) if self.gallery else 1.)
            shape=shape_signature(mask)
            anchor=getattr(self,'shape_anchor',None)
            value['shape_ratio'] = min(shape,anchor)/max(shape,anchor) if shape and anchor else 1.
            # Color remains diagnostic only, not an acceptance or ranking signal.
            value['score']=.5*value['anchor_similarity']+.5*value['iou']
        return value

    def identity_reason(self,value):
        if value is None:
            return 'empty_mask'
        if value['logit'] <= 0 or value['iou'] < self.cfg['search_iou']:
            return 'sam_quality_low'
        if value['anchor_similarity'] < self.cfg['search_similarity']:
            return 'selected_anchor_mismatch'
        if value['shape_ratio'] < self.cfg.get('native_shape_ratio',.4):
            return 'selected_shape_mismatch'
        if not self.cfg.get('native_area_min',.15) <= value['ratio'] <= self.cfg.get('native_area_max',6.):
            return 'selected_size_mismatch'
        return None

    def search_valid(self,value):
        return self.identity_reason(value) is None

    def motion_reason(self,value,frame,stamp):
        if value is None or self.last_box is None or self.last_good is None:
            return None
        elapsed=max(0.,stamp-self.last_good)
        old=np.asarray(self.last_box,float)
        current=np.asarray(value['box'],float)
        expected=(old[:2]+old[2:])/2+self.native_velocity*min(elapsed,1.)
        distance=float(np.linalg.norm((current[:2]+current[2:])/2-expected))
        diagonal=float(np.hypot(*frame.shape[:2]))
        radius=max(float(np.linalg.norm(old[2:]-old[:2]))*2,diagonal*.08)+diagonal*.15*min(elapsed,5.)
        value.update(motion_distance=distance,motion_limit=radius,lost_seconds=elapsed)
        return 'unexplained_motion_jump' if distance>radius else None

    def anchor_confirmation(self,frame,value):
        probe=self.sam.probe_anchor(frame.shape)
        anchor=self.assess(frame,probe['mask'],probe['iou'],probe['object_logit'])
        self.anchor_observation=anchor
        reason=self.identity_reason(anchor)
        agreement=mask_agreement(value['mask'],probe['mask']) if value else 0.
        if value is not None:
            value['anchor_mask_iou']=agreement
        if reason:
            return 'anchor_memory_'+reason
        if agreement < self.cfg.get('native_mask_agreement',.55):
            return 'temporal_anchor_disagreement'
        return None

    def ambiguity(self,frame,value):
        # Stateless SAM point decoding is used only to REJECT competing hypotheses.
        # Its mask can never initialize or replace the tracked object's memory.
        feature=F.normalize(self.feature_maps()[-1][0],dim=0)
        heat=torch.einsum('c,chw->hw',self.gallery[0][-1],feature).cpu().numpy()
        points=[]
        for _ in range(self.cfg.get('native_ambiguity_points',4)):
            y,x=np.unravel_index(heat.argmax(),heat.shape)
            points.append([(x+.5)*frame.shape[1]/heat.shape[1],(y+.5)*frame.shape[0]/heat.shape[0]])
            heat[max(0,y-3):y+4,max(0,x-3):x+4]=-10
        masks,scores,logits,stable=self.sam.decode_points(points,frame.shape[:2])
        second=0.
        self.proposals=[]
        for mask,score,logit,stability in zip(masks,scores,logits,stable):
            other=self.assess(frame,mask,score,logit,stability)
            if other and self.search_valid(other) and not any(overlap(other['box'],p['box'])>.7 for p in self.proposals):
                self.proposals.append(other)
            if other and self.search_valid(other) and overlap(other['box'],value['box']) < .3:
                second=max(second,other['score'])
        value['second_score']=second
        return 'ambiguous_sam_objects' if second and value['score']-second < self.cfg['margin'] else None

    def reject(self,stamp):
        self.state,self.lost_frames,self.hits,self.probation='SEARCHING',0,0,0
        self.candidate=None
        if self.native_rollback is not None:
            self.sam.outputs,self.sam.pinned_quality,self.sam.active_prompt_index=self.native_rollback
            self.native_rollback=None

    @torch.inference_mode()
    def step(self,frame,stamp,box=None):
        tick,before=time.perf_counter(),self.state
        if box is not None:
            self.shape_anchor=None
            self.native_rollback=None
            self.native_count=0
            self.native_velocity[:]=0
            result=super().step(frame,stamp,box)
            if result.get('valid'):
                self.shape_anchor=shape_signature(result['mask'])
                self.sam.motion_reference_stamp=stamp
            result['recovery_policy']='friend'
            return result
        if not self.gallery:
            return super().step(frame,stamp,box)
        self.native_count+=1
        motion=self.sam.snapshot_motion()
        result=self.sam.step(frame,stamp,continue_lost=True)
        value=self.assess(frame,result.get('mask'),result.get('iou',0),result.get('object_logit',-1))
        reason=self.identity_reason(value)
        if reason is None:
            reason=self.motion_reason(value,frame,stamp)
        checking=before!='TRACKING'
        # Every observation is already memory-conditioned SAM. Extra independent
        # anchor decoding is needed at trial entry/final commit, not on every
        # intermediate frame while memory is frozen. Restart if the mask jumps.
        verify=(self.native_count % self.cfg.get('native_memory_interval',3)==0 if not checking else
                self.candidate is None or (value is not None and overlap(value['box'],self.candidate)<.2)
                or (before=='TRACKING_PROBATION' and self.probation==1))
        if reason is None and verify:
            reason=self.anchor_confirmation(frame,value)
            if reason is None and checking:
                reason=self.ambiguity(frame,value)
        if reason is not None:
            self.sam.restore_motion(motion)
            self.reject(stamp)
            return self.decorate(result,before,reason,tick,value)
        if checking:
            consistent=overlap(value['box'],self.candidate) >= .2
            self.hits=self.hits+1 if consistent else 1
            self.candidate=value['box']
            if before!='TRACKING_PROBATION':
                if self.hits < self.cfg['normal_streak']:
                    self.sam.restore_motion(motion)
                    self.state='SEARCHING'
                    return self.decorate(result,before,'native_memory_streak',tick,value)
                self.state='TRACKING_PROBATION'
                self.probation=self.cfg['probation_frames']
                self.attempt_id+=1
            else:
                if not consistent:
                    self.sam.restore_motion(motion)
                    self.reject(stamp)
                    return self.decorate(result,before,'probation_motion_discontinuity',tick,value)
                self.probation-=1
                if self.probation==0:
                    self.state='TRACKING'
                    self.reacquired+=1
                    self.native_rollback=None
            # No tentative candidate ever enters memory, even during probation.
            if self.state!='TRACKING':
                self.sam.restore_motion(motion)
                return self.decorate(result,before,'native_memory_probation',tick,value)
        if self.last_box is not None and self.last_good is not None and stamp>self.last_good:
            old=np.asarray(self.last_box,float);new=np.asarray(value['box'],float)
            self.native_velocity=.5*self.native_velocity+.5*((new[:2]+new[2:]-old[:2]-old[2:])/2)/(stamp-self.last_good)
        self.last_good,self.last_box=stamp,value['box']
        self.update_motion(value['box'],stamp)
        self.sam.motion_reference_stamp=stamp
        if verify and value['iou']>=self.cfg['memory_iou'] and value['logit']>=self.cfg['memory_logit']:
            self.sam.commit_memory(self.cfg['pinned_frames'])
            self.recent_area=.9*self.recent_area+.1*int(value['mask'].sum())
        return self.decorate(result,before,'sam_selected_memory_verified',tick,value)

    @torch.inference_mode()
    def recenter_crop(self,frame,stamp,box):
        # A transport recenter is not permission to replace identity with its box.
        tick=time.perf_counter()
        old=({k:dict(v) for k,v in self.sam.outputs.items()},dict(self.sam.pinned_quality),
             self.sam.active_prompt_index,self.sam.last_box,self.sam.shape,self.recent_area,
             self.sam.snapshot_motion())
        result=super().recenter_crop(frame,stamp,box)
        value=self.assess(frame,result.get('mask'),result.get('iou',0),result.get('object_logit',-1))
        reason=self.identity_reason(value)
        if reason is None:
            reason=self.anchor_confirmation(frame,value)
        if reason:
            (self.sam.outputs,self.sam.pinned_quality,self.sam.active_prompt_index,
             self.sam.last_box,self.sam.shape,self.recent_area,motion)=old
            self.sam.restore_motion(motion)
            self.state='SEARCHING'
            return self.decorate(result,'TRACKING','recenter_'+reason,tick,value)
        self.native_velocity[:]=0
        self.sam.motion_reference_stamp=stamp
        return result

    @torch.inference_mode()
    def seed_candidate(self,frame,stamp,best):
        """SMALL/tile bridge: require original SAM memory, never prompt the proposal."""
        tick,before=time.perf_counter(),self.state
        probe=self.sam.probe_anchor(frame.shape)
        value=self.assess(frame,probe['mask'],probe['iou'],probe['object_logit'])
        if (not self.search_valid(value) or mask_agreement(probe['mask'],best['mask']) < self.cfg.get('native_mask_agreement',.55)
                or self.ambiguity(frame,value)):
            return None
        self.native_rollback=(self.sam.outputs,self.sam.pinned_quality,self.sam.active_prompt_index)
        anchor=self.sam.anchor_index
        self.sam.outputs={'cond_frame_outputs':{anchor:self.sam.outputs['cond_frame_outputs'][anchor]},'non_cond_frame_outputs':{}}
        self.sam.pinned_quality={}
        self.sam.active_prompt_index=anchor
        self.state,self.probation='TRACKING_PROBATION',self.cfg['probation_frames']
        self.attempt_id+=1
        self.candidate=value['box']
        self.hits=self.cfg['normal_streak']
        x,y,u,v=value['box']
        probe.update(dx=(x+u-frame.shape[1])/2,dy=(y+v-frame.shape[0])/2,
                     predictor_called=True,inference_ms=(time.perf_counter()-tick)*1000)
        return self.decorate(probe,before,'tile_original_memory_probation',tick,value)

    def decorate(self,result,before,reason,tick,value=None):
        result=super().decorate(result,before,reason,tick,value)
        # Include verification work instead of displaying only the first SAM call.
        if result.get('predictor_called',True):
            result['inference_ms']=result['pipeline_ms']
        result.update(recovery_policy='friend', anchor_similarity=None if value is None else value.get('anchor_similarity'),
                      shape_ratio=None if value is None else value.get('shape_ratio'),
                      anchor_mask_iou=None if value is None else value.get('anchor_mask_iou'),
                      search_top2=None if value is None else value.get('second_score'),
                      memory_frozen=self.state!='TRACKING',color_used_for_acceptance=False)
        if value is not None:
            result.update({key:value.get(key) for key in ('motion_distance','motion_limit','lost_seconds')})
        return result

    def label(self,correct,stamp,attempt_id=None):
        accepted=super().label(correct,stamp,attempt_id)
        if accepted and not correct:
            self.reject(stamp)
            anchor=self.sam.anchor_index
            self.sam.outputs={'cond_frame_outputs':{anchor:self.sam.outputs['cond_frame_outputs'][anchor]},
                              'non_cond_frame_outputs':{}}
            self.sam.pinned_quality={}
            self.sam.active_prompt_index=anchor
            self.sam.reset_motion()
            self.native_velocity[:]=0
            self.last_good=self.last_box=None
        return accepted
