"""One bounded verifier on an isolated SAM instance and CUDA stream.

Reuses cloned encoder features; never encodes a frame or touches live Kalman state.
The queue contains at most one job. Stale results cannot commit live memory.
"""
import threading
import time
import torch


def clone_tree(value):
    if isinstance(value,torch.Tensor):
        return value.detach().clone()
    if isinstance(value,dict):
        return {k:clone_tree(v) for k,v in value.items()}
    if isinstance(value,list):
        return [clone_tree(v) for v in value]
    if isinstance(value,tuple):
        return tuple(clone_tree(v) for v in value)
    return value


class ProbeWorker:
    def __init__(self,sam,config):
        from .continuity import ObjectContinuityRecovery
        self.tracker=ObjectContinuityRecovery(sam,config)
        self.lock=threading.Lock();self.wake=threading.Event();self.stop=threading.Event()
        self.job=self.result=None;self.busy=False;self.error=None;self.last_submit=0.
        self.thread=threading.Thread(target=self.run,daemon=True,name='sam-verification')
        self.thread.start()

    def submit(self,owner,frame,value,stamp):
        import numpy as np
        if value is None or value.get('mask') is None:
            value=dict(mask=np.zeros(frame.shape[:2],bool),iou=0.,logit=-1.)
        now=time.perf_counter()
        with self.lock:
            if self.busy or now-self.last_submit<owner.cfg.get('probe_period_s',.2):
                return False
            self.busy=True;self.last_submit=now
        try:
            anchor=owner.sam.anchor_index
            snapshot=dict(epoch=owner.identity_epoch,stamp=stamp,frame=frame.copy(),
                mask=value['mask'].copy(),iou=value['iou'],logit=value['logit'],
                features=clone_tree(owner.sam.cached_features),anchor=anchor,
                bank=clone_tree(owner.sam.outputs['cond_frame_outputs'][anchor]),index=owner.sam.index,
                gallery=clone_tree(owner.gallery),labs=[v.copy() for v in owner.labs],
                shape=owner.shape_anchor,area=owner.anchor_area,recent=owner.recent_area)
            snapshot['ready']=torch.cuda.Event();snapshot['ready'].record()
            with self.lock:
                self.job=snapshot
            self.wake.set()
            return True
        except Exception:
            with self.lock:
                self.busy=False
            raise

    def poll(self):
        if self.error:
            raise RuntimeError('SAM verification failed: '+self.error)
        with self.lock:
            return self.result

    @torch.inference_mode()
    def run(self):
        stream=torch.cuda.Stream(priority=0)
        while not self.stop.is_set():
            if not self.wake.wait(.05):
                continue
            self.wake.clear()
            with self.lock:
                job,self.job=self.job,None
            if job is None:
                continue
            try:
                tick=time.perf_counter()
                with torch.cuda.stream(stream):
                    stream.wait_event(job['ready'])
                    t=self.tracker;s=t.sam
                    s.cached_features=job['features'];s.index=job['index'];s.anchor_index=job['anchor']
                    s.outputs={'cond_frame_outputs':{job['anchor']:job['bank']},'non_cond_frame_outputs':{}}
                    t.gallery=job['gallery'];t.labs=job['labs'];t.shape_anchor=job['shape']
                    t.anchor_area=job['area'];t.recent_area=job['recent'];t.diag={}
                    value=t.assess(job['frame'],job['mask'],job['iou'],job['logit'])
                    if value is None:
                        probe=s.probe_anchor(job['frame'].shape)
                        value=t.assess(job['frame'],probe['mask'],probe['iou'],probe['object_logit'])
                    if value is None:
                        value=dict(mask=job['mask'],box=[0,0,0,0],score=0.)
                    anchor_reason=t.anchor_confirmation(job['frame'],value)
                    ambiguity_reason=t.ambiguity(job['frame'],value)
                    stream.synchronize()
                    result=dict(epoch=job['epoch'],stamp=job['stamp'],mask=job['mask'],
                        anchor_reason=anchor_reason,ambiguity_reason=ambiguity_reason,
                        second_score=value.get('second_score',0.),anchor_mask_iou=value.get('anchor_mask_iou'),
                        candidate_frame=job['frame'],candidate_masks=[p['mask'] for p in t.proposals[:4]],
                        distractors=t.distractors,
                        diag=t.diag,elapsed_ms=(time.perf_counter()-tick)*1000)
                with self.lock:
                    self.result=result
            except Exception as exc:
                self.error=str(exc)
            finally:
                with self.lock:
                    self.busy=False

    def close(self):
        self.stop.set();self.wake.set();self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('SAM verifier did not stop')
