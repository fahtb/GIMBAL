"""Immutable target identity and verified appearance gallery, outside SAM state."""
import uuid
import cv2
import numpy as np
from small_target import bounds,appearance,structural_match
from recovery.proposals import CandidateGenerator,overlap

class IdentityMemory:
    def __init__(self,frame,mask):
        self.target_id=uuid.uuid4().hex[:12]
        self.mask=mask.copy();self.box=bounds(mask)
        x,y,r,b=map(int,self.box)
        self.crop=frame[y:b,x:r].copy()
        self.template=cv2.cvtColor(self.crop,cv2.COLOR_BGR2GRAY)
        self.feature,self.texture=appearance(frame,mask)
        self.views=[]
        self.template_cache={}
        self.generator=CandidateGenerator()
        self.search_diagnostics={}

    def update(self,frame,mask,cfg):
        feature,texture=appearance(frame,mask)
        if texture<6 or structural_match(feature,self.feature)<cfg.gallery_update_similarity:return
        if max(structural_match(feature,ref) for ref in [self.feature,*[v[1] for v in self.views]])>=.97:return
        x,y,r,b=map(int,bounds(mask))
        self.views.append((cv2.cvtColor(frame[y:b,x:r],cv2.COLOR_BGR2GRAY).copy(),feature.copy()))
        self.views=self.views[-max(0,cfg.gallery_size-1):] if cfg.gallery_size>1 else []
        self.template_cache.clear()

    def similarity(self,feature):
        return max(structural_match(np.rot90(feature.reshape(24,24),k).ravel(),ref)
                   for k in range(4) for ref in [self.feature,*[v[1] for v in self.views]])

    def proposals(self,frame,cfg,predicted=None,local=False,radius_px=None,rotation=0):
        return self.generator.generate(self,frame,cfg,predicted,local,radius_px,rotation)

    def competing_score(self,frame,cfg,candidate):
        # Before accepting, compare different physical locations across rotations.
        others=[]
        for rotation in range(4):
            if rotation==candidate.get('rotation',0):continue
            others.extend(c['score'] for c in self.proposals(frame,cfg,rotation=rotation)
                          if overlap(c['box'],candidate['box'])<.25)
        return max(others) if others else None

