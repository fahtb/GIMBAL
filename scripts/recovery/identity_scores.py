"""Classical geometry and transparent confidence; color never decides identity."""
import math
import numpy as np


def relation(a,b):
    if a is None or b is None or a.shape!=b.shape:
        return dict(iou=0.,containment=0.)
    inter=int(np.logical_and(a,b).sum());aa=int(a.sum());bb=int(b.sum())
    return dict(iou=inter/max(1,aa+bb-inter),containment=inter/max(1,min(aa,bb)))


def same_support(a,b,cfg):
    r=relation(a,b)
    return r['iou']>=cfg.get('granularity_iou',.5) or r['containment']>=cfg.get('granularity_containment',.85)


def granularity_score(value,reference,cfg):
    r=relation(value['mask'],reference['mask'])
    area=min(value['mask'].sum(),reference['mask'].sum())/max(1,max(value['mask'].sum(),reference['mask'].sum()))
    return .55*r['iou']+.2*area+.15*value.get('shape_ratio',0)+.1*value.get('iou',0)


def fuse(value,diag,cfg):
    if value is None:
        return 0.,{}
    # Evidence score, NOT a calibrated probability of correct identity.
    parts=dict(sam_iou=np.clip(value['iou'],0,1),
               objectness=1/(1+math.exp(-float(np.clip(value['logit'],-30,30)))),
               continuity=np.clip(diag.get('flow_mask_iou',0),0,1),
               area_stability=math.exp(-abs(math.log(max(1e-6,diag.get('temporal_area_ratio',1.))))),
               prototype=np.clip(value.get('anchor_similarity',0),0,1),
               margin=np.clip((value['score']-value.get('second_score',0))/max(cfg['margin'],1e-6),0,1),
               motion=math.exp(-diag.get('kalman_innovation',diag.get('continuity_distance',0))/max(1.,diag.get('kalman_limit',diag.get('continuity_limit',1.)))))
    weights=dict(sam_iou=.15,objectness=.1,continuity=.25,area_stability=.1,prototype=.2,margin=.1,motion=.1)
    return float(sum(weights[k]*v for k,v in parts.items())),{k:float(v) for k,v in parts.items()}
