"""Bounded native tiles on the existing SAM inference worker, never UI thread."""
import time
import cv2
import numpy as np
import torch
from sam_memory import overlap


def tile_windows(shape, side, overlap_fraction, center):
    h,w = shape[:2]
    cw,ch = min(w,side),min(h,side)
    stride = max(1,round(side*(1-overlap_fraction)))
    xs = sorted(set([*range(0,w-cw+1,stride),w-cw]))
    ys = sorted(set([*range(0,h-ch+1,stride),h-ch]))
    windows = [[x,y,x+cw,y+ch] for y in ys for x in xs]
    return sorted(windows, key=lambda b: np.linalg.norm(np.array([(b[0]+b[2])/2,(b[1]+b[3])/2])-center))


class TileSearch:
    def __init__(self, tracker):
        self.core = tracker
        self.last_stamp = -float('inf')
        self.cursor = self.hits = 0
        self.pending_window = self.pending_box = None
        self.last_debug = []

    @torch.inference_mode()
    def step(self, original, stamp, current_window, last_box):
        cfg,core = self.core.cfg,self.core
        if not core.gallery or stamp-self.last_stamp < 1/cfg['tile_hz']:
            return None, None, {}
        self.last_stamp = stamp
        tick = time.perf_counter()
        center = (np.array(last_box[:2])+np.array(last_box[2:]))/2 if last_box else np.array(original.shape[1::-1])/2
        windows = tile_windows(original.shape,cfg['tile_side'],cfg['tile_overlap'],center)
        selected = [self.pending_window] if self.pending_window else []
        for _ in range(min(cfg['tile_budget'],len(windows))):
            window = windows[self.cursor % len(windows)]
            self.cursor += 1
            if window not in selected:
                selected.append(window)
            if len(selected) >= cfg['tile_budget']:
                break
        candidates = []
        diagnostics = []
        predicted = core.predicted_center
        try:
            for window in selected:
                a,b,c,d = window
                tile = original[b:d,a:c]
                core.sam.encode_features(tile)
                # Convert the classical prior to the tile's coordinates. The
                # global feature peaks still cover the tile when the prior misses.
                core.predicted_center = center-np.array([a,b])
                found = core.search(tile, proposals_only=True)
                if core.search_diagnostics is not None:
                    diagnostics.append(core.search_diagnostics)
                if found:
                    feats,pos,sizes = core.sam.cached_features
                    snapshot = ([f.clone() for f in feats],pos,sizes)
                    for item in found:
                        world = [float(v) for v in np.asarray(item['box'])+np.array([a,b,a,b])]
                        if not any(overlap(world,old['world']) > .7 for old in candidates):
                            candidates.append(dict(item,world=world,window=window,features=snapshot))
        finally:
            core.predicted_center = predicted
        candidates.sort(key=lambda v:v['score'],reverse=True)
        diagnostics.extend(dict(phase='coarse', world=item['world'], **{
            k:item[k] for k in ('score','similarity','color','iou','stability')}) for item in candidates)
        # A 512px tile changes object scale relative to its 256px anchor crop.
        # Coarse matches are proposals only: re-encode a NEW tight crop, then
        # apply the unchanged strict identity gates before any acceptance.
        refined = []
        h,w = original.shape[:2]
        for item in candidates[:cfg['tile_refine_budget']]:
            x,y,u,v = item['world']
            side = round(max(cfg['crop_min_size'],max(u-x,v-y)*cfg['crop_scale']))
            cw,ch = min(w,side),min(h,side)
            a = int(np.clip(round((x+u-cw)/2),0,w-cw))
            b = int(np.clip(round((y+v-ch)/2),0,h-ch))
            window = [a,b,a+cw,b+ch]
            tile = original[b:b+ch,a:a+cw]
            if window == item['window']:
                core.sam.cached_features = item['features']
            else:
                core.sam.encode_features(tile)
            # A box center can fall on background or a thin limb. Reuse this
            # encoding for a batch of interior prompts from the coarse SAM mask.
            distance = cv2.distanceTransform(item['mask'].astype(np.uint8),cv2.DIST_L2,3)
            py,px = np.unravel_index(distance.argmax(),distance.shape)
            yy,xx = np.nonzero(item['mask'])
            ox,oy = item['window'][:2]
            points = [[(x+u)/2-a,(y+v)/2-b], [px+ox-a,py+oy-b],
                      [float(xx.mean())+ox-a,float(yy.mean())+oy-b]]
            masks,ious,logits,stable = core.sam.decode_points(points,tile.shape[:2])
            verified = []
            for mask,iou,logit,stability in zip(masks,ious,logits,stable):
                value = core.assess(tile,mask,iou,logit,stability)
                if value:
                    diagnostics.append(dict(phase='refined',**{k:value[k] for k in ('score','similarity','color','ratio','iou','logit','stability')}))
                if value and core.search_valid(value):
                    verified.append(value)
            if verified:
                value = max(verified,key=lambda v:v['score'])
                feats,pos,sizes = core.sam.cached_features
                world = [float(v) for v in np.asarray(value['box'])+np.array([a,b,a,b])]
                refined.append(dict(value,world=world,window=window,features=([f.clone() for f in feats],pos,sizes)))
        candidates = sorted(refined,key=lambda v:v['score'],reverse=True)
        self.last_debug = [(v['window'],v['mask']) for v in candidates] if core.debug_search else []
        info = dict(tile_count=len(selected),tile_search_ms=(time.perf_counter()-tick)*1000,
                    tile_total=len(windows),tile_hits=0,tile_candidate_debug=diagnostics)
        if not candidates:
            self.hits = 0
            self.pending_window = self.pending_box = None
            return None,None,info
        best = candidates[0]
        second = candidates[1]['score'] if len(candidates)>1 else 0.
        required = (None if len(candidates)>1 and best['score']-second<cfg['margin'] else
                    cfg['strong_streak'] if best['score']>=cfg['accept_strong'] else
                    cfg['normal_streak'] if best['score']>=cfg['accept_normal'] and best['score']-second>=cfg['margin'] else
                    cfg['single_streak'] if len(candidates)==1 and best['score']>=cfg['accept_low'] else None)
        self.hits = self.hits+1 if required and overlap(best['world'],self.pending_box)>=.2 else 1 if required else 0
        self.pending_window,self.pending_box = best['window'],best['world']
        info.update(tile_hits=self.hits,tile_similarity=best['similarity'],tile_top2=second)
        if required and self.hits>=required:
            core.sam.cached_features = best['features']
            a,b,c,d = best['window']
            seeded = core.seed_candidate(original[b:d,a:c],stamp,best)
            if seeded is not None:
                self.hits = 0
                self.pending_window = self.pending_box = None
                seeded.update(info,transition_reason='native_tile_candidate_probation')
                return seeded,best['window'],info
        return None,None,info
