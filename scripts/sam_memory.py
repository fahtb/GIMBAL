"""SAM encoder prototypes, masked Lab identity gate and bounded SAM memory.

All model operations run in the existing inference worker. The display thread
only consumes completed masks. No other learned model is imported or loaded.
"""
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sam_only import SAMOnlyTracker


def lab_descriptor(frame, mask):
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    a, b, c, d = x.min(), y.min(), x.max()+1, y.max()+1
    patch = cv2.cvtColor(frame[b:d, a:c], cv2.COLOR_BGR2LAB)
    roi_mask = mask[b:d, a:c].astype(np.uint8)
    hist = cv2.calcHist([patch], [0, 1, 2], roi_mask, [8, 8, 8], [0, 256]*3).ravel()
    hist = np.sqrt(hist / max(1, hist.sum()))
    return hist / max(1e-8, np.linalg.norm(hist))


def mask_box(mask):
    y, x = np.nonzero(mask)
    return None if not len(x) else [int(x.min()), int(y.min()), int(x.max()), int(y.max())]


def overlap(a, b):
    if a is None or b is None:
        return 0.
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    inter = float(np.maximum(0, hi-lo+1).prod())
    return inter/max(1, np.prod(np.subtract(a[2:], a[:2])+1)+np.prod(np.subtract(b[2:], b[:2])+1)-inter)


class SAMMemoryTracker(SAMOnlyTracker):
    def __init__(self, sam, config=None, lost_stride=1, debug_search=False):
        super().__init__(sam, True, lost_stride)
        self.sam.policy = 'sam_memory'
        self.cfg = json.loads(Path(config or Path(__file__).with_name('sam_memory.json')).read_text())
        for key, value in self.cfg.items():
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid SAM memory config: {key}')
        self.debug_search = debug_search
        self.reset_identity()

    def reset_identity(self):
        self.gallery, self.labs = [], []
        self.backgrounds = []
        self.lab_backgrounds = []
        self.anchor_area = None
        self.recent_area = None
        self.bad = self.hits = self.probation = self.reacquired = self.attempt_id = 0
        self.confirmed_wrong = self.confirmed_correct = 0
        self.candidate = self.last_box = None
        self.last_good = None
        self.rollback = None
        self.motion = cv2.KalmanFilter(4, 2)
        self.motion.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.motion.processNoiseCov = np.eye(4, dtype=np.float32)*.1
        self.motion.measurementNoiseCov = np.eye(2, dtype=np.float32)*9
        self.motion.errorCovPost = np.eye(4, dtype=np.float32)*100
        self.motion_stamp = self.predicted_center = None
        self.state = 'WAIT_SELECTION'

    def feature_maps(self):
        feats, _, sizes = self.sam.cached_features
        return [f.permute(1, 2, 0).reshape(1, -1, *size).float() for f, size in zip(feats, sizes)]

    def prototype(self, mask):
        weights = torch.as_tensor(mask, device='cuda', dtype=torch.float32)[None, None]
        vectors = []
        for feature in self.feature_maps():
            weight = F.interpolate(weights, feature.shape[-2:], mode='area')
            vectors.append(F.normalize((feature*weight).sum((2, 3))/weight.sum().clamp_min(1e-6), dim=1)[0])
        return vectors

    def similarity(self, proto):
        return max(float(torch.stack([(a*b).sum() for a, b in zip(proto, item)]).mean()) for item in self.gallery)

    def remember(self, proto, lab, frame, mask):
        if self.gallery and self.similarity(proto) >= self.cfg['gallery_duplicate']:
            return
        self.gallery.append([v.detach().clone() for v in proto])
        self.backgrounds.append([F.normalize(f.mean((2,3)),dim=1)[0].detach().clone() for f in self.feature_maps()])
        self.labs.append(lab.copy())
        background = lab_descriptor(frame, ~mask)
        self.lab_backgrounds.append(np.zeros_like(lab) if background is None else background)
        if len(self.gallery) > self.cfg['gallery_size']:
            packed = F.normalize(torch.stack([torch.cat(v) for v in self.gallery]), dim=1)
            redundancy = packed @ packed.T
            redundancy.fill_diagonal_(-1)
            redundancy[0] = -1  # The selected anchor is never evicted.
            remove = int(redundancy.max(dim=1).values.argmax())
            del self.gallery[remove]
            del self.labs[remove]
            del self.backgrounds[remove]
            del self.lab_backgrounds[remove]

    def assess(self, frame, mask, score, logit, stability=1.):
        if mask is None or not np.any(mask):
            return None
        proto, lab = self.prototype(mask), lab_descriptor(frame, mask)
        similarity = self.similarity(proto) if self.gallery else 1.
        color = max(float(lab @ ref) for ref in self.labs) if self.labs else 1.
        ratio = float(mask.sum())/max(1, self.anchor_area or mask.sum())
        recent_ratio = float(mask.sum())/max(1, self.recent_area or mask.sum())
        plausible = self.cfg['area_min_ratio'] <= ratio <= self.cfg['area_max_ratio']
        return dict(mask=mask, box=mask_box(mask), proto=proto, lab=lab, similarity=similarity,
                    color=color, ratio=ratio, recent_ratio=recent_ratio, plausible=plausible, iou=float(score), logit=float(logit),
                    stability=float(stability), score=.45*similarity+.25*color+.25*float(score)+.05*float(stability))

    def update_motion(self, box, stamp):
        center = np.array([[(box[0]+box[2])/2], [(box[1]+box[3])/2]], np.float32)
        if self.motion_stamp is None:
            self.motion.statePost = np.vstack((center, np.zeros((2,1), np.float32)))
            self.motion_stamp = stamp
        else:
            self.motion.correct(center)
        self.predicted_center = center[:, 0]

    def search(self, frame, proposals_only=False):
        heat = None
        for level, feature in enumerate(self.feature_maps()):
            normalized = F.normalize(feature[0], dim=0)
            gallery = torch.stack([item[level] for item in self.gallery])
            background = torch.stack([item[level] for item in self.backgrounds])
            scores = (torch.einsum('gc,chw->ghw', gallery, normalized)
                      -self.cfg['heatmap_background_weight']*torch.einsum('gc,chw->ghw', background, normalized)).max(dim=0).values
            scores = F.interpolate(scores[None, None], (128, 128), mode='bilinear', align_corners=False)[0, 0]
            weights = [self.cfg['heatmap_fine_weight'],self.cfg['heatmap_mid_weight'],self.cfg['heatmap_coarse_weight']]
            scores = scores*weights[level]
            heat = scores if heat is None else heat+scores
        heat = (heat/len(self.feature_maps())).cpu().numpy()
        points = []
        for _ in range(self.cfg['search_points']):
            y, x = np.unravel_index(heat.argmax(), heat.shape)
            points.append([(x+.5)*frame.shape[1]/128, (y+.5)*frame.shape[0]/128])
            heat[max(0,y-6):y+7, max(0,x-6):x+7] = -10
        if self.predicted_center is not None:
            radius = self.cfg['grid_radius'] + self.cfg['grid_growth']*max(0, self.motion_stamp-self.last_good)
            cx, cy = self.predicted_center
            for dx in (-radius, 0, radius):
                for dy in (-radius, 0, radius):
                    x,y = cx+dx,cy+dy
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        points.append([float(x),float(y)])
        # Non-learned color proposals complement SAM cosine peaks for tiny
        # targets. Every proposal still passes SAM decoding and strict identity
        # verification; histogram alone never accepts a target.
        lab = cv2.cvtColor(frame,cv2.COLOR_BGR2LAB).astype(np.int32)//32
        bins = lab[:,:,0]*64+lab[:,:,1]*8+lab[:,:,2]
        lookup = np.maximum.reduce([fg*np.log((fg*fg+1e-4)/(bg*bg+1e-4))
                                    for fg,bg in zip(self.labs,self.lab_backgrounds)])
        color_heat = cv2.GaussianBlur(lookup[bins].astype(np.float32),(5,5),1.)
        color_start = len(points)
        for _ in range(self.cfg['lab_search_points']):
            y,x = np.unravel_index(color_heat.argmax(),color_heat.shape)
            points.append([float(x),float(y)])
            radius = max(4,min(frame.shape[:2])//64)
            color_heat[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1] = -100
            color_heat[bins == bins[y,x]] = -100
        self.last_search_points = points
        masks, scores, logits, stability = self.sam.decode_points(points, frame.shape[:2])
        candidates = []
        self.search_diagnostics = None
        self.last_search_values = []
        for index,(mask, score, logit, stable) in enumerate(zip(masks, scores, logits, stability)):
            value = self.assess(frame, mask, score, logit, stable)
            if value:
                value['proposal_source'] = 'lab_backprojection' if index>=color_start else 'sam_cosine_or_kalman'
                self.last_search_values.append(dict(point=points[index],box=value['box'],**{
                    k:value[k] for k in ('score','similarity','color','ratio','iou','logit','stability','proposal_source')}))
            if value and (self.search_diagnostics is None or value['score'] > self.search_diagnostics['score']):
                self.search_diagnostics = {k:value[k] for k in ('score','similarity','color','ratio','iou','logit','stability')}
            verified = value and (self.search_valid(value) if not proposals_only else
                value['logit'] > 0 and value['iou'] >= self.cfg['search_iou'] and value['plausible']
                and value['similarity'] >= self.cfg['tile_proposal_similarity']
                and value['color'] >= self.cfg['tile_proposal_lab'])
            if verified:
                if self.predicted_center is not None:
                    a,b,c,d = value['box']
                    distance = np.linalg.norm(np.array([(a+c)/2,(b+d)/2])-self.predicted_center)
                    value['score'] -= self.cfg['position_penalty']*min(1.,distance/max(frame.shape[:2]))
                if not any(overlap(value['box'], old['box']) > .7 for old in candidates):
                    candidates.append(value)
        return sorted(candidates, key=lambda v: v['score'], reverse=True)

    def search_valid(self, value):
        c = self.cfg
        small = getattr(self, 'small_mode', False)
        return (value['logit'] > 0 and value['iou'] >= c['search_iou'] and value['plausible']
                and value['stability'] >= (c['small_search_stability'] if small else c['search_stability'])
                and value['similarity'] >= c['search_similarity']
                and value['color'] >= (c['small_search_lab'] if small else c['search_lab']))

    def rollback_candidate(self):
        if self.rollback is not None:
            self.sam.outputs, self.sam.pinned_quality, self.sam.active_prompt_index, self.recent_area = self.rollback
            self.rollback = None
        self.probation = 0

    @torch.inference_mode()
    def recenter_crop(self, frame, stamp, box):
        """Verify a new spatial window against the existing identity first.

        Old recent spatial memory is discarded only on success. The original
        SAM conditioning anchor and entire appearance gallery always survive.
        """
        tick, before = time.perf_counter(), self.state
        old_box, old_shape = self.sam.last_box, self.sam.shape
        self.sam.reset_motion()
        result = self.sam.step(frame, stamp, box, continue_lost=True, preserve_memory=True)
        value = self.assess(frame, result.get('mask'), result.get('iou', 0), result.get('object_logit', -1))
        if value is None or not self.search_valid(value):
            self.sam.last_box, self.sam.shape = old_box, old_shape
            self.sam.reset_motion()
            self.state, self.lost_frames, self.hits = 'SEARCHING', 0, 0
            return self.decorate(result, before, 'crop_recenter_identity_rejected', tick, value)
        anchor = self.sam.anchor_index
        self.sam.outputs = {'cond_frame_outputs': {anchor: self.sam.outputs['cond_frame_outputs'][anchor]},
                            'non_cond_frame_outputs': {}}
        self.sam.pinned_quality = {}
        self.sam.commit_memory(self.cfg['pinned_frames'], pin=False, condition=True)
        self.last_good, self.last_box = stamp, value['box']
        self.bad = 0
        # Crop coordinates changed, so the classical position prior must reset.
        self.motion_stamp = None
        self.update_motion(value['box'], stamp)
        return self.decorate(result, before, 'crop_recenter_identity_verified_anchor_retained', tick, value)

    @torch.inference_mode()
    def step(self, frame, stamp, box=None):
        tick, before = time.perf_counter(), self.state
        if box is not None:
            self.reset_identity()
        if self.motion_stamp is not None:
            dt = max(.001, stamp-self.motion_stamp)
            self.motion.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float32)
            self.predicted_center = self.motion.predict()[:2,0]
            self.motion_stamp = stamp
        searching = box is None and before in ('LOST', 'SEARCHING')
        result = dict(mask=None, box=None, dx=None, dy=None, inference_ms=0., predictor_called=False)
        if box is None and not self.gallery:
            self.state = 'LOST'
            return self.decorate(result, before, 'initial_mask_failed_select_again', tick)
        if searching:
            self.lost_frames += 1
            if (self.lost_frames-1) % self.lost_stride:
                self.state = 'SEARCHING'
                return self.decorate(result, before, 'lost_stride_skip', tick)
        result = self.sam.step(frame, stamp, box, continue_lost=True)
        value = self.assess(frame, result.get('mask'), result.get('iou', 0), result.get('object_logit', -1))
        if box is not None:
            if value and value['logit'] > 0 and value['iou'] >= self.sam.min_iou:
                self.anchor_area = int(value['mask'].sum())
                self.recent_area = self.anchor_area
                self.remember(value['proto'], value['lab'], frame, value['mask'])
                self.sam.commit_memory(self.cfg['pinned_frames'])
                self.state, self.last_good, self.last_box = 'TRACKING', stamp, value['box']
                self.update_motion(value['box'], stamp)
                return self.decorate(result, before, 'anchor_initialized', tick, value)
            self.state = 'LOST'
            return self.decorate(result, before, 'initial_mask_failed_select_again', tick)
        if not searching:
            score_min = self.cfg['search_iou'] if before == 'TRACKING_PROBATION' else self.cfg['track_score_low']
            good = (value is not None and value['logit'] > 0 and value['iou'] >= score_min
                    and value['similarity'] >= self.cfg['track_similarity'] and value['color'] >= self.cfg['track_lab']
                    and value['plausible'] and value['recent_ratio'] >= self.cfg['area_collapse_ratio'])
            self.bad = (0 if good and value['iou'] >= self.cfg['track_score_high'] else
                        self.bad if good else self.bad+1)
            if not good and (self.bad >= self.cfg['bad_frames'] or value is None or before == 'TRACKING_PROBATION'):
                self.rollback_candidate()
                self.state, self.lost_frames, self.hits = 'LOST', 0, 0
                failures = ['empty_mask'] if value is None else [name for condition,name in (
                    (value['logit'] <= 0, 'object_logit_nonpositive'), (value['iou'] < score_min, 'mask_iou_low'),
                    (value['similarity'] < self.cfg['track_similarity'], 'prototype_similarity_low'),
                    (value['color'] < self.cfg['track_lab'], 'lab_similarity_low'),
                    (not value['plausible'], 'area_ratio_outside_limits'),
                    (value['recent_ratio'] < self.cfg['area_collapse_ratio'], 'recent_mask_area_collapse')) if condition]
                return self.decorate(result, before, '|'.join(failures), tick, value)
            if good:
                self.last_good, self.last_box = stamp, value['box']
                self.update_motion(value['box'], stamp)
                if self.probation:
                    self.probation -= 1
                    self.state = 'TRACKING_PROBATION' if self.probation else 'TRACKING'
                    if not self.probation:
                        self.reacquired += 1
                        self.rollback = None
                if (not self.probation and value['iou'] >= self.cfg['memory_iou'] and value['logit'] >= self.cfg['memory_logit']
                        and value['similarity'] >= self.cfg['search_similarity'] and value['color'] >= self.cfg['search_lab']):
                    self.sam.commit_memory(self.cfg['pinned_frames'])
                    self.remember(value['proto'], value['lab'], frame, value['mask'])
                    self.recent_area = .9*self.recent_area+.1*int(value['mask'].sum())
            return self.decorate(result, before, 'sam_identity_verified' if good else 'weak_observation_not_memorized', tick, value)
        self.state = 'SEARCHING'
        candidates = [value] if value and self.search_valid(value) else []
        if self.cfg.get('heatmap_search', True) and (self.lost_frames-1) % self.cfg['search_every'] == 0:
            for candidate in self.search(frame):
                if not any(overlap(candidate['box'], item['box']) > .7 for item in candidates):
                    candidates.append(candidate)
        elif self.cfg.get('heatmap_search', True) and not candidates and self.candidate is not None:
            a,b,c,d = self.candidate
            masks,scores,logits,stability = self.sam.decode_points([[(a+c)/2,(b+d)/2]], frame.shape[:2])
            candidate = self.assess(frame, masks[0], scores[0], logits[0], stability[0])
            if candidate and self.search_valid(candidate):
                candidates.append(candidate)
        candidates.sort(key=lambda v: v['score'], reverse=True)
        if not candidates:
            self.hits, self.candidate = 0, None
            return self.decorate(result, before, 'no_verified_sam_candidate', tick, value)
        best = candidates[0]
        second = candidates[1]['score'] if len(candidates) > 1 else 0.
        required = (self.cfg['strong_streak'] if best['score'] >= self.cfg['accept_strong'] else
                    self.cfg['normal_streak'] if best['score'] >= self.cfg['accept_normal'] and best['score']-second >= self.cfg['margin'] else
                    self.cfg['single_streak'] if len(candidates) == 1 and best['score'] >= self.cfg['accept_low'] else None)
        self.hits = self.hits+1 if required and overlap(best['box'], self.candidate) >= .2 else 1 if required else 0
        self.candidate = best['box'] if required else None
        result.update(search_top2=second, search_hits=self.hits, candidate_box=best['box'])
        if self.debug_search:
            result['search_masks'] = [v['mask'] for v in candidates]
        if required and self.hits >= required:
            seeded = self.seed_candidate(frame, stamp, best)
            if seeded is not None:
                return seeded
            self.hits = 0
        return self.decorate(result, before, 'candidate_streak_pending', tick, best)

    @torch.inference_mode()
    def seed_candidate(self, frame, stamp, best):
        tick, before = time.perf_counter(), self.state
        # Mask prompt uses cached features; neither gallery nor anchor resets.
        self.sam.reset_motion()
        seeded = self.sam.step(frame, stamp, continue_lost=True, mask_prompt=best['mask'], reuse_features=True)
        seeded_value = self.assess(frame, seeded.get('mask'), seeded.get('iou', 0), seeded.get('object_logit', -1))
        if not seeded_value or not self.search_valid(seeded_value):
            return None
        self.rollback = ({key: dict(bank) for key, bank in self.sam.outputs.items()}, dict(self.sam.pinned_quality),
                         self.sam.active_prompt_index, self.recent_area)
        index,out,bbox,_,_ = self.sam.pending_memory
        self.sam.pending_memory = index,out,bbox,best['iou'],best['logit']
        self.sam.commit_memory(self.cfg['pinned_frames'], pin=False, condition=True)
        seeded.update(iou=best['iou'], predicted_iou=best['iou'], object_logit=best['logit'],
                      object_score_logits=best['logit'])
        self.probation, self.state = self.cfg['probation_frames'], 'TRACKING_PROBATION'
        self.recent_area = int(best['mask'].sum())
        self.attempt_id += 1
        self.hits = 0
        return self.decorate(seeded, before, 'sam_memory_candidate_probation', tick, best)

    def decorate(self, result, before, reason, tick, value=None):
        valid = self.state in ('TRACKING', 'TRACKING_PROBATION') and result.get('mask') is not None
        result.update(status=self.state, valid=valid, target_mode='SAM_ONLY', transition=f'{before}->{self.state}',
                      sam_observation_valid=(result.get('mask') is not None) if result.get('predictor_called') else None,
                      keep_running=True,
                      transition_reason=reason, probation=self.state == 'TRACKING_PROBATION',
                      gallery_size=len(self.gallery), pinned_memory=len(self.sam.outputs['cond_frame_outputs']),
                      reid_top1=None if value is None else value['similarity'],
                      lab_similarity=None if value is None else value['color'],
                      identity_area_ratio=None if value is None else value['ratio'],
                      reacquired=self.reacquired, attempt_id=self.attempt_id, search_hits=self.hits,
                      confirmed_wrong=self.confirmed_wrong, confirmed_correct=self.confirmed_correct,
                      thresholds=self.cfg, pipeline_ms=(time.perf_counter()-tick)*1000)
        if not valid:
            result.update(mask=None, box=None, dx=None, dy=None)
        return result

    def label(self, correct, stamp, attempt_id=None):
        if not self.attempt_id or (attempt_id is not None and attempt_id != self.attempt_id):
            return False
        self.confirmed_correct += bool(correct)
        self.confirmed_wrong += not correct
        if not correct:
            self.rollback_candidate()
            self.state, self.hits = 'LOST', 0
        return True
