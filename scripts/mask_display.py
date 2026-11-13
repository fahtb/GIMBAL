"""Timestamp-aligned SAM masks, foreground LK and ROI-only compositing."""
from collections import OrderedDict
import time
import cv2
import numpy as np
from live_motion import FlowState, gray_frame
from video_profile import timed, sample


def affine3(before, after):
    out = np.eye(3)
    out[:2] = cv2.getAffineTransform(before[:3].astype(np.float32), after[:3].astype(np.float32))
    return out


class MaskDisplay:
    def __init__(self, history=90, max_age=1., flow_max_side=0, enabled=True):
        self.enabled=enabled
        self.capacity, self.max_age = history, max_age
        self.flow_max_side=flow_max_side
        self.reset()

    def reset(self):
        self.person = None
        self.flow = self.anchor = self.patch = self.patch_origin = None
        self.anchor_corners = None
        self.history = OrderedDict()
        self.geometry = {}
        self.last_sam = -1
        self.latest = None
        self.previous_mask = None
        self.fade = 0

    @timed('display_flow')
    def step(self, packet, sam):
        if not self.enabled:
            result=dict(sam or {})
            result.update(status=result.get('status','IDLE'),estimated=False,
                          age_ms=max(0,packet[1]-result.get('captured',packet[1]))*1000)
            if not sam or result['age_ms']>self.max_age*1000:
                result.update(mask=None,box=None,dx=None,dy=None,valid=False)
            return result
        if self.person is not None or (sam and (sam.get('small') or sam.get('target_mode') in ('PERSON', 'SAM_ONLY'))):
            if self.person is None:
                self.person = DenseMaskTransport(self.capacity,self.max_age,self.flow_max_side)
            return self.person.step(packet,sam)
        seq, stamp, frame = packet
        gray = gray_frame(frame)
        is_new = seq not in self.history
        if is_new:
            self.history[seq] = (stamp, gray)
            while len(self.history) > self.capacity:
                old, _ = self.history.popitem(last=False)
                self.geometry.pop(old, None)
        if self.flow is not None and self.flow.seq != seq:
            before = self.flow.corners.copy()/self.flow.scale
            if self.flow.advance(seq, stamp, gray):
                after = self.flow.corners/self.flow.scale
                self.geometry[seq] = affine3(before, after) @ self.geometry.get(self.last_seq, np.eye(3))
            else:
                self.flow = None
        self.last_seq = seq
        if sam is not None and self.last_sam < sam['seq'] <= seq:
            self.last_sam = sam['seq']
            self.latest = sam
            if sam.get('box') is None or sam.get('mask') is None:
                self.flow = self.patch = None
            elif stamp-sam['captured'] <= self.max_age:
                candidate = FlowState(sam['frame'],sam['mask'],sam['box'],sam['seq'],sam['captured'])
                transform = None
                if self.flow is not None and sam['seq'] in self.geometry and seq in self.geometry:
                    transform = self.geometry[seq] @ np.linalg.inv(self.geometry[sam['seq']])
                else:
                    origin_corners = candidate.corners.copy()/candidate.scale
                    for old_seq,(old_stamp,old_gray) in self.history.items():
                        if old_seq > candidate.seq and not candidate.advance(old_seq,old_stamp,old_gray):
                            break
                    if candidate.valid and candidate.seq == seq:
                        transform = affine3(origin_corners, candidate.corners/candidate.scale)
                if transform is not None:
                    y,x = np.nonzero(sam['mask'])
                    self.patch_origin = np.array([x.min(),y.min()])
                    self.patch = sam['mask'][y.min():y.max()+1,x.min():x.max()+1].astype(np.uint8)*255
                    self.anchor = transform
                    full, _, _ = self.warp(frame.shape[:2], transform)
                    ys,xs = np.nonzero(full)
                    if len(xs) >= 16:
                        box = [int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]
                        self.flow = FlowState(frame,full,box,seq,stamp)
                        self.anchor_corners = self.flow.corners.copy()/self.flow.scale
                        self.geometry[seq] = self.geometry.get(seq,np.eye(3))
                        self.fade = 2 if self.previous_mask is not None else 0
                else:
                    self.flow = self.patch = None
        base = self.latest or {}
        result = {k:v for k,v in base.items() if k not in ('frame','mask')}
        result.update(mask=None, box=None, dx=None,dy=None,valid=False, status=base.get('status','WAIT_SELECTION'),
                      sam_status=base.get('status','WAIT_SELECTION'),
                      sam_seq=base.get('seq'), source_frame=seq, estimated=True)
        if self.latest:
            result['age_ms'] = max(0.,stamp-self.latest['captured'])*1000
            result['sam_latency_ms'] = self.latest.get('latency_ms',0.)
        if self.flow is not None and self.patch is not None and result.get('age_ms',0) <= self.max_age*1000:
            now_corners = self.flow.corners/self.flow.scale
            transform = affine3(self.anchor_corners,now_corners) @ self.anchor
            mask, roi, binary = self.warp(frame.shape[:2],transform)
            if np.any(binary):
                bx,by,bw,bh = cv2.boundingRect(binary)
                x,y = roi
                box = [x+bx,y+by,x+bx+bw-1,y+by+bh-1]
                result.update(mask=mask, mask_roi=binary, roi_origin=roi,box=box,
                              dx=(box[0]+box[2]-frame.shape[1])/2,dy=(box[1]+box[3]-frame.shape[0])/2,
                              valid=bool(base.get('valid',True)),flow_valid=True)
                if self.fade and self.previous_mask is not None:
                    result['blend_previous'] = self.previous_mask
                    result['blend_weight'] = .75 if self.fade == 2 else .9
                    self.fade -= 1
                self.previous_mask = mask
        elif result['status'] in ('TRACKING','TRACKING_LOW','REACQUIRE','TRACKING_PROBATION','SMALL'):
            result['status'] = 'FLOW_UNCERTAIN'
            result['transition_reason'] = 'waiting_for_fresh_sam_or_motion_alignment'
        if not result.get('valid') or result.get('is_confirmed') is False:
            result.update(dx=None,dy=None,control_gain=0.)
        return result

    @timed('mask_warp')
    def warp(self, shape, transform):
        h,w = shape
        ph,pw = self.patch.shape
        x,y = self.patch_origin
        corners = np.array([[x,y,1],[x+pw,y,1],[x+pw,y+ph,1],[x,y+ph,1]]) @ transform.T
        x1,y1 = np.maximum(0,np.floor(corners[:,:2].min(0))-2).astype(int)
        x2,y2 = np.minimum([w,h],np.ceil(corners[:,:2].max(0))+2).astype(int)
        full = np.zeros(shape,bool)
        if x2 <= x1 or y2 <= y1:
            return full,(0,0),np.zeros((1,1),np.uint8)
        local = transform.copy()
        local[:2,2] += local[:2,:2] @ self.patch_origin - [x1,y1]
        binary = cv2.warpAffine(self.patch,local[:2],(x2-x1,y2-y1),flags=cv2.INTER_LINEAR)
        binary = (binary >= 128).astype(np.uint8)*255
        full[y1:y2,x1:x2] = binary > 0
        return full,(int(x1),int(y1)),binary


class DenseMaskTransport:
    """Classical ROI flow for non-rigid or tiny masks without an eight-corner gate."""
    def __init__(self,capacity,max_age,flow_max_side=0):
        self.capacity,self.max_age = min(capacity,45),max_age
        self.flow_max_side=flow_max_side
        self.engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.previous_gray = self.probability = self.latest = None
        self.seq = self.last_sam = -1
        self.frames = OrderedDict()

    @timed('dense_mask_warp')
    def warp(self,probability,current_gray,previous_gray):
        h,w = probability.shape
        x,y,bw,bh = cv2.boundingRect((probability>.05).astype(np.uint8))
        if not bw or not bh:
            return np.zeros_like(probability)
        margin = 48
        x1,y1,x2,y2 = max(0,x-margin),max(0,y-margin),min(w,x+bw+margin),min(h,y+bh+margin)
        # Native-resolution ROI preserves hand/foot motion without computing
        # dense flow over the background or resizing a full-frame flow field.
        current=current_gray[y1:y2,x1:x2].copy();previous=previous_gray[y1:y2,x1:x2].copy()
        rh,rw=current.shape
        scale=min(1.,self.flow_max_side/max(rh,rw)) if self.flow_max_side else 1.
        if scale<1.:
            size=(max(12,round(rw*scale)),max(12,round(rh*scale)))
            reduced=self.engine.calc(cv2.resize(current,size,interpolation=cv2.INTER_AREA),
                                     cv2.resize(previous,size,interpolation=cv2.INTER_AREA),None)
            full_flow=cv2.resize(reduced,(rw,rh),interpolation=cv2.INTER_LINEAR)
            full_flow[:,:,0]*=rw/size[0];full_flow[:,:,1]*=rh/size[1]
        else:
            full_flow = self.engine.calc(current,previous,None)
        yy,xx = np.mgrid[y1:y2,x1:x2].astype(np.float32)
        out = np.zeros_like(probability)
        out[y1:y2,x1:x2] = cv2.remap(probability,xx+full_flow[:,:,0],yy+full_flow[:,:,1],cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
        return out

    @timed('dense_flow')
    def step(self,packet,sam):
        seq,stamp,frame = packet
        h,w = frame.shape[:2]
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        previous_gray = self.previous_gray
        changed = seq != self.seq
        if seq != self.seq:
            self.frames[seq] = gray
            while len(self.frames)>self.capacity:
                self.frames.popitem(last=False)
            self.previous_gray,self.seq = gray,seq
        previous = None if self.probability is None else self.probability>=.5
        if sam is not None and self.last_sam < sam['seq'] <= seq:
            self.latest,self.last_sam = sam,sam['seq']
            if sam.get('mask') is None or stamp-sam['captured']>self.max_age:
                self.probability = None
            else:
                self.probability = sam['mask'].astype(np.float32)
                anchor_gray = cv2.cvtColor(sam['frame'],cv2.COLOR_BGR2GRAY)
                for old_seq,old_gray in self.frames.items():
                    if old_seq>sam['seq']:
                        self.probability = self.warp(self.probability,old_gray,anchor_gray)
                        anchor_gray = old_gray
        elif changed and previous_gray is not None and self.probability is not None:
            self.probability = self.warp(self.probability,gray,previous_gray)
        base=self.latest or {}
        result={k:v for k,v in base.items() if k not in ('frame','mask')}
        result.update(mask=None,box=None,dx=None,dy=None,valid=False,status=base.get('status','WAIT_SELECTION'),
                      sam_status=base.get('status','WAIT_SELECTION'),sam_seq=base.get('seq'),estimated=True,
                      age_ms=max(0,stamp-base.get('captured',stamp))*1000,
                      sam_latency_ms=base.get('latency_ms',0.),motion_reason='nonrigid_dense_foreground_flow')
        # A full blank/occluded image is not evidence for extrapolating a person.
        visible = bool(np.std(gray)>1 or np.mean(gray)>3)
        if self.probability is not None and result['age_ms']<=self.max_age*1000 and visible:
            mask = self.probability>=.5
            x,y,bw,bh=cv2.boundingRect(mask.astype(np.uint8))
            if bw*bh >= (1 if base.get('small') else 16):
                box=[x,y,x+bw-1,y+bh-1]
                result.update(mask=mask,box=box,dx=(box[0]+box[2]-w)/2,dy=(box[1]+box[3]-h)/2,
                              valid=base.get('valid',True),flow_valid=True)
                if previous is not None and sam is not None and sam['seq']==self.last_sam:
                    result.update(blend_previous=previous,blend_weight=.9)
        if result['box'] is None and result['status'] in ('TRACKING','TRACKING_LOW','REACQUIRE','TRACKING_PROBATION','SMALL'):
            result['status']='FLOW_UNCERTAIN'
        if not result.get('valid') or result.get('is_confirmed') is False:
            result.update(dx=None,dy=None,control_gain=0.)
        return result


def overlay(frame,result,quality='full',hud_bottom=150):
    if not result.get('valid') and result.get('candidate_frame') is not None:
        frame=result['candidate_frame']
    view = frame.copy()
    h,w = frame.shape[:2]
    cv2.drawMarker(view,(w//2,h//2),(255,255,255),cv2.MARKER_CROSS,18)
    from recovery.selection import draw_candidates
    draw_candidates(view,result,hud_bottom=hud_bottom)
    if result.get('candidate_frame') is not None and not result.get('valid'):
        cv2.putText(view,'CANDIDATES: last SAM probe frame | 1-9 select',(10,max(18,h-12)),0,.45,(0,220,255),1)
    mask,box = result.get('mask'),result.get('box')
    color = (0, 180, 255) if result.get('probation') or result.get('status') in ('UNCERTAIN','RECOVERED') else (0, 254, 0)
    for candidate in result.get('search_masks', []):
        ys,xs = np.nonzero(candidate)
        if len(xs):
            a,b,c,d = xs.min(),ys.min(),xs.max()+1,ys.max()+1
            roi = view[b:d,a:c]
            tint = np.empty_like(roi)
            tint[:] = (0, 180, 255)
            cv2.copyTo(cv2.addWeighted(roi,.85,tint,.15,0), candidate[b:d,a:c].astype(np.uint8), roi)
    if mask is not None and box is not None:
        x1,y1,x2,y2 = map(int,box)
        x1,y1,x2,y2 = max(0,x1-2),max(0,y1-2),min(w,x2+3),min(h,y2+3)
        tick = time.perf_counter()
        binary = mask[y1:y2,x1:x2].astype(np.uint8)*255
        if quality == 'fast' and min(binary.shape) > 8:
            small = cv2.resize(binary,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
            binary = (cv2.resize(small,(binary.shape[1],binary.shape[0]))>=128).astype(np.uint8)*255
        roi = view[y1:y2,x1:x2]
        tint = np.empty_like(roi)
        tint[:] = color
        blended = cv2.addWeighted(roi,.65,tint,.35,0)
        cv2.copyTo(blended,binary,roi)
        old = result.get('blend_previous')
        if old is not None:
            # A short visual cross-fade; geometry/telemetry stays on the new mask.
            previous = old[y1:y2,x1:x2].astype(np.uint8)*255
            edge = cv2.bitwise_and(previous, cv2.bitwise_not(binary))
            faded = cv2.addWeighted(roi,result['blend_weight'],blended,1-result['blend_weight'],0)
            cv2.copyTo(faded,edge,roi)
        sample('mask_overlay',time.perf_counter()-tick)
        tick = time.perf_counter()
        contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.polylines(view,[c+np.array([[[x1,y1]]]) for c in contours],True,color,1,cv2.LINE_AA)
        sample('contour',time.perf_counter()-tick)
        a,b,c,d = map(int,box)
        cv2.rectangle(view,(a,b),(c,d),color,2)
        radius = max(1, min(4, round(min(c-a+1,d-b+1)/6)))
        cv2.circle(view,((a+c)//2,(b+d)//2),radius,(0,0,255),-1)
    for key,color in (('predicted_box',(255,180,0)),('candidate_box',(0,180,255))):
        if result.get(key) is not None:
            a,b,c,d = map(round,result[key])
            cv2.rectangle(view,(a,b),(c,d),color,1)
    return view
