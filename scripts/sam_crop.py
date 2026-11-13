"""Native-resolution fixed-window SMALL transport, independent of display resize.

Window A: retain SAM memory inside a fixed native crop. Recenter at a crop edge
only while confidently TRACKING; reset temporal spatial memory, retain identity
prototypes and the user's original anchor crop. Global tiled recovery is separate.
"""
import cv2
import numpy as np


class NativeCropTracker:
    def __init__(self, tracker):
        self.tracker = tracker
        self.window = None
        self.anchor_crop = None
        self.last_native_box = None
        self.tiles = None
        self.tile_previous = None

    def __getattr__(self, name):
        return getattr(self.tracker, name)

    def make_window(self, shape, box):
        h, w = shape[:2]
        side = max(self.cfg.get('crop_min_size', 256), max(box[2]-box[0], box[3]-box[1])*self.cfg.get('crop_scale', 5))
        cw, ch = min(w, round(side)), min(h, round(side))
        x = int(np.clip(round((box[0]+box[2]-cw)/2), 0, w-cw))
        y = int(np.clip(round((box[1]+box[3]-ch)/2), 0, h-ch))
        return [x, y, x+cw, y+ch]

    def step(self, frame, stamp, box=None):
        original = getattr(frame, 'original', None)
        original = frame if original is None else original
        sx, sy = original.shape[1]/frame.shape[1], original.shape[0]/frame.shape[0]
        factor = np.array([sx, sy, sx, sy])
        correction, recenter = '', False
        if box is not None:
            if self.cfg.get('tiled_search', False):
                from sam_tiles import TileSearch
                self.tiles = TileSearch(self.tracker)
            self.tile_previous = None
            native = np.asarray(box)*factor
            self.window = (self.make_window(original.shape, native)
                           if self.cfg.get('small_enabled', True) and max(native[2:]-native[:2]) < self.cfg.get('small_side', 64) else None)
            self.last_native_box = None
            self.anchor_crop = None
            self.tracker.small_mode = self.window is not None
        elif self.window is not None and self.last_native_box is not None and self.tracker.state == 'TRACKING':
            a,b,c,d = self.window
            x,y,u,v = self.last_native_box
            margin = min(c-a,d-b)*.15
            if min(x-a,y-b,c-u,d-v) < margin:
                new = self.make_window(original.shape, self.last_native_box)
                if new != self.window:
                    old_window = self.window
                    self.window, recenter = new, True
                    native = np.asarray(self.last_native_box)
                    correction = 'small_crop_recenter_anchor_retained'
        if self.window is None:
            return self.tracker.step(frame, stamp, box)
        a,b,c,d = self.window
        crop = original[b:d,a:c]
        local_box = list(native-np.array([a,b,a,b])) if box is not None or recenter else None
        if box is not None:
            self.anchor_crop = crop.copy()
        if recenter:
            result = self.tracker.recenter_crop(crop, stamp, local_box)
            if not result.get('valid'):
                self.window = old_window
                correction = 'small_crop_recenter_identity_rejected'
        else:
            result = self.tracker.step(crop, stamp, local_box)
        if self.tile_previous is not None:
            if self.tracker.state in ('LOST','SEARCHING'):
                self.window,self.last_native_box = self.tile_previous
                self.tracker.sam.reset_motion()
                self.tile_previous = None
            elif self.tracker.state == 'TRACKING':
                self.tile_previous = None
        if self.tiles is not None and self.tracker.state in ('LOST','SEARCHING'):
            seeded,window,info = self.tiles.step(original,stamp,self.window,self.last_native_box)
            result.update(info)
            if seeded is not None:
                self.tile_previous = (list(self.window),self.last_native_box)
                self.window,result = window,seeded
            result['pipeline_ms'] = result.get('pipeline_ms',0)+info.get('tile_search_ms',0)
        a,b,c,d = self.window
        mask = result.get('mask')
        if result.get('box') is not None:
            self.last_native_box = [float(v) for v in np.asarray(result['box'])+np.array([a,b,a,b])]
            result['native_box'] = self.last_native_box
            result['box'] = [float(v) for v in np.asarray(self.last_native_box)/factor]
            x,y,u,v = result['box']
            result.update(dx=(x+u-frame.shape[1])/2, dy=(y+v-frame.shape[0])/2)
        def to_display(binary):
            out = np.zeros(frame.shape[:2], bool)
            x,y,u,v = round(a/sx),round(b/sy),round(c/sx),round(d/sy)
            if u > x and v > y:
                out[y:v,x:u] = cv2.resize(binary.astype(np.float32),(u-x,v-y),interpolation=cv2.INTER_LINEAR) >= .5
            return out
        if mask is not None:
            result['mask'] = to_display(mask)
        if result.get('candidate_box') is not None:
            result['candidate_box'] = [float(v) for v in (np.asarray(result['candidate_box'])+np.array([a,b,a,b]))/factor]
        if 'search_masks' in result:
            result['search_masks'] = [to_display(item) for item in result['search_masks']]
        if self.tiles is not None and self.debug_search and self.tracker.state in ('LOST','SEARCHING'):
            for window,binary in self.tiles.last_debug:
                ta,tb,tc,td = window
                x,y,u,v = round(ta/sx),round(tb/sy),round(tc/sx),round(td/sy)
                candidate = np.zeros(frame.shape[:2],bool)
                candidate[y:v,x:u] = cv2.resize(binary.astype(np.float32),(u-x,v-y))>=.5
                result.setdefault('search_masks',[]).append(candidate)
        result.update(small=True, crop_box=self.window, correction_reason=correction,
                      source_width=original.shape[1], source_height=original.shape[0])
        if result['status'] == 'TRACKING':
            result['status'] = 'SMALL'
        result['transition'] = '->'.join('SMALL' if state == 'TRACKING' else state
                                         for state in result['transition'].split('->'))
        return result
