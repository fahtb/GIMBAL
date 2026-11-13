"""Video realtime transport: latest-frame SAM, independent decode/display/output."""
from collections import OrderedDict
import json
from pathlib import Path
import queue
import threading
import time

import cv2

from video_profile import sample, Profile
from video_playback import VideoSource, FrameLog, record_selection, select_box
from video_audit import MaskAudit
from mask_display import MaskDisplay, overlay


class InferenceSlot:
    """Same capture interface consumed by camera.Worker/SearchDetector."""
    media_clock = True

    def __init__(self):
        self.start_read, self.stop = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.packet = None
        self.arrivals = OrderedDict()

    def publish(self, packet):
        with self.lock:
            self.packet = packet
            self.arrivals.setdefault(packet[0],time.perf_counter())
            while len(self.arrivals) > 180:
                self.arrivals.popitem(last=False)

    def get(self):
        with self.lock:
            return self.packet

    def age_ms(self,seq):
        with self.lock:
            return (time.perf_counter()-self.arrivals.get(seq,time.perf_counter()))*1000


class VideoDecoder:
    def __init__(self,args,slot):
        self.args,self.slot = args,slot
        self.lock = threading.Lock()
        self.packet = None
        self.paused = True
        self.done = False
        self.error = None
        self.fps = 30.
        self.end_index = None
        self.request = None
        self.epoch = 0
        self.anchor = None
        self.source_indices = OrderedDict()
        slot.source_indices=self.source_indices
        self.thread = threading.Thread(target=self.run,daemon=True,name='video-decode')
        self.thread.start()

    def get(self):
        with self.lock:
            return self.packet

    def pause(self,value):
        with self.lock:
            self.paused = value
            if not value and self.packet:
                self.anchor = time.perf_counter()-self.packet[1]

    def seek(self,index):
        with self.lock:
            self.epoch += 1
            self.request = index
            self.paused = True
            self.done = False

    def run(self):
        source = None
        try:
            while not self.slot.start_read.wait(.01):
                if self.slot.stop.is_set():
                    return
            start = self.args.start_frame if self.args.init_frame is None else self.args.init_frame
            source = VideoSource(self.args.source,start,self.args.resize_width,cv2.INTER_AREA,
                                 preserve_original=getattr(self.args, 'sam_only', True))
            self.fps = source.fps
            seq, pending = 0, None
            while not self.slot.stop.is_set():
                with self.lock:
                    request,self.request = self.request,None
                    epoch,paused,anchor,empty = self.epoch,self.paused,self.anchor,self.packet is None
                if request is not None:
                    source.seek(request)
                    pending = None
                    empty = True
                if paused and not empty:
                    self.slot.stop.wait(.002)
                    continue
                if not paused and anchor is not None:
                    due = source.index_at_time(time.perf_counter()-anchor)
                    if source.index+1 > due:
                        self.slot.stop.wait(.001)
                        continue
                    # grab advances the codec without BGR conversion/resize.
                    while pending is None and source.index+1 < due:
                        tick = time.perf_counter()
                        ok = source.cap.grab()
                        sample('grab_skip',time.perf_counter()-tick)
                        if not ok:
                            self.end_index = source.index
                            self.done = True
                            break
                        source.index += 1
                if self.done:
                    self.slot.stop.wait(.005)
                    continue
                raw = pending or source.read()
                pending = None
                if raw is None:
                    self.end_index = source.index
                    self.done = True
                    continue
                with self.lock:
                    if epoch != self.epoch:
                        continue
                    if self.paused and not empty:
                        pending = raw
                        continue
                    self.packet = (seq, raw[1],raw[2])
                    self.source_indices[seq]=raw[0]
                    while len(self.source_indices)>240:
                        self.source_indices.popitem(last=False)
                    seq += 1
        except Exception as exc:
            self.error = str(exc)
        finally:
            if source:
                source.close()


class OutputWorker:
    def __init__(self,args,path,fps,telemetry_class,capacity,emit_telemetry=True):
        self.args,self.path,self.fps = args,path,fps
        self.queue = queue.Queue(capacity)
        self.error = None
        self.telemetry = telemetry_class(args.telemetry)
        self.emit_telemetry=emit_telemetry
        self.log = None
        self.audit = MaskAudit(args.audit_masks) if getattr(args,'audit_masks',None) else None
        self.thread = threading.Thread(target=self.run,daemon=True,name='video-output')
        self.thread.start()

    def submit(self,cycle,index,stamp,view,result,elapsed):
        # All array objects submitted here are immutable snapshots, not reused.
        try:
            self.queue.put_nowait((cycle,index,stamp,view,result.copy(),elapsed))
        except queue.Full:
            raise RuntimeError('Video/CSV writer cannot sustain playback; output queue is full (no silent frame loss)')

    def run(self):
        writer = None
        previous = None
        try:
            self.log = FrameLog(self.path)
            while True:
                job = self.queue.get()
                if job is None:
                    break
                if job[0] == 'tail':
                    end_index = job[1]
                    if previous:
                        for skipped in range(previous[1]+1,end_index+1):
                            self.log.write(previous[0],(skipped,previous[3]+(skipped-previous[1])/self.fps,previous[2]),{'status':'SKIPPED'},0.)
                            if writer:
                                writer.write(previous[2])
                    continue
                cycle,index,stamp,view,result,elapsed = job
                if self.args.save_video and writer is None:
                    self.args.save_video.parent.mkdir(parents=True,exist_ok=True)
                    codec = 'MJPG' if self.args.save_video.suffix.lower()=='.avi' else 'mp4v'
                    writer = cv2.VideoWriter(str(self.args.save_video),cv2.VideoWriter_fourcc(*codec),
                                             self.fps,(view.shape[1],view.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f'Cannot write {self.args.save_video}')
                if previous and previous[0] == cycle and previous[1] < index:
                    for skipped in range(previous[1]+1,index):
                        self.log.write(cycle,(skipped,previous[3]+(stamp-previous[3])*(skipped-previous[1])/(index-previous[1]),view),{'status':'SKIPPED'},0.)
                        if writer:
                            tick = time.perf_counter()
                            writer.write(previous[2])
                            sample('video_write',time.perf_counter()-tick)
                self.log.write(cycle,(index,stamp,view),result,elapsed)
                if writer:
                    tick = time.perf_counter()
                    writer.write(view)
                    sample('video_write',time.perf_counter()-tick)
                if self.audit:
                    self.audit.add(index,view,result)
                data = {k:v for k,v in result.items() if k not in ('mask','mask_roi','blend_previous','frame','search_masks','candidate_frame')}
                data.update(frame=index,source_time=stamp,timestamp_unix=time.time(),
                            latency_ms=elapsed*1000,source_age_ms=result.get('age_ms',0.))
                tick = time.perf_counter()
                if self.emit_telemetry:
                    self.telemetry.submit(data)
                sample('telemetry_submit',time.perf_counter()-tick)
                previous = cycle,index,view,stamp
        except Exception as exc:
            self.error = str(exc)
        finally:
            if writer:
                writer.release()
            if self.log:
                self.log.summary()
                self.log.handle.close()
            if self.audit:
                self.audit.close()
            self.telemetry.close()

    def close(self):
        while self.thread.is_alive():
            try:
                self.queue.put(None,timeout=.1)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('Output worker did not drain in 30 seconds')
        if self.error:
            raise RuntimeError(self.error)


def run_realtime(args,telemetry_class):
    from camera import Worker
    import video_profile
    cfg = json.loads(args.realtime_config.read_text(encoding='utf-8'))
    path = args.csv or (args.save_video.with_suffix('.csv') if args.save_video else
                       Path(__file__).resolve().parents[1]/'runtime'/f'{Path(args.source).stem}_frames.csv')
    paths = [Path(args.source).resolve(),path.resolve()]
    if args.save_video:
        paths.append(args.save_video.resolve())
    if len(set(paths)) != len(paths):
        raise ValueError('Input/output/CSV paths must differ')
    profile = Profile(args.profile or path.with_suffix('.profile.json'))
    video_profile.ACTIVE = profile
    slot = InferenceSlot()
    decoder = VideoDecoder(args,slot)
    worker = Worker(slot,args)
    output = None
    display = MaskDisplay(cfg['history_frames'],cfg['max_mask_age'],getattr(args,'flow_max_side',0),enabled=not getattr(args,'no_flow',False))
    name = 'SOURCE SAM2 VIDEO | P pause | A/D step | S/R select | W/C label | Q/Esc quit'
    paused,armed,selected = True,bool(args.init_bbox),False
    packet = None
    last_seq = last_render_sam = -1
    count,cycle = 0,0
    next_display = 0.
    request_step = False
    selections = {}
    candidate_snapshot=None
    try:
        if not args.no_display:
            cv2.namedWindow(name,cv2.WINDOW_NORMAL)
        while True:
            if worker.error or decoder.error or (output and output.error):
                raise RuntimeError(worker.error or decoder.error or output.error)
            fresh = decoder.get()
            step_completed = False
            now = time.perf_counter()
            with worker.lock:
                sam = worker.result
            if fresh is not None and (packet is None or not paused or request_step):
                if fresh[0] != last_seq:
                    packet = fresh
                    step_completed = request_step
                    request_step = False
            if packet is not None and worker.ready:
                if output is None:
                    cv2.setNumThreads(cfg.get('opencv_threads',4))
                    with profile.lock:
                        profile.stages.clear()
                        profile.window.clear()
                    output = OutputWorker(args,path,decoder.fps,telemetry_class,cfg['writer_queue'])
                index = decoder.source_indices.get(packet[0],round(packet[1]*decoder.fps))
                if args.init_bbox and not selected:
                    x,y,w,h = args.init_bbox
                    if x+w > packet[2].shape[1] or y+h > packet[2].shape[0]:
                        raise ValueError('--init-bbox exceeds processing image')
                    box = [x,y,x+w,y+h]
                    record_selection(path,index,box)
                    selections[packet[0]] = [x,y,w,h]
                    slot.publish(packet)
                    worker.select(packet,box)
                    selected = True
                if armed and sam is not None:
                    paused,armed = False,False
                    decoder.pause(False)
                sam_seq = sam['seq'] if sam is not None else -1
                changed = packet[0] != last_seq or ((paused or last_render_sam == -1) and sam_seq != last_render_sam)
                if changed and (paused or now >= next_display):
                    tick = time.perf_counter()
                    slot.publish(packet)
                    result = display.step(packet,sam)
                    candidate_snapshot=sam if sam and result.get('sam_seq')==sam['seq'] else None
                    if packet[0] in selections and sam is not None:
                        result['selected_bbox'] = selections.pop(packet[0])
                    result['live_fps'] = profile.live_fps
                    from sam_only import sam_fps
                    result['sam_fps'] = sam_fps(result) if sam else 0.
                    view = overlay(packet[2],result,args.overlay_quality)
                    if getattr(args,'debug_tracking',False):
                        from small_target import debug_hud
                        debug_hud(view,result)
                    text = [f'SAM 2.1 | FPS {profile.live_fps:.1f}']
                    for i,line in enumerate(text):
                        cv2.putText(view,line,(10,22+23*i),0,.5,(0,230,255),1,cv2.LINE_AA)
                    if not args.no_display:
                        show = time.perf_counter()
                        cv2.imshow(name,view)
                        sample('imshow',time.perf_counter()-show)
                    elapsed = time.perf_counter()-tick
                    sample('display_total',elapsed)
                    # Each distinct source frame is counted once; initial SAM
                    # completion on the frozen selection frame replaces preview.
                    if not paused or step_completed or sam is not None:
                        output.submit(cycle,index,packet[1],view,result,elapsed)
                        profile.frame(result)
                        count += 1
                    last_seq,last_render_sam = packet[0],sam_seq
                    # Preserve the display clock across small scheduling jitter;
                    # adding a full period to 'now' repeatedly skips source frames.
                    period=1/min(cfg['display_fps'],decoder.fps)
                    next_display=(next_display if next_display else now)+period
                    if next_display < now-period:
                        next_display=now+period
            key = -1 if args.no_display else cv2.waitKey(1)&255
            if key == ord('h'):
                args.debug_tracking = not getattr(args,'debug_tracking',False)
            if key in (ord('k'),ord('i'),ord('o')):
                worker.toggle_feature({ord('k'):'kalman',ord('i'):'reid',ord('o'):'motion_flow'}[key])
                if key==ord('o'):
                    display.enabled=not display.enabled;display.reset()
            if key in (27,ord('q')) or (not args.no_display and cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1):
                break
            if args.max_frames is not None and count >= args.max_frames:
                break
            if key == ord('p') and packet is not None:
                armed = False
                paused = not paused
                decoder.pause(paused)
                next_display = 0.
            if key in (ord('w'),ord('c')):
                worker.label(key==ord('c'))
            if ord('1') <= key <= ord('9'):
                from recovery.selection import choose_worker
                if choose_worker(worker,key-ord('1'),path,candidate_snapshot):
                    display.reset()
            if key in (ord('s'),ord('r')) and packet is not None:
                was_paused = paused
                decoder.pause(True)
                box = select_box(packet[2], allow_point=getattr(args,'sam_only',True))
                if box is not None:
                    record_selection(path,index,box)
                    selections[packet[0]] = [box[0],box[1],box[2]-box[0],box[3]-box[1]]
                    display.reset()
                    slot.publish(packet)
                    worker.select(packet,box)
                    selected = True
                    last_seq = last_render_sam = -1
                decoder.pause(was_paused)
            if paused and key in (ord('a'),ord('d')) and packet is not None:
                target = max(args.start_frame,index+(-1 if key==ord('a') else 1))
                if target != index:
                    if key==ord('a'):
                        worker.clear()
                        display.reset()
                        selected = False
                        cycle += 1
                        args.init_bbox = None
                    decoder.seek(target)
                    request_step = True
            if decoder.done and fresh is not None and fresh[0] == last_seq:
                if output and decoder.end_index is not None:
                    output.queue.put_nowait(('tail',decoder.end_index))
                if not args.loop:
                    break
                worker.clear()
                display.reset()
                selected,paused = False,True
                cycle += 1
                decoder.seek(args.start_frame if args.init_frame is None else args.init_frame)
                request_step = True
                armed = bool(args.init_bbox) and args.no_display
                if not args.no_display:
                    args.init_bbox = None
            time.sleep(.001)
    finally:
        slot.stop.set()
        decoder.thread.join(timeout=5)
        worker.thread.join(timeout=10)
        if output:
            output.close()
        profile.close()
        video_profile.ACTIVE = None
        if not args.no_display:
            cv2.destroyAllWindows()
