"""Camera UI: capture, inference and display have independent execution."""
import argparse
import json
import os
import threading
import time
from pathlib import Path
import cv2
from video_profile import sample
ROOT = Path(__file__).resolve().parents[1]


def create_tracker(args):
    """The source camera predictor is the only AI tracking path."""
    from source_camera_tracker import SourceCameraTracker
    if not getattr(args, 'sam_only', True):
        raise ValueError('The source pipeline uses SAM only; remove --no-sam-only.')
    tracker = SourceCameraTracker(args.checkpoint,
        image_size=getattr(args, 'image_size', 512),
        model_config=getattr(args, 'model_config', None),
        precision=getattr(args, 'sam_precision', 'bf16'))
    tracker.warmup()
    if getattr(args,'engine','baseline') == 'enhanced':
        from hybrid_tracker import HybridTracker
        tracker=HybridTracker(tracker,getattr(args,'small_target_config',None),getattr(args,'hybrid_config',None))
        tracker.hybrid.kalman=tracker.hybrid.kalman and not getattr(args,'no_kalman',False)
        tracker.hybrid.reid=tracker.hybrid.reid and not getattr(args,'no_reid',False)
        tracker.hybrid.motion_flow=tracker.hybrid.motion_flow and not getattr(args,'no_flow',False)
        return tracker,None
    if getattr(args,'small_target','auto') == 'auto':
        from small_target import SmallTargetTracker
        tracker=SmallTargetTracker(tracker,getattr(args,'small_target_config',None))
    return tracker, None


def camera_timing(displayed, packet, now):
    """Flow updates the image, not the acquisition time of SAM evidence."""
    captured=displayed.get('captured',packet[1])
    latency=displayed.get('sam_latency_ms',displayed.get('latency_ms',0.))
    return dict(captured=captured,latency_ms=latency,sam_latency_ms=latency,
                age_ms=max(0.,now-captured)*1000,
                display_latency_ms=max(0.,now-packet[1])*1000)


def write_telemetry(path, data):
    from tracking_control import safe_output
    data=safe_output(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    temp.write_text(json.dumps(data), encoding='utf-8')
    try:
        for attempt in range(5):
            try:
                temp.replace(path)
                break
            except PermissionError:
                if attempt==4:
                    raise
                time.sleep(.01*2**attempt)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


class TelemetryWriter:
    """Latest-only disk worker; a locked JSON file never stalls camera display."""
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.pending = None
        self.last_control = None
        self.last_control_at = 0.
        self.stop = threading.Event()
        self.last_warning = 0.
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, data):
        with self.lock:
            self.pending = data
            self.last_control = data if 'gimbal_command' in data else None
            self.last_control_at = time.perf_counter()

    def flush_once(self):
        with self.lock:
            data = self.pending
            if data is None and self.last_control is not None:
                elapsed=(time.perf_counter()-self.last_control_at)*1000
                initial_age=max(float(self.last_control.get('age_ms') or 0),float(self.last_control.get('source_age_ms') or 0))
                if elapsed+initial_age>250:
                    data=dict(self.last_control,source_age_ms=elapsed+initial_age)
                    self.last_control=None
        if data is None:
            return
        try:
            write_telemetry(self.path, data)
        except OSError as exc:
            # Append failed snapshots to a separate journal. JSON is a latest
            # state channel; the frame CSV remains the complete event record.
            try:
                with self.path.with_suffix('.retry.jsonl').open('a',encoding='utf-8') as handle:
                    handle.write(json.dumps(data)+'\n')
            except OSError:
                pass
            if time.monotonic() - self.last_warning > 5:
                print(f'Telemetry unavailable, retrying; tracking continues: {exc}', flush=True)
                self.last_warning = time.monotonic()
            return
        with self.lock:
            if self.pending is data:
                self.pending = None

    def run(self):
        while not self.stop.wait(.05):
            self.flush_once()
        self.flush_once()

    def close(self):
        self.submit({'valid': False, 'status': 'STOPPED', 'dx': None, 'dy': None,
                     'is_confirmed':False,'recommended_control_mode':'HOLD',
                     'gimbal_command':dict(mode='HOLD',rate_x=0.,rate_y=0.,simulated=True)})
        self.stop.set()
        self.thread.join(timeout=1)


class LatestCapture:
    def __init__(self, source, config=None, record_raw=None):
        self.source = source
        self.config = config or {}
        self.record_raw = record_raw
        self.measured_fps = 0.
        self.started_at=None
        self.frame_count=0
        self.lock = threading.Lock()
        self.packet = None
        self.stop = threading.Event()
        self.start_read = threading.Event()
        self.done = False
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.start_read.wait(.05):
            if self.stop.is_set():
                self.done = True
                return
        cap = recorder = None
        try:
            backends = {'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF}
            if isinstance(self.source, int):
                requested = self.config.get('backend', 'auto').lower()
                choices = ['dshow','msmf'] if requested == 'auto' else [requested]
                for backend in choices:
                    cap = cv2.VideoCapture(self.source, backends[backend])
                    if cap.isOpened():
                        print(f'Camera backend: {backend}', flush=True)
                        break
                    cap.release()
            else:
                cap = cv2.VideoCapture(self.source, cv2.CAP_ANY)
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open camera/video: {self.source}')
            buffer_ok = cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if isinstance(self.source, int):
                cfg = self.config
                if cfg.get('mjpg', True):
                    print(f'MJPG accepted: {cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))}', flush=True)
                for prop,key,default in ((cv2.CAP_PROP_FRAME_WIDTH,'width',640),
                                         (cv2.CAP_PROP_FRAME_HEIGHT,'height',480), (cv2.CAP_PROP_FPS,'fps',30)):
                    cap.set(prop, cfg.get(key,default))
                auto = cfg.get('auto_exposure')
                if auto is not None:
                    setting = cfg.get('auto_exposure_value',1) if auto else cfg.get('manual_exposure_value',0)
                    ok_auto = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,setting)
                    ok_exposure = True if auto else cap.set(cv2.CAP_PROP_EXPOSURE,cfg.get('exposure',-6))
                    print(f'Exposure request auto={auto}: accepted={ok_auto}/{ok_exposure}, '
                          f'readback auto={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)} exposure={cap.get(cv2.CAP_PROP_EXPOSURE)}', flush=True)
            print(f'Capture buffer=1 accepted={buffer_ok}, readback={cap.get(cv2.CAP_PROP_BUFFERSIZE)}; '
                  f'driver FPS={cap.get(cv2.CAP_PROP_FPS)} (measured FPS follows)', flush=True)
            fps = cap.get(cv2.CAP_PROP_FPS)
            period = 1 / fps if fps > 0 else 1 / 30
            if self.record_raw:
                from raw_recording import RawRecorder
                recorder = RawRecorder(self.record_raw, 1/period)
            seq = 0
            report_start, report_frames = time.perf_counter(), 0
            while not self.stop.is_set():
                tick = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                if seq == 0:
                    print(f'Actual camera/frame size: {frame.shape[1]}x{frame.shape[0]}; requested '
                          f'{self.config.get("width",640)}x{self.config.get("height",480)}',flush=True)
                packet = (seq, time.perf_counter(), frame)
                if self.started_at is None:
                    self.started_at=packet[1]
                self.frame_count=seq+1
                if recorder:
                    recorder.submit(packet)
                with self.lock:
                    self.packet = packet
                seq += 1
                report_frames += 1
                elapsed = time.perf_counter()-report_start
                if elapsed >= 2:
                    self.measured_fps = report_frames/elapsed
                    print(f'Capture FPS (last {elapsed:.1f}s): {self.measured_fps:.1f}', flush=True)
                    if isinstance(self.source,int) and self.measured_fps<25:
                        print('Camera below 25 FPS: check exposure/light/USB/backend; display cannot create additional source frames.',flush=True)
                    report_start, report_frames = time.perf_counter(), 0
                if isinstance(self.source, str):
                    self.stop.wait(max(0, period - (time.perf_counter() - tick)))
        except Exception as exc:
            self.error = str(exc)
        finally:
            if recorder:
                try:
                    recorder.close()
                except Exception as exc:
                    self.error = str(exc)
            if cap is not None:
                cap.release()
            self.done = True

    def get(self):
        with self.lock:
            return self.packet


class Worker:
    def __init__(self, capture, args):
        self.capture, self.args = capture, args
        self.lock = threading.Lock()
        self.command = self.result = None
        self.label_command = None
        self.feature_commands=set()
        self.generation = 0
        self.error = None
        self.ready = False
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def select(self, packet, box):
        with self.lock:
            self.generation += 1
            self.command = (self.generation, packet, box)
            self.result = None

    def label(self, correct):
        with self.lock:
            if self.result:
                self.label_command = (self.generation, self.result.get('attempt_id', 0), correct,
                                      self.result.get('source_index',self.result['seq']),self.result['captured'],self.result.get('confidence'))

    def toggle_feature(self,name):
        with self.lock:
            self.feature_commands.symmetric_difference_update((name,))

    def clear(self):
        with self.lock:
            self.generation += 1
            self.command = (self.generation, None, None)
            self.result = None

    def run(self):
        search = None
        diagnostic = None
        try:
            tracker, detector = create_tracker(self.args)
            import torch
            inference_stream=torch.cuda.Stream(priority=-1)
            from video_playback import FrameLog
            source=str(getattr(self.args,'source','0'))
            log_path=getattr(self.args,'csv',None) or ROOT/'runtime'/('camera_frames.csv' if source.isdecimal() else Path(source).stem+'_frames.csv')
            diagnostic = FrameLog(log_path.with_suffix('.sam.csv'))
            if detector is not None:
                from search_detector import SearchDetector
                search = SearchDetector(self.capture, detector)
            self.capture.start_read.set()
            self.ready = True
            active, last_seq, generation = False, -1, 0
            from collections import OrderedDict
            packet_ids=OrderedDict()
            while not self.capture.stop.is_set():
                with self.lock:
                    command, self.command = self.command, None
                    label, self.label_command = self.label_command, None
                    features,self.feature_commands=self.feature_commands,set()
                if hasattr(tracker,'toggle_feature'):
                    for feature in features:tracker.toggle_feature(feature)
                if label and label[0] == generation and hasattr(tracker, 'label'):
                    current = self.capture.get()
                    stamp = current[1] if getattr(self.capture, 'media_clock', False) and current else time.perf_counter()
                    applied=tracker.label(label[2], stamp, label[1])
                    from recovery.selection import record_feedback
                    record_feedback(log_path,label[3],label[4],label[1],label[2],applied,label[5])
                box = None
                detections = None
                if command:
                    generation, packet, box = command
                    if packet is None:
                        active, last_seq = False, -1
                        tracker.reset()
                        continue
                    active = True
                else:
                    packet = self.capture.get()
                if search:
                    if search.error:
                        raise RuntimeError(search.error)
                    searching = (active and box is None and tracker.state in ('LOST','SEARCHING')
                                 and getattr(tracker, 'target_mode', 'PERSON') == 'PERSON')
                    if searching:
                        search.enabled.set()
                        found = search.get()
                        if found is None:
                            self.capture.stop.wait(.002)
                            continue
                        packet, boxes, detector_ms = found
                        detections = (boxes, detector_ms)
                    else:
                        search.enabled.clear()
                if not active or packet is None or (box is None and packet[0] <= last_seq):
                    self.capture.stop.wait(.005)
                    continue
                seq, stamp, frame = packet
                packet_ids[stamp]=seq
                while len(packet_ids)>120:
                    packet_ids.popitem(last=False)
                queue_ms=(self.capture.age_ms(seq) if hasattr(self.capture,'age_ms') else (time.perf_counter()-stamp)*1000)
                work_start=time.perf_counter()
                with torch.cuda.stream(inference_stream):
                    result = tracker.step(frame, stamp, box, detections=detections) if search else tracker.step(frame, stamp, box)
                work_ms=(time.perf_counter()-work_start)*1000
                result.update(queue_ms=max(0.,queue_ms),worker_compute_ms=work_ms)
                if result.get('sam_core_wall_ms') is not None:
                    result['recovery_checks_ms']=max(0.,work_ms-result['sam_core_wall_ms'])
                result.update(seq=seq, captured=stamp, frame=frame,
                              generation=generation,
                              skipped=max(0, seq - last_seq - 1) if box is None else 0)
                if result.get('candidate_stamp') is not None:
                    result['candidate_seq']=packet_ids.get(result['candidate_stamp'],seq)
                result['latency_ms'] = (self.capture.age_ms(seq) if hasattr(self.capture, 'age_ms')
                                        else (time.perf_counter() - stamp) * 1000)
                result['sam_latency_ms']=result['latency_ms']
                result['completed'] = time.perf_counter()
                source_index=getattr(self.capture,'source_indices',{}).get(seq,seq)
                result['source_index']=source_index
                result['frame_id']=source_index
                diagnostic.write(0,(source_index,stamp,frame),result,result['pipeline_ms']/1000 if result.get('pipeline_ms') else result.get('inference_ms',0)/1000)
                last_seq = seq
                with self.lock:
                    if generation == self.generation:
                        self.result = result
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error = str(exc)
        finally:
            if diagnostic:
                diagnostic.handle.close()
            if 'tracker' in locals() and hasattr(tracker, 'close'):
                tracker.close()
            if search:
                search.enabled.clear()
                # Capture.stop is owned by main; error also terminates this child.
                if self.error:
                    self.capture.stop.set()
                search.thread.join(timeout=2)


def draw(frame, result):
    view = frame.copy()
    h, w = view.shape[:2]
    cv2.drawMarker(view, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 18)
    if result and result['box'] is not None:
        mask = result['mask']
        if mask is not None:
            tick = time.perf_counter()
            view[mask] = (view[mask] * .65 + (0, 89, 0)).astype('uint8')
            sample('mask_overlay', time.perf_counter()-tick)
        x1, y1, x2, y2 = result['box']
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(view, ((x1 + x2) // 2, (y1 + y2) // 2), 5, (0, 0, 255), -1)
    if result:
        from recovery.selection import draw_candidates
        draw_candidates(view,result)
        for key, color, label in (('predicted_box', (255, 180, 0), 'PREDICTED - NO CONTROL'),
                                  ('candidate_box', (0, 180, 255), 'CANDIDATE - VERIFYING')):
            if result.get(key) is not None:
                x1, y1, x2, y2 = map(round, result[key])
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 1)
                cv2.putText(view, label, (x1, max(15, y1 - 6)), 0, .4, color, 1)
    return view


def draw_preview(frame, result):
    """Composite at display size; inference masks/coordinates remain untouched."""
    h, w = frame.shape[:2]
    preview = cv2.resize(frame, (640, 480))
    scaled = None
    if result is not None:
        scaled = result.copy()
        if result.get('search_masks'):
            scaled['search_masks']=[cv2.resize(m.astype('uint8'),(640,480),interpolation=cv2.INTER_NEAREST)>0 for m in result['search_masks']]
        for key in ('predicted_box', 'candidate_box'):
            if result.get(key) is not None:
                scaled[key] = [round(v * (640 / w if i % 2 == 0 else 480 / h))
                               for i, v in enumerate(result[key])]
        if result['box'] is not None:
            scaled['box'] = [round(v * (640 / w if i % 2 == 0 else 480 / h))
                             for i, v in enumerate(result['box'])]
            scaled['mask'] = cv2.resize(result['mask'].astype('uint8'), (640, 480),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
    return draw(preview, scaled)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--small-target',choices=('auto','off'),default='auto',help='adaptive ROI or original full-frame baseline')
    p.add_argument('--small-target-config',type=Path,default=ROOT/'configs/small_target.json')
    p.add_argument('--debug-tracking',action='store_true',help='show state, ROI and actual target pixel dimensions')
    p.add_argument('--engine',choices=('baseline','enhanced'),default='baseline')
    p.add_argument('--hybrid-config',type=Path,default=ROOT/'configs/hybrid.json')
    p.add_argument('--no-kalman',action='store_true')
    p.add_argument('--no-flow',action='store_true')
    p.add_argument('--source', default='0', help='camera index or video path')
    p.add_argument('--cam-auto-exposure',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--cam-exposure',type=float,help='driver-specific manual exposure; implies manual mode')
    p.add_argument('--cam-size',help='camera size WIDTHxHEIGHT, e.g. 640x480')
    p.add_argument('--sync-probes',action='store_true',help='A/B diagnostic: run verification in tracking worker')
    p.add_argument('--sam-resize',choices=('linear','bicubic'),default='linear',help='controlled preprocessing A/B')
    p.add_argument('--sam-precision',choices=('fp16','bf16'),default='bf16',help='BF16 A/B uses eager execution, not FP16 CUDA graphs')
    p.add_argument('--sam-jpeg-quality',type=int,choices=range(0,101),default=0,metavar='0..100',help='0 disables experimental JPEG roundtrip')
    p.add_argument('--timing-details',action='store_true',help='CUDA event stage diagnostics; adds synchronization in inference worker')
    p.add_argument('--flow-max-side',type=int,default=0,help='optional dense-flow ROI resolution cap; 0 preserves original resolution')
    p.add_argument('--record-raw', type=Path, help='camera raw video + .timestamps.csv, asynchronous')
    p.add_argument('--mode', choices=('realtime', 'offline'), default='realtime')
    p.add_argument('--sam-only', action=argparse.BooleanOptionalAction, default=True,
                   help='source pipeline is SAM only; --no-sam-only is rejected')
    p.add_argument('--keep-running', action='store_true',
                   help='diagnostic: continue SAM without new prompts after loss (SAM-only)')
    p.add_argument('--lost-stride', type=int, default=1, help='SAM calls every k input frames after loss')
    p.add_argument('--sam-diagnostic', action='store_true', help='Step 1 diagnostics only; do not accept recovery')
    p.add_argument('--sam-memory-config', type=Path, help='SAM-only memory/search thresholds JSON')
    p.add_argument('--recovery', choices=('current','friend'), default='friend',
                   help='legacy compatibility option; both values now use source camera predictor')
    p.add_argument('--debug-search', action='store_true', help='show faint verified candidate masks')
    p.add_argument('--start-frame', type=int, default=0, help='zero-based video frame')
    p.add_argument('--max-frames', type=int, help='maximum processed video frames across loops/seeks')
    p.add_argument('--resize-width', type=int, help='video processing width; default: original')
    p.add_argument('--loop', action='store_true', help='restart video and select a new target at EOF')
    p.add_argument('--save-video', type=Path, help='output overlay video (.mp4 or .avi)')
    p.add_argument('--csv', type=Path, help='frame CSV; default: beside output video or runtime/<stem>_frames.csv')
    p.add_argument('--profile', type=Path, help='stage timings JSON; prints mean/p95 each second')
    p.add_argument('--audit-masks', type=Path, help='optional packed masks for regression comparison (.npz)')
    p.add_argument('--overlay-quality', choices=('full','fast'), default='full')
    p.add_argument('--realtime-config', type=Path, default=ROOT / 'configs/realtime.json')
    p.add_argument('--init-bbox', help='initial box x,y,w,h in processing-image pixels')
    p.add_argument('--init-point', help='SAM-only positive point x,y in processing-image pixels')
    p.add_argument('--init-frame', type=int, default=None, help='zero-based video frame to initialize')
    p.add_argument('--no-display', action='store_true', help='headless video; requires --init-bbox')
    p.add_argument('--object-config', type=Path, default=ROOT / 'configs/object.json')
    p.add_argument('--target-mode', choices=('auto', 'person', 'object'), default='auto',
                   help='compatibility option; source predictor tracks any selected target without class routing')
    p.add_argument('--model-config', help='Hydra config inside vendored source sam2; default Tiny 512 or 1024')
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'sam2/checkpoints/sam2.1_hiera_tiny.pt')
    p.add_argument('--memory', type=int, default=24)
    p.add_argument('--image-size', type=int, choices=(512, 1024), default=512,
                   help='SAM encoder size; 512 is faster, 1024 preserves original detail')
    p.add_argument('--min-iou', type=float, default=.5)
    p.add_argument('--max-gap', type=float, default=1.)
    p.add_argument('--no-cuda-graph', action='store_true', help='Use original encoder execution for diagnostics')
    p.add_argument('--telemetry', type=Path, default=ROOT / 'runtime/target.json')
    p.add_argument('--reid-config', type=Path, default=ROOT / 'configs/reid.json')
    p.add_argument('--no-reid', action='store_true', help='Original SAMURAI-only manual recovery')
    p.add_argument('--camera-config', type=Path, default=ROOT / 'configs/camera.json')
    args = p.parse_args()
    # Resource/output paths are relative to project, never terminal cwd.
    for key, value in vars(args).items():
        if isinstance(value, Path) and not value.is_absolute():
            setattr(args, key, ROOT / value)
    if not args.source.isdecimal() and not Path(args.source).is_absolute():
        args.source = str(ROOT / args.source)
    if args.lost_stride < 1:
        p.error('--lost-stride must be >= 1')
    if args.keep_running and not args.sam_only:
        p.error('--keep-running is a SAM-only diagnostic; remove --no-sam-only')
    if args.flow_max_side and args.flow_max_side<64:
        p.error('--flow-max-side must be 0 or >=64')
    if args.init_bbox:
        try:
            import math
            values = [float(v) for v in args.init_bbox.split(',')]
            if len(values) != 4 or not all(math.isfinite(v) for v in values) or min(values[:2]) < 0 or min(values[2:]) <= 0:
                raise ValueError()
            args.init_bbox = values
        except ValueError:
            p.error('--init-bbox requires finite x,y,w,h; x,y >= 0 and w,h > 0')
    if args.init_point:
        if args.init_bbox or not args.sam_only:
            p.error('--init-point requires SAM-only and cannot be combined with --init-bbox')
        try:
            import math
            xy = list(map(float, args.init_point.split(',')))
            if len(xy) != 2 or not all(math.isfinite(v) and v >= 0 for v in xy):
                raise ValueError()
            args.init_bbox = [*xy, 0., 0.]
        except ValueError:
            p.error('--init-point requires finite, nonnegative x,y')
    if args.no_display and (not args.init_bbox or args.source.isdecimal()):
        p.error('--no-display requires a video source and --init-bbox')
    if args.init_frame is not None and args.init_frame < args.start_frame:
        p.error('--init-frame must be >= --start-frame')
    if args.start_frame < 0 or any(v is not None and v <= 0 for v in (args.max_frames, args.resize_width)):
        p.error('start-frame must be >= 0; max-frames and resize-width must be > 0')
    if args.record_raw and (args.record_raw.exists() or args.record_raw.with_suffix('.timestamps.csv').exists()):
        p.error(f'Recording already exists: {args.record_raw}. Choose a new recording filename.')
    if not args.source.isdecimal():
        if args.record_raw:
            p.error('--record-raw is for live cameras only')
        if args.mode == 'realtime':
            from realtime_video import run_realtime
            run_realtime(args, TelemetryWriter)
            return
        from video_playback import run_video
        run_video(args, create_tracker, draw, TelemetryWriter)
        return
    print(f'Loading source SAM2 Tiny {args.image_size}x{args.image_size} on CUDA and warming up.', flush=True)
    print('Source SAM2 camera predictor; persistent online memory. S/R reselects; Q/Esc exits.', flush=True)
    camera_config = json.loads(args.camera_config.read_text(encoding='utf-8'))
    if args.cam_auto_exposure is not None:
        camera_config['auto_exposure']=args.cam_auto_exposure
    if args.cam_exposure is not None:
        camera_config.update(auto_exposure=False,exposure=args.cam_exposure)
    if args.cam_size:
        try:
            width,height=map(int,args.cam_size.lower().split('x'))
            if min(width,height)<=0:
                raise ValueError()
            camera_config.update(width=width,height=height)
        except ValueError:
            p.error('--cam-size must be WIDTHxHEIGHT with positive integers')
    import video_profile
    runtime_config=json.loads(args.realtime_config.read_text(encoding='utf-8'))
    cv2.setNumThreads(runtime_config.get('opencv_threads',4))
    camera_profile=video_profile.Profile(args.profile or (args.csv or ROOT/'runtime/camera_frames.csv').with_suffix('.profile.json'))
    video_profile.ACTIVE=camera_profile
    capture = LatestCapture(int(args.source) if args.source.isdecimal() else args.source, camera_config, args.record_raw)
    worker = Worker(capture, args)
    name = 'SOURCE SAM2 MASK'
    last_export = None
    from mask_display import MaskDisplay, overlay
    from realtime_video import OutputWorker
    transport = MaskDisplay(max_age=args.max_gap,flow_max_side=args.flow_max_side,enabled=not args.no_flow)
    camera_output = OutputWorker(args,args.csv or ROOT/'runtime/camera_frames.csv',camera_config.get('fps',30),TelemetryWriter,90,emit_telemetry=False)
    telemetry = camera_output.telemetry
    last_output_seq = -1
    displayed = None
    display_generation = -1
    candidate_snapshot = None
    cached_key, cached_tracked = None, None
    last_stats = 0.
    selection = {'enabled': True, 'start': None, 'end': None}
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

    def mouse(event, x, y, flags, userdata):
        if not selection['enabled'] or not worker.ready:
            return
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < 640 and 0 <= y < 480:
            selection['start'] = (x, y)
        if selection['start'] is not None:
            selection['end'] = (min(639, max(0, x)), min(479, max(0, y)))
        if event == cv2.EVENT_LBUTTONUP and selection['start'] is not None:
            current = capture.get()
            x0, y0 = selection['start']
            x1, y1 = selection['end']
            h, w = current[2].shape[:2]
            box = [min(x0, x1) * w / 640, min(y0, y1) * h / 480,
                   max(x0, x1) * w / 640, max(y0, y1) * h / 480]
            if (box[2] - box[0] >= 1 and box[3] - box[1] >= 1) or (box[0]==box[2] and box[1]==box[3]):
                from video_playback import record_selection
                record_selection(args.csv or ROOT / 'runtime/camera_frames.csv', current[0], box)
                worker.select(current, box)
                selection['enabled'] = False
            selection['start'] = selection['end'] = None

    cv2.setMouseCallback(name, mouse)
    try:
        while True:
            if capture.error or worker.error:
                raise RuntimeError(capture.error or worker.error)
            packet = capture.get()
            if packet is None:
                if capture.done:
                    break
                if cv2.waitKey(10) & 255 in (27, ord('q')):
                    break
                continue
            with worker.lock:
                result = worker.result
                generation = worker.generation
            if generation != display_generation:
                transport.reset();displayed=None;display_generation=generation
            if camera_output.error:
                raise RuntimeError(camera_output.error)
            rate_elapsed=max(.001,time.perf_counter()-(getattr(capture,'started_at',None) or time.perf_counter()))
            capture_rate=(getattr(capture,'frame_count',0)-1)/rate_elapsed if getattr(capture,'started_at',None) else capture.measured_fps
            live_rate=max(0,camera_profile.frames-1)/rate_elapsed if getattr(capture,'started_at',None) else camera_profile.live_fps
            if displayed is None or packet[0] != last_output_seq:
                displayed = transport.step(packet,result)
                candidate_snapshot=result if result and displayed.get('sam_seq')==result['seq'] else None
                displayed.update(camera_timing(displayed,packet,time.perf_counter()))
                displayed.update(generation=result.get('generation',0) if result else 0,seq=packet[0],
                                 frame=packet[2],
                                 inference_ms=(result or {}).get('inference_ms',0),skipped=(result or {}).get('skipped',0))
                view = overlay(packet[2],displayed,args.overlay_quality)
                if args.debug_tracking:
                    from small_target import debug_hud
                    debug_hud(view,displayed)
                displayed['live_fps']=live_rate
                cv2.putText(view,f'SAM 2.1 | FPS {live_rate:.1f}',
                            (10,25),0,.6,(0,220,255),2)
                if packet[0]!=last_output_seq:
                    camera_output.submit(0,packet[0],packet[1],view,displayed,displayed['latency_ms']/1000)
                    camera_profile.frame(displayed)
                last_output_seq=packet[0]
            result = displayed
            age = None if result is None else time.perf_counter() - result['captured']
            stale = age is not None and age > args.max_gap
            if result and time.perf_counter()-last_stats >= 2:
                print(f'Camera avg {capture_rate:.1f} FPS | Live avg {live_rate:.1f} FPS | SAM latency {result["latency_ms"]:.1f} ms | '
                      f'SAM age {age*1000:.1f} ms | image latency {result["display_latency_ms"]:.1f} ms', flush=True)
                last_stats = time.perf_counter()
            # MaskDisplay transports completed SAM masks onto the current image.
            live = cv2.resize(packet[2], (640, 480))
            cv2.putText(live, 'LIVE PREVIEW', (15, 30), 0, .7, (255, 255, 255), 2)
            render_key = (result['generation'], result['seq'], stale) if result else None
            if result:
                if render_key != cached_key:
                    # Composite each segmentation once, not on every GUI refresh.
                    cached_tracked = cv2.resize(overlay(result['frame'],result if not stale else dict(mask=None,box=None),hud_bottom=round(result['frame'].shape[0]*.5)),(640,480))
                    cached_key = render_key
                tracked = cached_tracked.copy()
            else:
                tracked = live.copy()
            lines = [f'SAM 2.1 | FPS {live_rate:.1f}']
            if args.debug_tracking and result:
                from small_target import debug_hud
                debug_result=result.copy()
                if result.get('crop_box'):
                    hh,ww=result['frame'].shape[:2]
                    debug_result['crop_box']=[v*(640/ww if i%2==0 else 480/hh) for i,v in enumerate(result['crop_box'])]
                debug_hud(tracked,debug_result)
            for i, line in enumerate(lines):
                cv2.putText(tracked, line, (15, 30 + 27 * i), 0, .6, (0, 220, 255), 2)
            import numpy as np
            if selection['enabled']:
                cv2.putText(live, 'Drag a box on LIVE preview', (15, 460), 0, .6, (0, 255, 255), 2)
            if selection['start'] and selection['end']:
                cv2.rectangle(live, selection['start'], selection['end'], (0, 255, 255), 2)
            cv2.imshow(name, np.hstack((live, tracked)))
            export_key = (result['generation'], result['seq'], stale) if result else None
            if result is None and last_export != 'no_target':
                telemetry.submit({'valid': False, 'status': 'SELECTING', 'dx': None, 'dy': None})
                last_export = 'no_target'
            if result and export_key != last_export:
                data = {k: v for k, v in result.items() if k not in ('mask', 'frame', 'search_masks','blend_previous','mask_roi','candidate_frame')}
                data['valid'] = result.get('valid', result['box'] is not None) and not stale
                data['timestamp_unix'] = time.time()
                data['source_age_ms'] = age * 1000
                data['age_minus_latency_ms'] = age*1000-result['latency_ms']
                data['camera_fps'] = capture.measured_fps
                if not data['valid']:
                    data.update(dx=None, dy=None, box=None, control_gain=0.)
                data['dx_normalized'] = None if data['dx'] is None else data['dx'] / (packet[2].shape[1] / 2)
                data['dy_normalized'] = None if data['dy'] is None else data['dy'] / (packet[2].shape[0] / 2)
                telemetry.submit(data)
                last_export = export_key
            key = cv2.waitKey(1) & 255
            if key in (27, ord('q')):
                break
            if key == ord('h'):
                args.debug_tracking = not args.debug_tracking
            if key in (ord('k'),ord('i'),ord('o')):
                worker.toggle_feature({ord('k'):'kalman',ord('i'):'reid',ord('o'):'motion_flow'}[key])
                if key==ord('o'):
                    transport.enabled=not transport.enabled;transport.reset()
            if key in (ord('s'), ord('r')) and worker.ready:
                selection['enabled'] = True
                selection['start'] = selection['end'] = None
            if key in (ord('w'), ord('c')):
                worker.label(key == ord('c'))
            if ord('1') <= key <= ord('9'):
                from recovery.selection import choose_worker
                if choose_worker(worker,key-ord('1'),args.csv or ROOT/'runtime/camera_frames.csv',candidate_snapshot):
                    transport.reset();displayed=None
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if capture.done:
                break
    finally:
        capture.stop.set()
        capture.thread.join(timeout=3)
        worker.thread.join(timeout=5)
        cv2.destroyAllWindows()
        camera_output.close()
        camera_profile.close()
        video_profile.ACTIVE=None


if __name__ == '__main__':
    main()
