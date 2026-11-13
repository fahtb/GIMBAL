"""Camera UI: capture, inference and display have independent execution."""
import argparse
import json
import os
import threading
import time
from pathlib import Path
import cv2
from live_tracker import LiveTracker, ROOT


def create_tracker(args):
    """One pipeline for camera and file playback, including flow/ReID recovery."""
    tracker = LiveTracker(args.checkpoint, args.memory, args.min_iou, args.max_gap,
                          image_size=getattr(args, 'image_size', 512),
                          cuda_graph=not getattr(args, 'no_cuda_graph', False))
    tracker.warmup()
    detector = None
    if not getattr(args, 'no_reid', False):
        from person_reid import ReIDConfig, ReIDTracker, YOLOPersonDetector
        config = ReIDConfig.load(getattr(args, 'reid_config', ROOT / 'configs/reid.json'))
        detector = YOLOPersonDetector(config)
        tracker = ReIDTracker(tracker, detector, config)
        print('Person ReID enabled: HSV gallery + YOLOv8 + SAMURAI verification.', flush=True)
    return tracker, detector


def write_telemetry(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    temp.write_text(json.dumps(data), encoding='utf-8')
    try:
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


class TelemetryWriter:
    """Latest-only disk worker; a locked JSON file never stalls camera display."""
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.pending = None
        self.stop = threading.Event()
        self.last_warning = 0.
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, data):
        with self.lock:
            self.pending = data

    def flush_once(self):
        with self.lock:
            data = self.pending
        if data is None:
            return
        try:
            write_telemetry(self.path, data)
        except OSError as exc:
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
        self.submit({'valid': False, 'status': 'STOPPED', 'dx': None, 'dy': None})
        self.stop.set()
        self.thread.join(timeout=1)


class LatestCapture:
    def __init__(self, source, config=None):
        self.source = source
        self.config = config or {}
        self.measured_fps = 0.
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
        cap = None
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
            seq = 0
            report_start, report_frames = time.perf_counter(), 0
            while not self.stop.is_set():
                tick = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                with self.lock:
                    self.packet = (seq, time.perf_counter(), frame)
                seq += 1
                report_frames += 1
                elapsed = time.perf_counter()-report_start
                if elapsed >= 2:
                    self.measured_fps = report_frames/elapsed
                    print(f'Capture measured FPS: {self.measured_fps:.1f}', flush=True)
                    report_start, report_frames = time.perf_counter(), 0
                if isinstance(self.source, str):
                    self.stop.wait(max(0, period - (time.perf_counter() - tick)))
        except Exception as exc:
            self.error = str(exc)
        finally:
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
                self.label_command = (self.generation, self.result.get('attempt_id', 0), correct)

    def run(self):
        search = None
        try:
            tracker, detector = create_tracker(self.args)
            if detector is not None:
                from search_detector import SearchDetector
                search = SearchDetector(self.capture, detector)
            self.capture.start_read.set()
            self.ready = True
            active, last_seq, generation = False, -1, 0
            while not self.capture.stop.is_set():
                with self.lock:
                    command, self.command = self.command, None
                    label, self.label_command = self.label_command, None
                if label and label[0] == generation and hasattr(tracker, 'label'):
                    tracker.label(label[2], time.perf_counter(), label[1])
                box = None
                detections = None
                if command:
                    generation, packet, box = command
                    active = True
                else:
                    packet = self.capture.get()
                if search:
                    if search.error:
                        raise RuntimeError(search.error)
                    searching = active and box is None and tracker.state in ('LOST','SEARCHING')
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
                result = tracker.step(frame, stamp, box, detections=detections) if search else tracker.step(frame, stamp, box)
                result.update(seq=seq, captured=stamp, frame=frame,
                              generation=generation,
                              skipped=max(0, seq - last_seq - 1) if box is None else 0)
                result['latency_ms'] = (time.perf_counter() - stamp) * 1000
                result['completed'] = time.perf_counter()
                last_seq = seq
                with self.lock:
                    if generation == self.generation:
                        self.result = result
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error = str(exc)
        finally:
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
            view[mask] = (view[mask] * .65 + (0, 89, 0)).astype('uint8')
        x1, y1, x2, y2 = result['box']
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(view, ((x1 + x2) // 2, (y1 + y2) // 2), 5, (0, 0, 255), -1)
    if result:
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
    p.add_argument('--source', default='0', help='camera index or video path')
    p.add_argument('--mode', choices=('realtime', 'offline'), default='realtime')
    p.add_argument('--start-frame', type=int, default=0, help='zero-based video frame')
    p.add_argument('--max-frames', type=int, help='maximum processed video frames across loops/seeks')
    p.add_argument('--resize-width', type=int, help='video processing width; default: original')
    p.add_argument('--loop', action='store_true', help='restart video and select a new target at EOF')
    p.add_argument('--save-video', type=Path, help='output overlay video (.mp4 or .avi)')
    p.add_argument('--csv', type=Path, help='frame CSV; default: beside output video or runtime/<stem>_frames.csv')
    p.add_argument('--init-bbox', help='initial box x,y,w,h in processing-image pixels')
    p.add_argument('--init-frame', type=int, default=None, help='zero-based video frame to initialize')
    p.add_argument('--no-display', action='store_true', help='headless video; requires --init-bbox')
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'sam2/checkpoints/sam2.1_hiera_tiny.pt')
    p.add_argument('--memory', type=int, default=24)
    p.add_argument('--image-size', type=int, choices=(512, 768, 1024), default=512,
                   help='SAM encoder size; 512 is faster, 1024 preserves original detail')
    p.add_argument('--min-iou', type=float, default=.5)
    p.add_argument('--max-gap', type=float, default=1.)
    p.add_argument('--no-cuda-graph', action='store_true', help='Use original encoder execution for diagnostics')
    p.add_argument('--telemetry', type=Path, default=ROOT / 'runtime/target.json')
    p.add_argument('--reid-config', type=Path, default=ROOT / 'configs/reid.json')
    p.add_argument('--no-reid', action='store_true', help='Original SAMURAI-only manual recovery')
    p.add_argument('--camera-config', type=Path, default=ROOT / 'configs/camera.json')
    args = p.parse_args()
    if args.init_bbox:
        try:
            import math
            values = [float(v) for v in args.init_bbox.split(',')]
            if len(values) != 4 or not all(math.isfinite(v) for v in values) or min(values[:2]) < 0 or min(values[2:]) <= 0:
                raise ValueError()
            args.init_bbox = values
        except ValueError:
            p.error('--init-bbox requires finite x,y,w,h; x,y >= 0 and w,h > 0')
    if args.no_display and (not args.init_bbox or args.source.isdecimal()):
        p.error('--no-display requires a video source and --init-bbox')
    if args.init_frame is not None and args.init_frame < args.start_frame:
        p.error('--init-frame must be >= --start-frame')
    if args.start_frame < 0 or any(v is not None and v <= 0 for v in (args.max_frames, args.resize_width)):
        p.error('start-frame must be >= 0; max-frames and resize-width must be > 0')
    if not args.source.isdecimal():
        from video_playback import run_video
        run_video(args, create_tracker, draw, TelemetryWriter)
        return
    print(f'Loading SAMURAI Tiny {args.image_size}x{args.image_size} on CUDA and warming up.', flush=True)
    print('SAMURAI mask tracking + automatic person ReID. S/R reselects; Q/Esc exits.', flush=True)
    camera_config = json.loads(args.camera_config.read_text(encoding='utf-8'))
    capture = LatestCapture(int(args.source) if args.source.isdecimal() else args.source, camera_config)
    worker = Worker(capture, args)
    name = 'SAMURAI MASK | S/R select | W wrong / C correct | Q/Esc quit'
    last_export = None
    telemetry = TelemetryWriter(args.telemetry)
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
            if box[2] - box[0] > 3 and box[3] - box[1] > 3:
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
            age = None if result is None else time.perf_counter() - result['captured']
            stale = age is not None and age > args.max_gap
            if result and time.perf_counter()-last_stats >= 2:
                print(f'Camera {capture.measured_fps:.1f} FPS | latency {result["latency_ms"]:.1f} ms | '
                      f'age {age*1000:.1f} ms | age-latency {age*1000-result["latency_ms"]:.1f} ms', flush=True)
                last_stats = time.perf_counter()
            # Masks are rendered on their exact inference frame, alongside live
            # preview. Never overlay old coordinates onto a newer camera frame.
            live = cv2.resize(packet[2], (640, 480))
            cv2.putText(live, 'LIVE PREVIEW', (15, 30), 0, .7, (255, 255, 255), 2)
            render_key = (result['generation'], result['seq'], stale) if result else None
            if result:
                if render_key != cached_key:
                    # Composite each segmentation once, not on every GUI refresh.
                    cached_tracked = draw_preview(result['frame'], result if not stale else None)
                    cached_key = render_key
                tracked = cached_tracked.copy()
            else:
                tracked = live.copy()
            state = ('STALE / NO CONTROL' if stale else result['status']) if result else ('S/R: select target' if worker.ready else 'Loading SAM 2.1 Tiny...')
            if state == 'WAIT_SELECTION':
                state = 'WAIT_SELECTION - S/R to select'
            if result and result.get('provisional'):
                state = 'TRACKING provisional - low gain'
            lines = [f'{state} | SAM {args.image_size}']
            if result:
                ms = result['inference_ms']
                lines += [f'SAM compute {1000 / ms if ms else 0:.1f} FPS | {ms:.0f} ms',
                          f'Latency {result["latency_ms"]:.0f} ms | age {age * 1000:.0f} ms',
                          f'dx={None if stale else result["dx"]} dy={None if stale else result["dy"]} | skipped={result["skipped"]}']
                if 'reid_top1' in result:
                    lines += [f'ReID {result["reid_top1"]:.3f} / {result["reid_top2"]:.3f} | streak {result["reid_hits"]}',
                              f'Reacq {result["reacquired"]} | C correct {result.get("confirmed_correct",0)} | W wrong {result["confirmed_wrong"]}',
                              f'Gallery {result["gallery_size"]}/20 | pipeline {result["pipeline_ms"]:.0f} ms']
                    lines += [f'{result.get("reid_reason", "")} | {result.get("reid_region", "")}',
                              f'Camera {capture.measured_fps:.1f} FPS | age-latency {age*1000-result["latency_ms"]:.0f} ms']
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
                data = {k: v for k, v in result.items() if k not in ('mask', 'frame')}
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
            key = cv2.waitKey(10) & 255
            if key in (27, ord('q')):
                break
            if key in (ord('s'), ord('r')) and worker.ready:
                selection['enabled'] = True
                selection['start'] = selection['end'] = None
            if key in (ord('w'), ord('c')):
                worker.label(key == ord('c'))
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if capture.done:
                break
    finally:
        capture.stop.set()
        capture.thread.join(timeout=3)
        worker.thread.join(timeout=5)
        cv2.destroyAllWindows()
        telemetry.close()


if __name__ == '__main__':
    main()
