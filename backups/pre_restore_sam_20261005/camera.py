"""Camera UI: capture, inference and display have independent execution."""
import argparse
import json
import os
import threading
import time
from pathlib import Path
import cv2
from live_tracker import LiveTracker, ROOT
from live_motion import MotionWorker


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
    def __init__(self, source):
        self.source = source
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
        cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW if isinstance(self.source, int) else cv2.CAP_ANY)
        try:
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open camera/video: {self.source}')
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            fps = cap.get(cv2.CAP_PROP_FPS)
            period = 1 / fps if fps > 0 else 1 / 30
            seq = 0
            while not self.stop.is_set():
                tick = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                with self.lock:
                    self.packet = (seq, time.perf_counter(), frame)
                seq += 1
                if isinstance(self.source, str):
                    self.stop.wait(max(0, period - (time.perf_counter() - tick)))
        except Exception as exc:
            self.error = str(exc)
        finally:
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

    def run(self):
        try:
            tracker = LiveTracker(self.args.checkpoint, self.args.memory, self.args.min_iou, self.args.max_gap,
                                  image_size=getattr(self.args, 'image_size', 512))
            tracker.warmup()
            self.capture.start_read.set()
            self.ready = True
            active, last_seq, generation = False, -1, 0
            while not self.capture.stop.is_set():
                with self.lock:
                    command, self.command = self.command, None
                box = None
                if command:
                    generation, packet, box = command
                    active = True
                else:
                    packet = self.capture.get()
                if not active or packet is None or (box is None and packet[0] == last_seq):
                    self.capture.stop.wait(.005)
                    continue
                seq, stamp, frame = packet
                result = tracker.step(frame, stamp, box)
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


def draw(frame, result):
    view = frame.copy()
    h, w = view.shape[:2]
    cv2.drawMarker(view, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 18)
    if result and result['box'] is not None:
        mask = result['mask']
        if mask is not None:
            view[mask] = (view[mask] * .65 + (0, 89, 0)).astype('uint8')
        x1, y1, x2, y2 = map(round, result['box'])
        color = (0, 200, 255) if result.get('estimated') else (0, 255, 0)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        cv2.circle(view, ((x1 + x2) // 2, (y1 + y2) // 2), 5, (0, 0, 255), -1)
    return view


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='0', help='camera index or video path')
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'sam2/checkpoints/sam2.1_hiera_tiny.pt')
    p.add_argument('--memory', type=int, default=24)
    p.add_argument('--image-size', type=int, choices=(512, 768, 1024), default=512,
                   help='SAM encoder size; 512 is faster, 1024 preserves original detail')
    p.add_argument('--min-iou', type=float, default=.5)
    p.add_argument('--max-gap', type=float, default=1.)
    p.add_argument('--view', choices=('live', 'sam'), default='live',
                   help='live: low-latency motion box; sam: delayed diagnostic segmentation')
    p.add_argument('--max-sam-age', type=float, default=1.2,
                   help='maximum age in seconds of SAMURAI confirmation before motion tracking stops')
    p.add_argument('--telemetry', type=Path, default=ROOT / 'runtime/target.json')
    args = p.parse_args()
    print(f'Loading SAMURAI Tiny {args.image_size}x{args.image_size} on CUDA and warming up.', flush=True)
    print(f'View: {args.view}; live box uses optical flow, SAM masks are asynchronous. Telemetry: {args.telemetry}', flush=True)
    capture = LatestCapture(int(args.source) if args.source.isdecimal() else args.source)
    worker = Worker(capture, args)
    motion_worker = MotionWorker(capture, worker, args.max_sam_age)
    name = 'SAMURAI + live motion | S/R select | M view | Q/Esc quit'
    last_export = None
    telemetry = TelemetryWriter(args.telemetry)
    cached_key, cached_tracked = None, None
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
                worker.select(current, box)
                selection['enabled'] = False
            selection['start'] = selection['end'] = None

    cv2.setMouseCallback(name, mouse)
    try:
        while True:
            if capture.error or worker.error or motion_worker.error:
                raise RuntimeError(capture.error or worker.error or motion_worker.error)
            packet = capture.get()
            if packet is None:
                if capture.done:
                    break
                if cv2.waitKey(10) & 255 in (27, ord('q')):
                    break
                continue
            with worker.lock:
                sam_result, generation = worker.result, worker.generation
            with motion_worker.lock:
                motion_result = motion_worker.result
            result = motion_result if args.view == 'live' else sam_result
            if result and result.get('generation') != generation:
                result = None
            age = None if result is None else time.perf_counter() - result['captured']
            stale = age is not None and age > (.2 if args.view == 'live' else args.max_gap)
            if args.view == 'live' and result and result['sam_age_ms'] is not None:
                stale |= result['sam_age_ms'] + age * 1000 > args.max_sam_age * 1000
            # Masks are rendered on their exact inference frame, alongside live
            # preview. Never overlay old coordinates onto a newer camera frame.
            live = cv2.resize(packet[2], (640, 480))
            cv2.putText(live, 'LIVE PREVIEW', (15, 30), 0, .7, (255, 255, 255), 2)
            render_key = (id(result), stale)
            if result:
                if render_key != cached_key:
                    # Composite each segmentation once, not on every GUI refresh.
                    cached_tracked = cv2.resize(draw(result['frame'], result if not stale else None), (640, 480))
                    cached_key = render_key
                tracked = cached_tracked.copy()
            else:
                tracked = live.copy()
            state = ('STALE / NO CONTROL' if stale else result['status']) if result else ('S/R: select target' if worker.ready else 'Loading SAM 2.1 Tiny...')
            lines = [state]
            if result:
                if args.view == 'live':
                    lines += [f'Live updates {result["tracking_fps"]:.1f} FPS | flow {result["flow_ms"]:.1f} ms',
                              f'Latency {result["latency_ms"]:.0f} ms | image age {age * 1000:.0f} ms',
                              f'SAM check age {(result["sam_age_ms"] or 0) + age * 1000:.0f} ms | inference {result["sam_inference_ms"]:.0f} ms',
                              f'dx={None if stale else result["dx"]} dy={None if stale else result["dy"]}']
                else:
                    ms = result['inference_ms']
                    lines += [f'SAM MASK (DELAYED): {1000 / ms if ms else 0:.1f} FPS | {ms:.0f} ms',
                              f'Latency {result["latency_ms"]:.0f} ms | image age {age * 1000:.0f} ms']
            for i, line in enumerate(lines):
                cv2.putText(tracked, line, (15, 30 + 27 * i), 0, .6, (0, 220, 255), 2)
            import numpy as np
            if selection['enabled']:
                cv2.putText(live, 'Drag a box on LIVE preview', (15, 460), 0, .6, (0, 255, 255), 2)
            if selection['start'] and selection['end']:
                cv2.rectangle(live, selection['start'], selection['end'], (0, 255, 255), 2)
            cv2.imshow(name, np.hstack((live, tracked)))
            # Control/telemetry always describes live motion, regardless of UI view.
            result = motion_result if motion_result and motion_result['generation'] == generation else None
            age = None if result is None else time.perf_counter() - result['captured']
            stale = age is not None and age > .2
            if result and result['sam_age_ms'] is not None:
                stale |= result['sam_age_ms'] + age * 1000 > args.max_sam_age * 1000
            export_key = (generation, result['seq'], stale) if result else None
            if result is None and last_export != 'no_target':
                telemetry.submit({'valid': False, 'status': 'SELECTING', 'dx': None, 'dy': None})
                last_export = 'no_target'
            if result and export_key != last_export:
                data = {k: v for k, v in result.items() if k not in ('mask', 'frame')}
                data['valid'] = result['box'] is not None and not stale
                data['timestamp_unix'] = time.time()
                data['source_age_ms'] = age * 1000
                if not data['valid']:
                    data.update(dx=None, dy=None, box=None)
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
            if key == ord('m'):
                args.view = 'sam' if args.view == 'live' else 'live'
                cached_key = None
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if capture.done:
                break
    finally:
        capture.stop.set()
        capture.thread.join(timeout=3)
        worker.thread.join(timeout=5)
        motion_worker.thread.join(timeout=2)
        cv2.destroyAllWindows()
        telemetry.close()


if __name__ == '__main__':
    main()
