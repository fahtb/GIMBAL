"""File transport for camera.py; tracking stays in the shared live pipeline."""
import csv
import json
import math
import time
from pathlib import Path

import cv2


def record_selection(csv_path, frame, box):
    xywh = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
    text = ','.join(f'{v:g}' for v in xywh)
    print(f'SELECTION frame={frame} bbox_xywh={text} | --init-frame {frame} --init-bbox {text}', flush=True)
    path = Path(csv_path).with_suffix('.selections.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if header:
            writer.writerow(('frame', 'x', 'y', 'w', 'h'))
        writer.writerow((frame, *xywh))


class VideoSource:
    def __init__(self, path, start=0, width=None):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f'Cannot open video: {path}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.
            print('Video has no valid FPS; using 30 FPS.', flush=True)
        self.width = width
        self.index = -1
        try:
            self.seek(start)
        except Exception:
            self.close()
            raise

    def seek(self, index):
        # Decode from the beginning for reliable exact seeks, including codecs
        # whose CAP_PROP_POS_FRAMES seeking lands on a nearby keyframe.
        if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            raise RuntimeError('Video backend does not support seeking')
        for _ in range(index):
            if not self.cap.grab():
                raise ValueError(f'Frame {index} is outside the video')
        self.index = index - 1

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.index += 1
        if self.width is not None and frame.shape[1] != self.width:
            height = max(1, round(frame.shape[0] * self.width / frame.shape[1]))
            frame = cv2.resize(frame, (self.width, height))
        return self.index, self.index / self.fps, frame

    def close(self):
        self.cap.release()


def select_box(frame):
    """Wheel zoom about the pointer; drag in viewport; Enter accepts."""
    height, width = frame.shape[:2]
    vw, vh = min(1200, width), min(800, height)
    scale = min(vw / width, vh / height)
    vw, vh = max(1, round(width * scale)), max(1, round(height * scale))
    state = dict(zoom=1., x=0., y=0., start=None, end=None)
    name = 'Select: wheel zoom | drag box | Enter accept | Esc cancel'

    def point(x, y):
        return (min(width - 1, max(0., state['x'] + x / (scale * state['zoom']))),
                min(height - 1, max(0., state['y'] + y / (scale * state['zoom']))))

    def mouse(event, x, y, flags, _):
        if event == cv2.EVENT_MOUSEWHEEL:
            px, py = point(x, y)
            state['zoom'] = min(16., max(1., state['zoom'] * (1.5 if flags > 0 else 1 / 1.5)))
            factor = scale * state['zoom']
            state['x'] = min(max(0., px - x / factor), max(0., width - vw / factor))
            state['y'] = min(max(0., py - y / factor), max(0., height - vh / factor))
        elif event == cv2.EVENT_LBUTTONDOWN:
            state['start'] = state['end'] = point(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            state['end'] = point(x, y)
        elif event == cv2.EVENT_LBUTTONUP and state['start'] is not None:
            state['end'] = point(x, y)

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(name, mouse)
    try:
        while True:
            import numpy as np
            factor = scale * state['zoom']
            matrix = np.array([[factor, 0, -state['x'] * factor],
                               [0, factor, -state['y'] * factor]], dtype=float)
            view = cv2.warpAffine(frame, matrix, (vw, vh))
            if state['start'] is not None:
                points = [tuple(round((v - state[k]) * factor) for v, k in zip(p, ('x', 'y')))
                          for p in (state['start'], state['end'])]
                cv2.rectangle(view, *points, (0, 255, 255), 1)
            cv2.imshow(name, view)
            key = cv2.waitKey(15) & 255
            if key in (27, ord('q')) or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in (10, 13, 32) and state['start'] is not None:
                a, b = state['start'], state['end']
                box = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
                if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                    return box
    finally:
        cv2.destroyWindow(name)


class FrameLog:
    fields = ('pass', 'frame', 'time_s', 'status', 'x1', 'y1', 'x2', 'y2',
              'center_x', 'center_y', 'dx', 'dy', 'similarity', 'latency_ms', 'age_ms', 'sam_fps',
              'selected_bbox', 'transition', 'transition_reason', 'iou', 'object_logit', 'mask_area',
              'sam_reason', 'sam_threshold', 'bbox_area_ratio', 'displacement', 'displacement_limit',
              'yolo_person', 'thresholds', 'target_mode', 'correction_reason', 'search_reason',
              'motion_reason', 'crop_box', 'coco_class', 'low_score_streak', 'low_histogram_streak',
              'search_top2', 'search_hits')

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields)
        self.writer.writeheader()
        self.processed = self.tracking = self.lost = self.recovered = self.skipped = 0
        self.active_seconds = 0.
        self.previous = None

    def write(self, cycle, packet, result, elapsed):
        index, stamp, _ = packet
        status = result['status']
        box = result.get('box')
        row = dict(zip(self.fields[:4], (cycle, index, stamp, status)))
        if box is not None:
            row.update(zip(('x1', 'y1', 'x2', 'y2'), box))
            row.update(center_x=(box[0] + box[2]) / 2, center_y=(box[1] + box[3]) / 2)
        ms = result.get('inference_ms', 0.)
        row.update(dx=result.get('dx'), dy=result.get('dy'), similarity=result.get('reid_top1'),
                   latency_ms=elapsed * 1000, age_ms=result.get('age_ms', elapsed * 1000),
                   sam_fps=1000 / ms if ms else 0.)
        for field in self.fields[16:]:
            value = result.get(field)
            row[field] = json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
        self.writer.writerow(row)
        self.handle.flush()
        if status == 'SKIPPED':
            self.skipped += 1
            return
        self.processed += 1
        self.active_seconds += elapsed
        self.tracking += status == 'TRACKING'
        self.lost += status.startswith('LOST') and self.previous == 'TRACKING'
        self.recovered += status == 'TRACKING' and self.previous in ('LOST', 'SEARCHING', 'REACQUIRE')
        self.previous = status

    def summary(self):
        print(f'Video summary: frames={self.processed}, skipped={self.skipped}, '
              f'TRACKING={100 * self.tracking / max(1, self.processed):.1f}%, '
              f'lost={self.lost}, reacquired={self.recovered}, '
              f'average processing FPS={self.processed / max(.000001, self.active_seconds):.2f}', flush=True)


def run_video(args, create_tracker, draw, telemetry_class):
    source_path = Path(args.source).resolve()
    csv_path = args.csv or (args.save_video.with_suffix('.csv') if args.save_video else
                            Path(__file__).resolve().parents[1] / 'runtime' / f'{source_path.stem}_frames.csv')
    paths = [source_path, csv_path.resolve()]
    if args.save_video:
        paths.append(args.save_video.resolve())
    if len(set(paths)) != len(paths):
        raise ValueError('Input video, output video and CSV must have different paths')
    source = log = telemetry = writer = None
    no_display = getattr(args, 'no_display', False)
    initial_box = getattr(args, 'init_bbox', None)
    init_frame = getattr(args, 'init_frame', None)
    init_frame = args.start_frame if init_frame is None else init_frame
    name = 'SAMURAI VIDEO | P pause | A/D step | S/R select | W/C label | Q/Esc quit'
    try:
        source = VideoSource(source_path, init_frame, args.resize_width)
        packet = source.read()
        if packet is None:
            raise ValueError('No frame at --start-frame')
        tracker, _ = create_tracker(args)
        log = FrameLog(csv_path)
        telemetry = telemetry_class(args.telemetry)
        telemetry.submit(dict(valid=False, status='WAIT_SELECTION', dx=None, dy=None))
        print(f'Video {source.fps:g} FPS | {args.mode} | CSV: {csv_path}', flush=True)
        if not no_display:
            print('S/R select, wheel zoom, drag, Enter; P play. A/D step while paused.', flush=True)
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        paused, active, dirty, cycle = True, False, False, 0
        box = None
        if initial_box:
            x, y, w, h = initial_box
            if x + w > packet[2].shape[1] or y + h > packet[2].shape[0]:
                raise ValueError('--init-bbox exceeds processing image dimensions')
            box = [x, y, x + w, y + h]
            record_selection(csv_path, packet[0], box)
            active, dirty, paused = True, True, False
        result = dict(status='WAIT_SELECTION', box=None, mask=None, dx=None, dy=None)
        anchor = time.perf_counter() - packet[1]
        limit_reached = False
        pending_started = None

        def export(current, output, elapsed):
            nonlocal writer
            log.write(cycle, current, output, elapsed)
            view = draw(current[2], output)
            text = (f'{current[0]} | {current[1]:.3f}s | {output["status"]} | '
                    f'SAM {1000 / output["inference_ms"]:.1f} FPS' if output.get('inference_ms') else
                    f'{current[0]} | {current[1]:.3f}s | {output["status"]}')
            cv2.putText(view, text, (10, 25), 0, .6, (0, 220, 255), 2)
            if args.save_video:
                if writer is None:
                    args.save_video.parent.mkdir(parents=True, exist_ok=True)
                    codec = 'MJPG' if args.save_video.suffix.lower() == '.avi' else 'mp4v'
                    writer = cv2.VideoWriter(str(args.save_video), cv2.VideoWriter_fourcc(*codec),
                                             source.fps, (view.shape[1], view.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f'Cannot write video: {args.save_video}')
                writer.write(view)
            return view

        view = draw(packet[2], result)
        while True:
            ready = False
            if dirty or pending_started is not None:
                started = pending_started if pending_started is not None else time.perf_counter()
                if pending_started is not None:
                    resolved = tracker.poll_search(packet[2], packet[1])
                    if resolved is not None:
                        result, pending_started, ready = resolved, None, True
                elif active:
                    result = tracker.step(packet[2], packet[1], box)
                    ready = True
                else:
                    result = dict(status='WAIT_SELECTION', box=None, mask=None, dx=None, dy=None)
                    ready = True
                if ready and result.get('search_pending') and args.mode == 'offline':
                    pending_started, ready = started, False
                dirty = False
            if ready:
                elapsed = time.perf_counter() - started
                if box is not None:
                    result['selected_bbox'] = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
                if result.get('transition') and result['transition'].split('->')[0] != result['status']:
                    print(f'frame={packet[0]} {result["transition"]}: {result.get("transition_reason")}', flush=True)
                result['age_ms'] = (max(elapsed, time.perf_counter() - anchor - packet[1])
                                    if args.mode == 'realtime' and not paused else elapsed) * 1000
                view = export(packet, result, elapsed)
                data = {k: v for k, v in result.items() if k != 'mask'}
                data.update(frame=packet[0], source_time=packet[1], latency_ms=elapsed * 1000,
                            valid=result.get('valid', result.get('box') is not None))
                telemetry.submit(data)
                box, dirty = None, False
                limit_reached = args.max_frames is not None and log.processed >= args.max_frames
            display = view.copy()
            cv2.putText(display, 'PAUSED | P play | S select | A/D step' if paused else 'PLAYING | P pause',
                        (10, max(20, display.shape[0] - 15)), 0, .6, (0, 255, 255), 2)
            if not no_display:
                cv2.imshow(name, display)
            key = -1 if no_display else cv2.waitKey(10 if paused else 1) & 255
            if key in (27, ord('q')) or (not no_display and cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1) or limit_reached:
                break
            if key in (ord('s'), ord('r')):
                chosen = select_box(packet[2])
                if chosen is not None:
                    record_selection(csv_path, packet[0], chosen)
                    box, active, dirty = chosen, True, True
                    pending_started = None
                    log.previous = None  # manual selection isn't automatic recovery
                anchor = time.perf_counter() - packet[1]
                continue
            if key in (ord('w'), ord('c')) and active and hasattr(tracker, 'label'):
                tracker.label(key == ord('c'), packet[1], result.get('attempt_id', 0))
            if key == ord('p'):
                paused = not paused
                anchor = time.perf_counter() - packet[1]
                if not paused and log.processed == 0:
                    dirty = True
                continue
            step = paused and key in (ord('a'), ord('d'))
            if pending_started is not None and not (step and key == ord('a')):
                if no_display:
                    time.sleep(.002)
                continue
            if paused and not step:
                continue
            if step and key == ord('a'):
                if packet[0] <= args.start_frame:
                    continue
                source.seek(packet[0] - 1)
                active = False  # chronological SAM memory cannot run backwards
                pending_started = None
                log.previous = None
                cycle += 1
            if not paused and args.mode == 'realtime':
                due = int((time.perf_counter() - anchor) * source.fps)
                if due <= packet[0]:
                    if no_display:
                        time.sleep(.001)
                    continue
            else:
                due = source.index + 1
            following = source.read()
            while following is not None and following[0] < due:
                export(following, dict(status='SKIPPED', box=None, mask=None), 0.)
                following = source.read()
            if following is None:
                if not args.loop or step:
                    break
                log.summary()
                source.seek(args.start_frame)
                following = source.read()
                active, paused = False, True
                if no_display and initial_box:
                    source.seek(init_frame)
                    following = source.read()
                    x, y, w, h = initial_box
                    box, active, paused = [x, y, x+w, y+h], True, False
                    record_selection(csv_path, following[0], box)
                    anchor = time.perf_counter() - following[1]
                log.previous = None
                cycle += 1
            packet, dirty = following, True
            if paused:
                anchor = time.perf_counter() - packet[1]
    finally:
        if source:
            source.close()
        if writer:
            writer.release()
        if log:
            log.summary()
            log.handle.close()
        if telemetry:
            telemetry.close()
        if 'tracker' in locals() and hasattr(tracker, 'close'):
            tracker.close()
        if not no_display:
            cv2.destroyAllWindows()
