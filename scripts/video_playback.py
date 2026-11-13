"""File transport for camera.py; tracking stays in the shared live pipeline."""
import csv
import json
import math
import time
from pathlib import Path

import cv2
from video_profile import sample, timed


def record_selection(csv_path, frame, box):
    xywh = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
    text = ','.join(f'{v:g}' for v in xywh)
    prompt = f'--init-point {xywh[0]:g},{xywh[1]:g}' if xywh[2:] == [0, 0] else f'--init-bbox {text}'
    print(f'SELECTION frame={frame} bbox_xywh={text} | --init-frame {frame} {prompt}', flush=True)
    path = Path(csv_path).with_suffix('.selections.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if header:
            writer.writerow(('frame', 'x', 'y', 'w', 'h'))
        writer.writerow((frame, *xywh))


class VideoSource:
    def __init__(self, path, start=0, width=None, interpolation=cv2.INTER_LINEAR, preserve_original=False):
        self.interpolation = interpolation
        self.preserve_original = preserve_original
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f'Cannot open video: {path}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.
            print('Video has no valid FPS; using 30 FPS.', flush=True)
        self.width = width
        sidecar = Path(path).with_suffix('.timestamps.csv')
        self.timestamps = None
        if sidecar.exists():
            with sidecar.open(encoding='utf-8') as handle:
                self.timestamps = [float(row['time_s']) for row in csv.DictReader(handle)]
            if not self.timestamps or any(b <= a for a,b in zip(self.timestamps,self.timestamps[1:])):
                raise ValueError('Raw timestamps must be strictly increasing')
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
        tick = time.perf_counter()
        ok, frame = self.cap.read()
        sample('decode', time.perf_counter()-tick)
        if not ok:
            return None
        self.index += 1
        original = frame
        if self.width is not None and frame.shape[1] != self.width:
            tick = time.perf_counter()
            height = max(1, round(frame.shape[0] * self.width / frame.shape[1]))
            frame = cv2.resize(frame, (self.width, height), interpolation=self.interpolation)
            sample('resize', time.perf_counter()-tick)
        if self.preserve_original:
            from source_frame import SourceFrame
            frame = SourceFrame(frame, original)
        stamp = self.timestamps[self.index] if self.timestamps is not None else self.index/self.fps
        return self.index, stamp, frame

    def index_at_time(self, stamp):
        if self.timestamps is None:
            return int(stamp*self.fps)
        import bisect
        return bisect.bisect_right(self.timestamps, stamp)-1

    def close(self):
        self.cap.release()


def select_box(frame, allow_point=False):
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
            a,b = state['start'],state['end']
            if max(abs(a[0]-b[0]),abs(a[1]-b[1])) < 24 and state['zoom'] < 4:
                state['zoom'] = 4.
                factor = scale*state['zoom']
                state['x'] = min(max(0.,(a[0]+b[0])/2-vw/(2*factor)),max(0.,width-vw/factor))
                state['y'] = min(max(0.,(a[1]+b[1])/2-vh/(2*factor)),max(0.,height-vh/factor))

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
                if allow_point and max(box[2]-box[0],box[3]-box[1]) < 2:
                    x,y = (box[0]+box[2])/2,(box[1]+box[3])/2
                    return [x,y,x,y]
    finally:
        cv2.destroyWindow(name)


class FrameLog:
    fields = ('pass', 'frame', 'time_s', 'status', 'x1', 'y1', 'x2', 'y2',
              'center_x', 'center_y', 'dx', 'dy', 'similarity', 'latency_ms', 'age_ms', 'sam_fps',
              'selected_bbox', 'transition', 'transition_reason', 'iou', 'object_logit', 'mask_area',
              'sam_reason', 'sam_threshold', 'bbox_area_ratio', 'displacement', 'displacement_limit',
              'yolo_person', 'thresholds', 'target_mode', 'correction_reason', 'search_reason',
              'motion_reason', 'crop_box', 'coco_class', 'low_score_streak', 'low_histogram_streak',
              'search_top2', 'search_hits', 'live_fps', 'sam_seq', 'sam_latency_ms', 'estimated', 'flow_valid', 'sam_status',
              'predictor_called', 'object_score_logits', 'predicted_iou', 'sam_observation_valid', 'keep_running',
              'probation', 'gallery_size', 'pinned_memory', 'lab_similarity', 'attempt_id', 'reacquired', 'pipeline_ms',
              'small', 'native_box', 'source_width', 'source_height', 'identity_area_ratio',
              'tile_count', 'tile_total', 'tile_hits', 'tile_similarity', 'tile_top2', 'tile_search_ms', 'tile_candidate_debug',
              'recovery_policy', 'anchor_similarity', 'shape_ratio', 'anchor_mask_iou',
              'memory_frozen', 'color_used_for_acceptance', 'motion_distance', 'motion_limit', 'lost_seconds',
              'raw_mask_area','mask_components','component_selected','flow_mask_iou','continuity_distance',
              'continuity_limit','temporal_area_ratio','continuity_flow_valid','ambiguous','candidate_scores',
              'confirmed_wrong','confirmed_correct','suspected_merge','display_latency_ms',
              'valid','generation','queue_ms','worker_compute_ms','preprocess_ms','gpu_core_ms',
              'postprocess_ms','sam_core_wall_ms','recovery_checks_ms','confidence','confidence_parts',
              'control_gain','proposal_count','identity_candidates','chosen_proposal','proposal_signals',
              'probe_age_ms','probe_ms','kalman_innovation','kalman_limit','identity_ambiguous',
              'anchor_probe_reason','identity_probe_reason','quarantine_size','sam_frame_idx',
              'conditioning_count','recent_memory_count','continuity_dt_s','continuity_tau','continuity_area_limit','memory_distractor_overlap','probe_wait_reason','target_pixels','model_target_pixels','appearance_score','roi_reinitialized','appearance_samples','target_id','frame_id','state','target_center','normalized_error_x','normalized_error_y','target_velocity','identity_score','tracking_quality','is_visible','is_confirmed','recommended_control_mode','gimbal_command','kalman_box','proposal_top1','proposal_top2','state_events','mask_stability','flow_inlier_ratio','identity_gallery','gpu_allocated_mb','search_stage','search_box')

    fields += ('partial_observation','partial_evidence','features','proposal_search_ms','search_rotation')

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path=path
        self.handle = path.open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields)
        self.writer.writeheader()
        self.processed = self.tracking = self.lost = self.recovered = self.skipped = 0
        self.active_seconds = 0.
        self.previous = None
        from collections import defaultdict
        self.state_seconds=defaultdict(float)
        self.last_stamp=self.last_cycle=self.last_generation=None
        self.last_display_state=None
        self.had_lock=self.missing=False

    @timed('csv')
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
            value = result.get(field, False if field == 'predictor_called' else None)
            row[field] = json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
        self.writer.writerow(row)
        self.handle.flush()
        same_segment=(self.last_stamp is not None and cycle==self.last_cycle and stamp>=self.last_stamp)
        if same_segment:
            self.state_seconds[self.last_display_state]+=stamp-self.last_stamp
        self.last_stamp,self.last_cycle,self.last_display_state=stamp,cycle,status
        if status == 'SKIPPED':
            self.skipped += 1
            return
        self.processed += 1
        self.active_seconds += elapsed
        self.tracking += status in ('TRACKING', 'SMALL','RECOVERED')
        sam_status = result.get('sam_status',status)
        from session_audit import locked
        manual=result.get('selected_bbox') is not None or result.get('transition_reason')=='anchor_initialized'
        generation=result.get('generation')
        if manual or not same_segment or (generation is not None and generation!=self.last_generation):
            self.had_lock=self.missing=False
        self.last_generation=generation
        if locked(dict(status=sam_status,valid=result.get('valid',''),x1=None if box is None else box[0])):
            self.recovered+=int(self.missing)
            self.had_lock=True;self.missing=False
        elif self.had_lock and not self.missing:
            self.lost+=1;self.missing=True
        self.previous = sam_status

    def summary(self):
        print(f'Video summary: frames={self.processed}, skipped={self.skipped}, '
              f'TRACKING={100 * self.tracking / max(1, self.processed):.1f}%, '
              f'lock_losses={self.lost}, lock_resumes={self.recovered} (not identity accuracy).', flush=True)
        print('Observed state seconds: '+', '.join(f'{k}={v:.2f}' for k,v in sorted(self.state_seconds.items())),flush=True)
        from session_audit import audit
        self.handle.flush()
        audit(self.path,verbose=False)
        print(f'State/event audit: {self.path.with_suffix(".audit.json")}',flush=True)


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
    name = 'SOURCE SAM2 VIDEO | P pause | A/D step | S/R select | W/C label | Q/Esc quit'
    try:
        source = VideoSource(source_path, init_frame, args.resize_width,
                             preserve_original=getattr(args, 'sam_only', True))
        packet = source.read()
        if packet is None:
            raise ValueError('No frame at --start-frame')
        tracker, _ = create_tracker(args)
        from video_audit import MaskAudit
        audit = MaskAudit(args.audit_masks) if getattr(args, 'audit_masks', None) else None
        import video_profile
        profile = None
        if getattr(args, 'profile', None):
            profile = video_profile.Profile(args.profile)
            video_profile.ACTIVE = profile
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
            if audit and output['status'] != 'SKIPPED':
                audit.add(current[0], current[2], output)
            log.write(cycle, current, output, elapsed)
            from mask_display import overlay
            view = overlay(current[2], output, getattr(args,'overlay_quality','full'))
            if getattr(args,'debug_tracking',False):
                from small_target import debug_hud
                debug_hud(view,output)
            text = f'SAM 2.1 | FPS {1 / elapsed if elapsed > 0 else 0:.1f}'
            cv2.putText(view, text, (10, 25), 0, .6, (0, 220, 255), 2)
            if args.save_video:
                if writer is None:
                    args.save_video.parent.mkdir(parents=True, exist_ok=True)
                    codec = 'MJPG' if args.save_video.suffix.lower() == '.avi' else 'mp4v'
                    writer = cv2.VideoWriter(str(args.save_video), cv2.VideoWriter_fourcc(*codec),
                                             source.fps, (view.shape[1], view.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f'Cannot write video: {args.save_video}')
                tick = time.perf_counter()
                writer.write(view)
                sample('video_write', time.perf_counter()-tick)
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
                tick = time.perf_counter()
                view = export(packet, result, elapsed)
                sample('export_total', time.perf_counter()-tick)
                data = {k: v for k, v in result.items() if k not in ('mask', 'search_masks','candidate_frame')}
                data.update(frame=packet[0], source_time=packet[1], latency_ms=elapsed * 1000,
                            valid=result.get('valid', result.get('box') is not None))
                tick = time.perf_counter()
                telemetry.submit(data)
                sample('telemetry_submit', time.perf_counter()-tick)
                box, dirty = None, False
                limit_reached = args.max_frames is not None and log.processed >= args.max_frames
            display = view.copy()
            if not no_display:
                tick = time.perf_counter()
                cv2.imshow(name, display)
                sample('imshow', time.perf_counter()-tick)
            if ready and profile:
                profile.frame(result)
            key = -1 if no_display else cv2.waitKey(10 if paused else 1) & 255
            if key == ord('h'):
                args.debug_tracking = not getattr(args,'debug_tracking',False)
            if key in (ord('k'),ord('i'),ord('o')) and hasattr(tracker,'toggle_feature'):
                tracker.toggle_feature({ord('k'):'kalman',ord('i'):'reid',ord('o'):'motion_flow'}[key])
            if key in (27, ord('q')) or (not no_display and cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1) or limit_reached:
                break
            if key in (ord('s'), ord('r')):
                chosen = select_box(packet[2], allow_point=getattr(args,'sam_only',True))
                if chosen is not None:
                    record_selection(csv_path, packet[0], chosen)
                    box, active, dirty = chosen, True, True
                    pending_started = None
                    log.previous = None  # manual selection isn't automatic recovery
                anchor = time.perf_counter() - packet[1]
                continue
            if key in (ord('w'), ord('c')) and active and hasattr(tracker, 'label'):
                applied=tracker.label(key == ord('c'), packet[1], result.get('attempt_id', 0))
                from recovery.selection import record_feedback
                record_feedback(csv_path,packet[0],packet[1],result.get('attempt_id',0),key==ord('c'),applied,result.get('confidence'))
            if ord('1') <= key <= ord('9'):
                from recovery.selection import candidate_box, record_candidate
                chosen = candidate_box(result,key-ord('1'))
                if chosen is not None:
                    if result.get('candidate_frame') is not None:
                        packet=(result.get('candidate_seq',packet[0]),result['candidate_stamp'],result['candidate_frame'])
                    record_candidate(csv_path,packet[0],key-ord('1'),chosen)
                    box,active,dirty=chosen,True,True
                    pending_started=None
                    continue
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
        if 'profile' in locals() and profile:
            profile.close()
            video_profile.ACTIVE = None
        if 'audit' in locals() and audit:
            audit.close()
        if not no_display:
            cv2.destroyAllWindows()
