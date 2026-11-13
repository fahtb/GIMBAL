"""Choose PERSON/OBJECT once per explicit selection; never change person gates."""
from person_reid import ReIDTracker, iou
from object_tracking import ObjectTracker


class TargetTracker:
    def __init__(self, sam, detector, person_config, object_config, mode='auto'):
        self.sam, self.detector = sam, detector
        self.person = ReIDTracker(sam, detector, person_config)
        self.object = ObjectTracker(sam, detector, object_config)
        self.requested_mode, self.target_mode = mode, None
        self.current = self.person

    @property
    def state(self):
        return self.current.state

    def step(self, frame, stamp, box=None, detections=None):
        people = None
        if box is not None:
            # Use the original person detector settings for the PERSON decision.
            people = self.detector.detect(frame)
            matches_person = any(iou(box, b) >= self.person.cfg.person_match_iou for b in people)
            self.target_mode = ('PERSON' if matches_person else 'OBJECT') if self.requested_mode == 'auto' else self.requested_mode.upper()
            if self.target_mode == 'PERSON':
                self.object.close()
                self.current, self.sam.policy = self.person, 'person'
            else:
                detections_all = self.detector.detect_all(frame, size=self.object.cfg.detector_size)
                matches = [(iou(box,d['box']), d['class_id']) for d in detections_all]
                best = max(matches, default=(0., None))
                class_id = best[1] if best[0] >= self.object.cfg.class_match_iou else None
                self.current = self.object
                print(f'Target mode OBJECT | COCO class={class_id} | recovery={"YOLO class" if class_id is not None else "DINOv2-small"}', flush=True)
                result = self.object.initialize(frame, stamp, box, class_id)
                result['yolo_person'] = people
                return result
            print('Target mode PERSON | original person ReID gates', flush=True)
        result = self.current.step(frame, stamp, box, detections=detections)
        result['target_mode'] = self.target_mode
        if people is not None:
            result['yolo_person'] = people
        return result

    def poll_search(self, frame, stamp):
        return self.object.poll_search(frame, stamp) if self.target_mode == 'OBJECT' else None

    def label(self, correct, stamp, attempt_id=None):
        if self.target_mode == 'PERSON':
            return self.person.label(correct, stamp, attempt_id)
        return False

    def close(self):
        self.object.close()
