"""SAM-only Step 1 diagnostic adapter. No detector or embedding imports.

Keep-running measures raw SAM recovery; it deliberately does not accept a new
identity after loss. That requires the separate recovery/probation stage.
"""


def sam_fps(result):
    milliseconds = result.get('inference_ms', 0.) or 0.
    return 1000. / milliseconds if milliseconds > 0 and result.get('predictor_called', True) else 0.


class SAMOnlyTracker:
    target_mode = 'SAM_ONLY'

    def __init__(self, sam, keep_running=False, lost_stride=1):
        if lost_stride < 1:
            raise ValueError('--lost-stride must be >= 1')
        self.sam, self.keep_running, self.lost_stride = sam, keep_running, lost_stride
        self.sam.policy = 'sam_only'
        self.state, self.lost_frames = 'WAIT_SELECTION', 0

    def step(self, frame, stamp, box=None):
        before = self.state
        searching = box is None and before in ('LOST', 'SEARCHING')
        if box is not None:
            self.lost_frames = 0
        run = not searching or (self.keep_running and self.lost_frames % self.lost_stride == 0)
        if searching:
            self.lost_frames += 1
        if run:
            result = self.sam.step(frame, stamp, box, continue_lost=searching)
            observation = result['status'].startswith('TRACKING')
            self.state = 'SEARCHING' if searching else 'TRACKING' if observation else 'LOST'
            reason = 'diagnostic_keep_running' if searching else result.get('sam_reason', '')
        else:
            self.state = 'SEARCHING'
            reason = 'lost_stride_skip' if self.keep_running else 'sam_stopped_after_loss'
            result = dict(mask=None, box=None, dx=None, dy=None, inference_ms=0.,
                          predictor_called=False, sam_reason=reason)
            observation = None
        result.update(status=self.state, target_mode=self.target_mode,
                      valid=self.state == 'TRACKING', transition=f'{before}->{self.state}',
                      transition_reason=reason, sam_observation_valid=observation,
                      thresholds={'min_iou': self.sam.min_iou, 'object_logit_min': 0,
                                  'min_area': 16, 'lost_stride': self.lost_stride},
                      keep_running=self.keep_running)
        if searching:
            result.update(mask=None, box=None, dx=None, dy=None)
        return result

    def label(self, correct, stamp, attempt_id=None):
        # No recovery identity decision exists yet; keep the UI callback safe.
        return False
