"""Frame-level diagnostics, also usable with the unchanged baseline tracker."""
import csv
import time
from datetime import datetime, timezone
from pathlib import Path


class SearchLog:
    fields = ('session', 'utc', 'frame_time', 'state', 'candidates', 'top1', 'top2',
              'threshold', 'high_threshold', 'required_margin', 'actual_margin',
              'streak', 'required_streak', 'reason', 'detector_ms', 'new_detection',
              'lost_time', 'first_candidate_time', 'accepted_time', 'region', 'gallery_size')

    def __init__(self, path):
        self.path = Path(path)
        self.last_warning = 0.
        self.failed_rows = 0

    def write(self, **row):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists() or self.path.stat().st_size == 0
            with self.path.open('a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, self.fields)
                if new:
                    writer.writeheader()
                writer.writerow(dict(utc=datetime.now(timezone.utc).isoformat(), **row))
        except OSError as exc:
            self.failed_rows += 1
            if time.monotonic()-self.last_warning > 5:
                print(f'Search CSV unavailable ({self.failed_rows} missed rows): {exc}', flush=True)
                self.last_warning = time.monotonic()


def instrument_baseline(base):
    class MeasuredBaseline(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            original_detector = self.detector
            class TimedDetector:
                def detect(self, frame):
                    tick = time.perf_counter()
                    boxes = original_detector.detect(frame)
                    self.last_ms = (time.perf_counter()-tick)*1000
                    return boxes
            self.detector = TimedDetector()
            self.search_log = SearchLog(str(self.log.path).replace('.csv', '_frames.csv'))
            self.first_candidate = None
            self.detector_ms = 0.
            self.candidate_count = 0
            self.scanned = False

        def lose(self, stamp):
            self.first_candidate = None
            super().lose(stamp)

        def rank(self, frame):
            tick = time.perf_counter()
            ranked = super().rank(frame)
            self.detector_ms = self.detector.last_ms
            self.scanned = True
            self.candidate_count = len(ranked)
            if ranked and self.first_candidate is None:
                self.first_candidate = self.last_input
                self.event('first_candidate', self.last_input)
            return ranked

        def step(self, frame, stamp, box=None):
            before = self.state
            self.scanned = False
            result = super().step(frame, stamp, box)
            if before in ('LOST', 'SEARCHING', 'REACQUIRE'):
                reason = ('waiting_detector_interval' if not self.scanned else
                          'no_candidates' if not self.candidate_count else
                          'below_threshold' if self.top1 <= self.cfg.similarity else
                          'insufficient_margin' if self.top1-self.top2 <= self.cfg.margin else
                          'accepted' if self.state == 'TRACKING' else
                          'sam_verification' if self.state == 'REACQUIRE' else 'insufficient_streak')
                self.search_log.write(session=self.session, frame_time=stamp, state=self.state,
                    candidates=self.candidate_count, top1=self.top1, top2=self.top2,
                    threshold=self.cfg.similarity, high_threshold='', required_margin=self.cfg.margin,
                    actual_margin=self.top1-self.top2, streak=self.hits,
                    required_streak=self.cfg.consecutive_frames, reason=reason,
                    detector_ms=self.detector_ms if self.scanned else 0., new_detection=self.scanned,
                    lost_time=self.lost_at, first_candidate_time=self.first_candidate,
                    accepted_time=stamp if self.state == 'TRACKING' else '', region='full',
                    gallery_size=len(self.gallery))
            return result
    return MeasuredBaseline
