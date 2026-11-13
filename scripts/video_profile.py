"""Thread-safe stage timings; wall-clock display FPS is separate from SAM FPS."""
from collections import defaultdict, deque
from functools import wraps
import json
from pathlib import Path
import threading
import time
import numpy as np

ACTIVE = None


def sample(name, seconds):
    if ACTIVE is not None:
        ACTIVE.sample(name, seconds)


def timed(name):
    def decorate(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = None
            try:
                result = function(*args, **kwargs)
                return result
            finally:
                if not (name == 'sam' and isinstance(result, dict) and result.get('predictor_called') is False):
                    sample(name, time.perf_counter()-start)
        return wrapper
    return decorate


class Profile:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.stages = defaultdict(lambda:deque(maxlen=10000))
        self.window = defaultdict(lambda:deque(maxlen=10000))
        self.stage_counts=defaultdict(int);self.stage_totals=defaultdict(float)
        self.times = deque(maxlen=120)
        self.first = self.last_report = time.perf_counter()
        self.frames = self.tracking = 0
        self.age = deque(maxlen=10000);self.age_total=0.
        self.live_fps = 0.

    def sample(self, name, seconds):
        with self.lock:
            self.stages[name].append(seconds*1000)
            self.window[name].append(seconds*1000)
            self.stage_counts[name]+=1;self.stage_totals[name]+=seconds*1000

    def frame(self, result):
        now = time.perf_counter()
        if not self.frames:
            self.first = now
        self.frames += 1
        self.tracking += result.get('status') in ('TRACKING', 'SMALL','RECOVERED')
        self.times.append(now)
        self.live_fps = (len(self.times)-1)/max(1e-9, now-self.times[0])
        self.age.append(result.get('age_ms', 0.) or 0.)
        self.age_total+=result.get('age_ms',0.) or 0.
        if now-self.last_report >= 1:
            with self.lock:
                stages, self.window = self.window, defaultdict(lambda:deque(maxlen=10000))
            values = ' | '.join(f'{k} {np.mean(v):.2f}/{np.percentile(v,95):.2f}' for k,v in stages.items() if v)
            print(f'PROFILE mean/p95 ms | Live {self.live_fps:.1f} FPS | {values}', flush=True)
            self.last_report = now

    def close(self):
        elapsed = time.perf_counter()-self.first
        with self.lock:
            stages = {k:dict(calls=self.stage_counts[k], mean_ms=self.stage_totals[k]/self.stage_counts[k], p95_ms=float(np.percentile(v,95)),p95_window_calls=len(v))
                      for k,v in self.stages.items() if v}
        result = dict(frames=self.frames, elapsed_s=elapsed,
                      live_fps=max(0,self.frames-1)/max(1e-9,elapsed),
                      sam_updates_fps=stages.get('sam',{}).get('calls',0)/max(1e-9,elapsed),
                      sam_compute_fps=1000/max(1e-9,stages.get('sam',{}).get('mean_ms',1e9)),
                      tracking_percent=100*self.tracking/max(1,self.frames),
                      age_mean_ms=self.age_total/max(1,self.frames), stages=stages)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(result,indent=2), encoding='utf-8')
        print(f'PROFILE saved {self.path}: Live {result["live_fps"]:.2f} FPS, SAM updates {result["sam_updates_fps"]:.2f} FPS', flush=True)
        return result
