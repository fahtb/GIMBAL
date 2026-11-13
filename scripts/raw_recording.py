"""Bounded asynchronous raw recording with per-capture timestamp sidecar."""
import csv
from pathlib import Path
import queue
import threading
import cv2


class RawRecorder:
    def __init__(self, path, fps=30., capacity=90):
        self.path, self.fps = Path(path), fps
        if self.path.exists() or self.path.with_suffix('.timestamps.csv').exists():
            raise FileExistsError(f'Recording already exists: {self.path}')
        self.queue = queue.Queue(capacity)
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True, name='raw-writer')
        self.thread.start()

    def submit(self, packet):
        if self.error:
            raise RuntimeError(self.error)
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            raise RuntimeError('Raw recording queue full: disk too slow; recording incomplete')

    def run(self):
        writer = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.with_suffix('.timestamps.csv').open('x', newline='', encoding='utf-8') as handle:
                log = csv.writer(handle)
                log.writerow(('frame','source_seq','time_s','capture_monotonic_s'))
                origin, index = None, 0
                while True:
                    packet = self.queue.get()
                    if packet is None:
                        break
                    seq, stamp, frame = packet
                    if writer is None:
                        origin = stamp
                        codec = 'MJPG' if self.path.suffix.lower()=='.avi' else 'mp4v'
                        writer = cv2.VideoWriter(str(self.path),cv2.VideoWriter_fourcc(*codec),self.fps,
                                                 (frame.shape[1],frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError(f'Cannot write {self.path}')
                    writer.write(frame)
                    log.writerow((index,seq,stamp-origin,stamp))
                    index += 1
        except Exception as exc:
            self.error = str(exc)
        finally:
            if writer is not None:
                writer.release()

    def close(self):
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=.1)
                break
            except queue.Full:
                pass
        self.thread.join(timeout=10)
        if self.thread.is_alive() or self.error:
            raise RuntimeError(self.error or 'Recording writer did not finish')
