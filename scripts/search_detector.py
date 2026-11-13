"""Dedicated latest-frame detector worker; one output slot, no frame backlog."""
import threading
import time


class SearchDetector:
    def __init__(self, capture, detector):
        self.capture, self.detector = capture, detector
        self.enabled = threading.Event()
        self.lock = threading.Lock()
        self.result = None
        self.error = None
        self.thread = threading.Thread(target=self.run, name='person-search', daemon=True)
        self.thread.start()

    def run(self):
        last_seq = -1
        try:
            while not self.capture.stop.is_set():
                if not self.enabled.wait(.02):
                    continue
                packet = self.capture.get()
                if packet is None or packet[0] <= last_seq:
                    self.capture.stop.wait(.002)
                    continue
                seq, stamp, frame = packet
                tick = time.perf_counter()
                boxes = self.detector.detect(frame)
                ms = (time.perf_counter()-tick)*1000
                with self.lock:
                    self.result = (packet, boxes, ms)
                last_seq = seq
        except Exception as exc:
            self.error = str(exc)

    def get(self):
        with self.lock:
            return self.result
