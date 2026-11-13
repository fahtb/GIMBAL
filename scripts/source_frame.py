"""Display-sized ndarray carrying one original decoded frame for SAM crops.

Copies intentionally drop the original reference, so display/output queues do
not retain full-resolution video. Only the bounded inference packet retains it.
"""
import numpy as np


class SourceFrame(np.ndarray):
    def __new__(cls, display, original):
        value = np.asarray(display).view(cls)
        value.original = original
        return value

    def __array_finalize__(self, source):
        self.original = None
