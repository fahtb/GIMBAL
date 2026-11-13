"""Bounded, causal admission of SAM-produced memory (never flow predictions)."""
from collections import deque


class MemoryQuarantine:
    def __init__(self):
        self.entries=deque()
        self.since=None
        self.observations=0

    def clear(self):
        self.entries.clear();self.since=None;self.observations=0

    def observe(self,sam,stamp,eligible,cfg):
        if not eligible:
            self.clear()
            return False
        if self.since is None:
            self.since=stamp
        self.observations+=1
        # Clone only compact memory, not the encoder frame/full prediction.
        entry=sam.snapshot_pending_memory()
        self.entries.append(entry)
        capacity=int(cfg.get('quarantine_frames',3))
        while len(self.entries)>capacity:
            self.entries.popleft()
        ready=(self.observations>=capacity and stamp-self.since>=cfg.get('quarantine_s',.2))
        if not ready:
            return False
        index,compact,bbox,score= self.entries.popleft()
        sam.store_verified_memory(index,compact,bbox,score,cfg['pinned_frames'])
        return True
