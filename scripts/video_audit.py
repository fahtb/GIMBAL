"""Optional packed-mask evidence for repeatable same-source-frame comparisons."""
import json
from pathlib import Path
import numpy as np


class MaskAudit:
    def __init__(self, path):
        self.path, self.rows, self.masks = Path(path), [], []
        self.shape = None

    def add(self, index, frame, result):
        self.shape = frame.shape[:2]
        mask = result.get('mask')
        if mask is None:
            mask = np.zeros(self.shape, bool)
        self.masks.append(np.packbits(mask))
        self.rows.append(dict(frame=index, status=result['status'], box=result.get('box'),
                              age_ms=result.get('age_ms',0.), sam_seq=result.get('sam_seq')))

    def close(self):
        if self.rows:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(self.path, masks=np.asarray(self.masks), shape=self.shape,
                                rows=json.dumps(self.rows))
