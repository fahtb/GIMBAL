"""Replaceable appearance backend and bounded anchor + diverse rolling gallery."""
from collections import deque
from dataclasses import dataclass
from typing import Protocol
import cv2
import numpy as np
import warnings


class EmbeddingBackend(Protocol):
    # An OSNet adapter can return its own descriptor type and cosine score here.
    # Existing HSV descriptors must be rebuilt on backend changes, never mixed.
    def encode(self, frame, box, mask=None): ...
    def compare(self, query, reference): ...  # (similarity, visible_region)


@dataclass(frozen=True)
class FallbackDescriptor:
    hsv: object
    learned: object


class HSVFallbackBackend:
    """Inject an OSNet adapter later; retain HSV descriptors for safe fallback.

    Adapter implements encode/compare. Never compare vectors from different
    spaces. Its score calibration is the adapter owner's responsibility.
    """
    def __init__(self, primary, hsv=None):
        self.primary, self.hsv = primary, hsv or VisibleHSV()
        self.warned = False

    def warn(self, exc):
        if not self.warned:
            warnings.warn(f'Embedding backend failed; using HSV fallback: {exc}', RuntimeWarning)
            self.warned = True

    def encode(self, frame, box, mask=None):
        hsv = self.hsv.encode(frame, box, mask)
        learned = None
        try:
            learned = self.primary.encode(frame, box, mask)
        except Exception as exc:
            self.warn(exc)
        return FallbackDescriptor(hsv, learned) if hsv is not None or learned is not None else None

    def compare(self, query, reference):
        if query is None or reference is None:
            return 0., 'invalid'
        if query.learned is not None and reference.learned is not None:
            try:
                return self.primary.compare(query.learned, reference.learned)
            except Exception as exc:
                self.warn(exc)
        value, region = self.hsv.compare(query.hsv, reference.hsv)
        return value, 'hsv_fallback:'+region


@dataclass(frozen=True)
class HSVDescriptor:
    bands: np.ndarray
    crop: str
    aspect: float
    feature: np.ndarray
    templates: dict


class VisibleHSV:
    def __init__(self, edge_fraction=.015, min_visible=.35):
        self.edge_fraction, self.min_visible = edge_fraction, min_visible

    def encode(self, frame, box, mask=None):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        if x2-x1 < 8 or y2-y1 < 24:
            return None
        top, bottom = y1 <= h*self.edge_fraction, y2 >= h*(1-self.edge_fraction)
        side = x1 <= w*self.edge_fraction or x2 >= w*(1-self.edge_fraction)
        aspect = (y2-y1)/max(1, x2-x1)
        crop = 'middle' if top and bottom else 'lower' if top else 'upper' if bottom else 'side' if side else 'full'
        if crop == 'full' and aspect < 1.2:
            crop = 'upper_abnormal'
        hsv = cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (64, 256)), cv2.COLOR_BGR2HSV)
        # Same central crop on SAM and detector boxes. Store histograms, no images.
        rows = []
        for start in range(0, 256, 8):
            region = hsv[start:start+8, 12:52]
            hs = cv2.calcHist([region], [0,1], None, [16,8], [0,180,0,256]).ravel()
            v = cv2.calcHist([region], [2], None, [16], [0,256]).ravel()
            rows.append(np.r_[hs, v])
        bands = np.asarray(rows, dtype=np.float32)
        bands.setflags(write=False)
        feature = self.feature(bands)
        templates = {}
        if crop in ('full','side'):
            for kind in ('upper','lower','middle'):
                vectors, labels = [], []
                for count in range(max(12, round(32*self.min_visible)),33,2):
                    start = 32-count if kind == 'lower' else (32-count)//2 if kind == 'middle' else 0
                    vectors.append(self.feature(bands[start:start+count]))
                    labels.append(f'{kind}:{start}/32+{count}/32')
                matrix = np.asarray(vectors, np.float32)
                matrix.setflags(write=False)
                templates[kind] = (matrix, labels)
        feature.setflags(write=False)
        return HSVDescriptor(bands, crop, aspect, feature, templates)

    @staticmethod
    def feature(bands):
        edges = [0, max(1, round(len(bands)*.2)), max(2, round(len(bands)*.6)), len(bands)]
        parts = []
        for a, b in zip(edges, edges[1:]):
            raw = bands[a:b].sum(0)
            vector = np.r_[np.sqrt(raw[:128]/max(1, raw[:128].sum())),
                           np.sqrt(raw[128:]/max(1, raw[128:].sum()))]
            parts.append(vector/max(np.linalg.norm(vector), 1e-8))
        return np.concatenate(parts)/np.sqrt(3)

    def compare(self, query, reference):
        if query is None or reference is None:
            return 0., 'invalid'
        # Compare only visible portions of the full reference for edge truncation.
        if query.crop not in ('full', 'side') and reference.crop in ('full', 'side'):
            partial, full = query, reference
        elif reference.crop not in ('full', 'side') and query.crop in ('full', 'side'):
            partial, full = reference, query
        else:
            return float(np.clip(query.feature @ reference.feature, 0, 1)), query.crop
        # Aspect-based scale estimate is unreliable when arms are extended. Search
        # a bounded set of visible extents, never a tiny matching patch.
        kind = partial.crop if partial.crop in ('upper','lower','middle') else 'upper'
        matrix, labels = full.templates[kind]
        values = matrix @ partial.feature
        index = int(np.argmax(values))
        return float(np.clip(values[index], 0, 1)), labels[index]


class IdentityGallery:
    def __init__(self, backend, size=20, top_k=1, duplicate_similarity=.985):
        self.backend, self.size, self.top_k = backend, size, top_k
        self.duplicate_similarity = duplicate_similarity
        self.anchor = None
        self.rolling = deque(maxlen=max(1, size-1))

    def __len__(self):
        return (self.anchor is not None) + len(self.rolling)

    def seed(self, descriptor):
        self.anchor = descriptor
        self.rolling.clear()

    def score(self, descriptor):
        references = ([self.anchor] if self.anchor is not None else []) + list(self.rolling)
        scores = sorted((self.backend.compare(descriptor, item) for item in references), reverse=True)
        if not scores:
            return 0., 'empty'
        return float(np.mean([s[0] for s in scores[:self.top_k]])), scores[0][1]

    def add(self, descriptor):
        if descriptor is None or self.anchor is None or self.size < 2:
            return False
        # Always use max for duplicate detection, regardless of scoring top-k.
        references = [self.anchor, *self.rolling]
        if max(self.backend.compare(descriptor, item)[0] for item in references) >= self.duplicate_similarity:
            return False
        self.rolling.append(descriptor)
        return True
