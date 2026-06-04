from __future__ import annotations

import math
import os
from typing import Callable, Iterable


Embedding = tuple[float, ...]
DeepEmbeddingProvider = Callable[[object, tuple[float, float, float, float]], Embedding | None]
_DEEP_PROVIDER: DeepEmbeddingProvider | None = None


def set_deep_embedding_provider(provider: DeepEmbeddingProvider | None) -> None:
    """Register an optional OSNet/FastReID-style embedding provider.

    The core pipeline stays dependency-light. If a deployment has a stronger
    person Re-ID model available, register a callable that returns a normalized
    embedding for `(frame, bbox)` and `crop_embedding` will concatenate it with
    the handcrafted descriptor.
    """
    global _DEEP_PROVIDER
    _DEEP_PROVIDER = provider


def cosine_similarity(left: Iterable[float] | None, right: Iterable[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    a = tuple(float(v) for v in left)
    b = tuple(float(v) for v in right)
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def embedding_distance(left: Iterable[float] | None, right: Iterable[float] | None) -> float:
    return 1.0 - cosine_similarity(left, right)


def blend_embedding(current: Embedding | None, incoming: Embedding | None, samples: int, alpha: float = 0.25) -> Embedding | None:
    if incoming is None:
        return current
    if current is None or samples <= 0 or len(current) != len(incoming):
        return incoming
    weight = max(0.05, min(0.95, alpha))
    blended = tuple((1.0 - weight) * c + weight * i for c, i in zip(current, incoming))
    norm = math.sqrt(sum(v * v for v in blended))
    if norm <= 1e-12:
        return blended
    return tuple(v / norm for v in blended)


def crop_embedding(frame, bbox: tuple[float, float, float, float]) -> Embedding | None:
    """Return a compact color/shape embedding for a person crop.

    This deliberately avoids a heavyweight Re-ID dependency so the pipeline
    still runs on reviewer machines. The vector combines HSV histograms from
    the full crop and torso band with normalized box shape features.
    """
    if frame is None:
        return None
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(width - 1, x1)))
    x2 = int(max(0, min(width, x2)))
    y1 = int(max(0, min(height - 1, y1)))
    y2 = int(max(0, min(height, y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    full_hist = _hsv_hist(cv2, hsv, bins=(16, 10), weight=0.35)
    stripe_vectors = []
    stripes = [
        (0.00, 0.28, 0.55),  # head/shoulder cue
        (0.20, 0.58, 1.00),  # torso is the strongest clothing cue
        (0.55, 1.00, 0.65),  # lower-body cue
    ]
    for start, end, weight in stripes:
        sy1 = int(crop.shape[0] * start)
        sy2 = max(sy1 + 1, int(crop.shape[0] * end))
        stripe_vectors.append(_hsv_hist(cv2, hsv[sy1:sy2, :], bins=(12, 8), weight=weight))

    edge_vector = _edge_descriptor(cv2, crop) * 0.22
    dominant = _dominant_color_descriptor(np, hsv) * 0.18
    shape = np.array(
        [
            box_w / max(1.0, float(width)),
            box_h / max(1.0, float(height)),
            box_w / box_h,
            (x1 + x2) / (2.0 * max(1.0, float(width))),
            (y1 + y2) / (2.0 * max(1.0, float(height))),
        ],
        dtype="float32",
    )
    parts = [full_hist, *stripe_vectors, edge_vector, dominant, shape * 0.10]
    deep = _optional_deep_embedding(frame, bbox)
    if deep is not None:
        parts.append(np.array(deep, dtype="float32") * 1.15)
    vector = np.concatenate(parts).astype("float32")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return tuple(float(v) for v in vector / norm)


def _hsv_hist(cv2, hsv_crop, bins: tuple[int, int], weight: float):
    hist = cv2.calcHist([hsv_crop], [0, 1], None, list(bins), [0, 180, 0, 256]).flatten()
    cv2.normalize(hist, hist)
    return hist.astype("float32") * weight


def _edge_descriptor(cv2, crop):
    import numpy as np

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 64), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    bins = np.zeros(8, dtype="float32")
    bin_idx = np.floor((ang % 180) / 22.5).astype("int32")
    for i in range(8):
        bins[i] = float(mag[bin_idx == i].sum())
    norm = float(np.linalg.norm(bins))
    return bins if norm <= 1e-12 else bins / norm


def _dominant_color_descriptor(np, hsv_crop):
    # Mean/std per channel plus coarse hue occupancy. It helps when histograms
    # are similar but one person has a distinctive dominant outfit color.
    pixels = hsv_crop.reshape(-1, 3).astype("float32")
    means = pixels.mean(axis=0) / np.array([180.0, 255.0, 255.0], dtype="float32")
    stds = pixels.std(axis=0) / np.array([180.0, 255.0, 255.0], dtype="float32")
    hue_bins, _ = np.histogram(pixels[:, 0], bins=8, range=(0, 180))
    hue_bins = hue_bins.astype("float32")
    hue_norm = float(np.linalg.norm(hue_bins))
    if hue_norm > 1e-12:
        hue_bins = hue_bins / hue_norm
    return np.concatenate([means, stds, hue_bins]).astype("float32")


def _optional_deep_embedding(frame, bbox: tuple[float, float, float, float]) -> Embedding | None:
    provider = _DEEP_PROVIDER
    if provider is None and os.getenv("STORE_INTEL_REID_PROVIDER"):
        provider = _load_provider(os.environ["STORE_INTEL_REID_PROVIDER"])
    if provider is None:
        return None
    try:
        embedding = provider(frame, bbox)
    except Exception:
        return None
    if embedding is None:
        return None
    values = tuple(float(v) for v in embedding)
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-12:
        return None
    return tuple(v / norm for v in values)


def _load_provider(spec: str) -> DeepEmbeddingProvider | None:
    import importlib

    if ":" not in spec:
        return None
    module_name, function_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        provider = getattr(module, function_name)
    except Exception:
        return None
    return provider if callable(provider) else None
