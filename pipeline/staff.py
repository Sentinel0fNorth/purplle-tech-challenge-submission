from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

try:
    import cv2  # type: ignore
    import numpy as np
except Exception:  # pragma: no cover
    cv2 = None
    np = None


@dataclass
class StaffClassifier:
    uniform_rgb: tuple[tuple[int, int, int], ...] = ()
    uniform_hsv: tuple[tuple[int, int, int], ...] = ()
    threshold: float = 65.0
    hsv_threshold: float = 55.0

    def classify(self, frame, bbox: tuple[float, float, float, float]) -> tuple[bool, float]:
        """Return (is_staff, confidence). Heuristic fallback; can be replaced by FashionCLIP.

        The torso crop is used because uniforms and lanyards usually appear there. If no uniform
        palette is configured the classifier deliberately returns false with low confidence rather
        than inventing staff labels.
        """
        if frame is None or np is None or cv2 is None:
            return False, 0.5
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return False, 0.5
        torso_y1 = y1 + int((y2 - y1) * 0.20)
        torso_y2 = y1 + int((y2 - y1) * 0.65)
        crop = frame[torso_y1:torso_y2, x1:x2]
        if crop.size == 0:
            return False, 0.5

        # If HSV profile is defined, perform HSV-based classification.
        if self.uniform_hsv:
            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mean_hsv = hsv_crop.reshape(-1, 3).mean(axis=0)
            h_val, s_val, v_val = mean_hsv[0], mean_hsv[1], mean_hsv[2]
            distances = []
            for target in self.uniform_hsv:
                t_h, t_s, t_v = target
                # Hue difference taking cyclicity of 180 into account
                dh = min(abs(h_val - t_h), 180 - abs(h_val - t_h))
                ds = abs(s_val - t_s)
                # Compute distance using Hue and Saturation, scaling Hue by 2 to align weights (0-180 vs 0-255)
                dist = math.sqrt((dh * 2.0)**2 + ds**2)
                distances.append(dist)
            best = min(distances)
            is_staff = best <= self.hsv_threshold
            confidence = max(0.51, min(0.99, 1.0 - best / 255.0)) if is_staff else max(0.5, min(0.95, best / 255.0))
            return is_staff, round(confidence, 3)

        # Fallback to BGR Euclidean distance
        if not self.uniform_rgb:
            return False, 0.5
        # Convert OpenCV BGR to RGB average.
        mean_rgb = crop.reshape(-1, 3).mean(axis=0)[::-1]
        distances = [float(np.linalg.norm(mean_rgb - np.array(rgb))) for rgb in self.uniform_rgb]
        best = min(distances)
        is_staff = best <= self.threshold
        confidence = max(0.51, min(0.99, 1.0 - best / 255.0)) if is_staff else max(0.5, min(0.95, best / 255.0))
        return is_staff, round(confidence, 3)
