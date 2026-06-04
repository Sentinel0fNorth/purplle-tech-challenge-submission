from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import cv2  # type: ignore
    import numpy as np
except Exception:  # pragma: no cover - pipeline can still run in simulation mode
    cv2 = None
    np = None


@dataclass(frozen=True)
class SkuSubZone:
    sku_zone: str
    polygon: list[tuple[float, float]]


@dataclass(frozen=True)
class Zone:
    zone_id: str
    polygon: list[tuple[float, float]]
    sku_zone: str | None = None
    camera_ids: tuple[str, ...] = ()
    sku_subzones: tuple[SkuSubZone, ...] = ()


@dataclass(frozen=True)
class Camera:
    camera_id: str
    role: str
    video_path: str | None = None
    entry_line_y: float | None = None
    entry_line: tuple[tuple[float, float], tuple[float, float]] | None = None
    entry_line_side: str = "below"


@dataclass(frozen=True)
class StoreLayout:
    store_id: str
    zones: tuple[Zone, ...]
    cameras: tuple[Camera, ...]
    staff_uniform_rgb: tuple[tuple[int, int, int], ...] = ()
    staff_uniform_hsv: tuple[tuple[int, int, int], ...] = ()


def _normalise_polygon(poly: Iterable[Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for point in poly:
        if isinstance(point, dict):
            points.append((float(point["x"]), float(point["y"])))
        else:
            points.append((float(point[0]), float(point[1])))
    return points


def _parse_store(raw: dict[str, Any]) -> StoreLayout:
    zones = []
    for z in raw.get("zones", []):
        subzones = tuple(
            SkuSubZone(
                sku_zone=item.get("sku_zone") or item.get("brand_name") or item.get("name"),
                polygon=_normalise_polygon(item.get("polygon", [])),
            )
            for item in z.get("sku_subzones", z.get("sku_zones", []))
        )
        zones.append(
            Zone(
                zone_id=z.get("zone_id") or z.get("name"),
                polygon=_normalise_polygon(z.get("polygon", [])),
                sku_zone=z.get("sku_zone"),
                camera_ids=tuple(z.get("camera_ids", z.get("camera_coverage", []))),
                sku_subzones=subzones,
            )
        )
    cameras = []
    for c in raw.get("cameras", []):
        raw_line = c.get("entry_line")
        entry_line = None
        if raw_line is not None:
            entry_line = ((float(raw_line[0][0]), float(raw_line[0][1])), (float(raw_line[1][0]), float(raw_line[1][1])))
        cameras.append(
            Camera(
                camera_id=c.get("camera_id") or c.get("id"),
                role=c.get("role", "unknown"),
                video_path=c.get("video_path"),
                entry_line_y=c.get("entry_line_y"),
                entry_line=entry_line,
                entry_line_side=c.get("entry_line_side", "below"),
            )
        )
    uniforms = tuple(tuple(map(int, rgb)) for rgb in raw.get("staff_uniform_rgb", []))
    uniforms_hsv = tuple(tuple(map(int, hsv)) for hsv in raw.get("staff_uniform_hsv", []))
    return StoreLayout(
        store_id=raw["store_id"],
        zones=tuple(zones),
        cameras=tuple(cameras),
        staff_uniform_rgb=uniforms,
        staff_uniform_hsv=uniforms_hsv,
    )


def load_layout(path: str | Path) -> dict[str, StoreLayout]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "stores" in data:
        stores = [_parse_store(s) for s in data["stores"]]
    elif "store_id" in data:
        stores = [_parse_store(data)]
    else:
        # Flexible challenge files often key by store_id.
        stores = [_parse_store({"store_id": sid, **body}) for sid, body in data.items()]
    return {s.store_id: s for s in stores}


def foot_point(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if not polygon:
        return False
    if cv2 is not None and np is not None:
        contour = np.array(polygon, dtype=np.float32)
        return cv2.pointPolygonTest(contour, point, False) >= 0
    # Pure Python ray casting fallback.
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def zone_for_bbox(layout: StoreLayout, camera_id: str, bbox: tuple[float, float, float, float]) -> Zone | None:
    point = foot_point(bbox)
    for zone in layout.zones:
        if zone.camera_ids and camera_id not in zone.camera_ids:
            continue
        if point_in_polygon(point, zone.polygon):
            return zone
    return None


def sku_zone_for_bbox(layout: StoreLayout, camera_id: str, bbox: tuple[float, float, float, float], zone: Zone | None = None) -> str | None:
    point = foot_point(bbox)
    candidate_zones = [zone] if zone is not None else list(layout.zones)
    for candidate in candidate_zones:
        if candidate is None:
            continue
        if candidate.camera_ids and camera_id not in candidate.camera_ids:
            continue
        for sku_subzone in candidate.sku_subzones:
            if point_in_polygon(point, sku_subzone.polygon):
                return sku_subzone.sku_zone
    return zone.sku_zone if zone else None


def camera_for_id(layout: StoreLayout, camera_id: str) -> Camera | None:
    return next((camera for camera in layout.cameras if camera.camera_id == camera_id), None)
