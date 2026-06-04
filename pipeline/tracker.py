from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from .layout import StoreLayout, Zone, camera_for_id, sku_zone_for_bbox, zone_for_bbox, foot_point, point_in_polygon
from .reid import Embedding, blend_embedding, cosine_similarity


@dataclass
class Detection:
    camera_id: str
    track_id: str
    bbox: tuple[float, float, float, float]
    confidence: float
    frame_index: int
    timestamp: datetime
    is_staff: bool = False
    embedding: Embedding | None = None


@dataclass
class VisitorState:
    visitor_id: str
    camera_track_key: str
    store_id: str
    current_zone: str | None = None
    current_sku_zone: str | None = None
    zone_entered_at: datetime | None = None
    last_dwell_emit_at: datetime | None = None
    last_seen_at: datetime | None = None
    last_bbox: tuple[float, float, float, float] | None = None
    is_staff: bool = False
    session_seq: int = 0
    queue_joined: bool = False
    billing_left_at: datetime | None = None
    billing_abandon_emitted: bool = False
    exited: bool = False
    entry_side_history: list[str] = field(default_factory=list)
    entry_seen_inside: bool = False
    track_keys: set[str] = field(default_factory=set)
    last_camera_track_key: str | None = None
    appearance_embedding: Embedding | None = None
    appearance_samples: int = 0
    reid_matches: int = 0
    last_reid_score: float | None = None


class RetailStateMachine:
    """Frame-by-frame retail event generator.

    This class is intentionally model-agnostic: ByteTrack/YOLO supplies detections with stable
    track IDs and this state machine converts them into the challenge JSON event schema.
    """

    def __init__(
        self,
        layout: StoreLayout,
        fps: float = 15.0,
        dwell_seconds: int = 30,
        line_confirm_frames: int = 3,
        queue_abandon_seconds: int = 300,
        reid_threshold: float = 0.82,
        reid_min_appearance: float = 0.78,
        reid_time_window_seconds: int = 900,
    ) -> None:
        self.layout = layout
        self.fps = fps
        self.dwell_delta = timedelta(seconds=dwell_seconds)
        self.line_confirm_frames = max(1, line_confirm_frames)
        self.queue_abandon_delta = timedelta(seconds=queue_abandon_seconds)
        self.reid_threshold = reid_threshold
        self.reid_min_appearance = reid_min_appearance
        self.reid_time_window = timedelta(seconds=reid_time_window_seconds)
        self.track_to_visitor: dict[str, VisitorState] = {}
        self.retired_entry_visitors: list[VisitorState] = []
        self._frame_claimed_visitors: set[str] = set()

    def _visitor_for(self, det: Detection) -> VisitorState:
        key = f"{det.camera_id}:{det.track_id}"
        if key in self.track_to_visitor:
            state = self.track_to_visitor[key]
            self._frame_claimed_visitors.add(state.visitor_id)
            return state
        state, score = self._match_existing_visitor(det, key)
        if state is None:
            visitor_id = f"VIS_{uuid4().hex[:8]}"
            state = VisitorState(visitor_id=visitor_id, camera_track_key=key, store_id=self.layout.store_id)
        else:
            state.reid_matches += 1
            state.last_reid_score = score
        state.track_keys.add(key)
        state.last_camera_track_key = key
        self.track_to_visitor[key] = state
        self._frame_claimed_visitors.add(state.visitor_id)
        return state

    def _match_existing_visitor(self, det: Detection, key: str) -> tuple[VisitorState | None, float | None]:
        if det.embedding is None:
            return None, None
        best_state: VisitorState | None = None
        best_score = self.reid_threshold
        camera = key.split(":", 1)[0]
        candidates = {state.visitor_id: state for state in self.track_to_visitor.values()}
        for state in candidates.values():
            if state.visitor_id in self._frame_claimed_visitors:
                continue
            if key in state.track_keys:
                continue
            if state.last_seen_at is None or det.timestamp - state.last_seen_at > self.reid_time_window:
                continue
            if any(existing_key.startswith(f"{camera}:") for existing_key in state.track_keys):
                continue
            appearance_score = cosine_similarity(det.embedding, state.appearance_embedding)
            if appearance_score < self.reid_min_appearance:
                continue
            elapsed = max(0.0, (det.timestamp - state.last_seen_at).total_seconds())
            score = appearance_score
            score += self._transition_bonus(state, det.camera_id)
            score += self._time_bonus(elapsed)
            
            # Spatio-temporal constraint with frame-freeze override
            if state.last_camera_track_key is not None:
                prev_camera = state.last_camera_track_key.split(":", 1)[0]
                if prev_camera != det.camera_id:
                    # Shopper transitions between different cameras. Enforce travel constraint.
                    if elapsed < 3.0:
                        # If elapsed is too small, check if it's a freeze (very high appearance match)
                        if appearance_score < 0.90:
                            # Apply heavy penalty to prevent matching
                            score -= 0.35
            
            if score > best_score:
                best_score = score
                best_state = state
        return best_state, best_score if best_state else None

    def _time_bonus(self, elapsed_seconds: float) -> float:
        if elapsed_seconds <= 20:
            return 0.05
        if elapsed_seconds <= 120:
            return 0.03
        if elapsed_seconds <= 300:
            return 0.01
        return -0.03

    def _transition_bonus(self, state: VisitorState, new_camera_id: str) -> float:
        if state.last_camera_track_key is None:
            return 0.0
        prev_camera_id = state.last_camera_track_key.split(":", 1)[0]
        if prev_camera_id == new_camera_id:
            return -0.08
        prev_role = (camera_for_id(self.layout, prev_camera_id).role if camera_for_id(self.layout, prev_camera_id) else "").lower()
        new_role = (camera_for_id(self.layout, new_camera_id).role if camera_for_id(self.layout, new_camera_id) else "").lower()
        if "entry" in prev_role and ("zone" in new_role or "floor" in new_role):
            return 0.06
        if ("zone" in prev_role or "floor" in prev_role) and "billing" in new_role:
            return 0.06
        if "billing" in prev_role and "entry" in new_role:
            return 0.04
        return 0.02

    def _update_appearance(self, state: VisitorState, det: Detection) -> None:
        if det.embedding is None:
            return
        state.appearance_embedding = blend_embedding(state.appearance_embedding, det.embedding, state.appearance_samples)
        state.appearance_samples += 1

    def _event(
        self,
        state: VisitorState,
        det: Detection,
        event_type: str,
        zone: Zone | None,
        dwell_ms: int = 0,
        queue_depth: int | None = None,
        sku_zone: str | None = None,
    ) -> dict[str, Any]:
        state.session_seq += 1
        return {
            "event_id": str(uuid4()),
            "store_id": self.layout.store_id,
            "camera_id": det.camera_id,
            "visitor_id": state.visitor_id,
            "event_type": event_type,
            "timestamp": det.timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "zone_id": zone.zone_id if zone else None,
            "dwell_ms": dwell_ms,
            "is_staff": state.is_staff,
            "confidence": round(max(0.0, min(1.0, det.confidence)), 4),
            "metadata": {
                "queue_depth": queue_depth,
                "sku_zone": sku_zone if sku_zone is not None else (zone.sku_zone if zone else None),
                "session_seq": state.session_seq,
                "group_size": None,
                "reid_matches": state.reid_matches,
                "reid_score": round(state.last_reid_score, 4) if state.last_reid_score is not None else None,
            },
        }

    @staticmethod
    def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _velocity_px_per_sec(self, previous_bbox, current_bbox, dt_seconds: float) -> float:
        if previous_bbox is None or dt_seconds <= 0:
            return float("inf")
        x1, y1 = self._center(previous_bbox)
        x2, y2 = self._center(current_bbox)
        return math.hypot(x2 - x1, y2 - y1) / dt_seconds

    def _entry_line_y(self, det: Detection, zone: Zone | None) -> float | None:
        camera = camera_for_id(self.layout, det.camera_id)
        if camera:
            if camera.entry_line is not None:
                p1, p2 = camera.entry_line
                x1, y1 = p1
                x2, y2 = p2
                foot_x = (det.bbox[0] + det.bbox[2]) / 2.0
                if abs(x2 - x1) < 1e-6:
                    return (y1 + y2) / 2.0
                return y1 + ((y2 - y1) / (x2 - x1)) * (foot_x - x1)
            if camera.entry_line_y is not None:
                return float(camera.entry_line_y)
        if zone and zone.zone_id == "ENTRY" and zone.polygon:
            ys = [point[1] for point in zone.polygon]
            return (min(ys) + max(ys)) / 2.0
        return None

    def _entry_zone_for_camera(self, camera_id: str) -> Zone | None:
        return next(
            (
                zone
                for zone in self.layout.zones
                if zone.zone_id == "ENTRY" and (not zone.camera_ids or camera_id in zone.camera_ids)
            ),
            None,
        )

    def _entry_direction_event(self, state: VisitorState, det: Detection, zone: Zone | None) -> str | None:
        line_y = self._entry_line_y(det, zone)
        if line_y is None:
            return None
        _, foot_y = ((det.bbox[0] + det.bbox[2]) / 2.0, det.bbox[3])
        camera = camera_for_id(self.layout, det.camera_id)
        side_op = camera.entry_line_side if camera else "below"
        if side_op == "above":
            side = "inside" if foot_y <= line_y else "outside"
        else:
            side = "inside" if foot_y >= line_y else "outside"
        state.entry_side_history.append(side)
        if len(state.entry_side_history) > self.line_confirm_frames:
            state.entry_side_history = state.entry_side_history[-self.line_confirm_frames :]
        confirmed_inside = len(state.entry_side_history) == self.line_confirm_frames and all(
            value == "inside" for value in state.entry_side_history
        )
        confirmed_outside = len(state.entry_side_history) == self.line_confirm_frames and all(
            value == "outside" for value in state.entry_side_history
        )
        if confirmed_inside and not state.entry_seen_inside:
            state.entry_seen_inside = True
            state.exited = False
            return "REENTRY" if state.visitor_id in {s.visitor_id for s in self.retired_entry_visitors} else "ENTRY"
        if confirmed_outside and state.entry_seen_inside and not state.exited:
            state.exited = True
            state.entry_seen_inside = False
            return "EXIT"
        return None

    def flush_stale_queue(self, now: datetime) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for state in self.track_to_visitor.values():
            if (
                state.queue_joined
                and state.billing_left_at is not None
                and not state.billing_abandon_emitted
                and now - state.billing_left_at >= self.queue_abandon_delta
            ):
                prev_zone = next((z for z in self.layout.zones if z.zone_id == "BILLING"), None)
                if state.last_seen_at is None or state.last_bbox is None:
                    continue
                det = Detection(
                    camera_id=(state.last_camera_track_key or state.camera_track_key).split(":", 1)[0],
                    track_id=(state.last_camera_track_key or state.camera_track_key).split(":", 1)[1],
                    bbox=state.last_bbox,
                    confidence=0.75,
                    frame_index=0,
                    timestamp=now,
                    is_staff=state.is_staff,
                )
                events.append(self._event(state, det, "BILLING_QUEUE_ABANDON", prev_zone, queue_depth=0))
                state.billing_abandon_emitted = True
        return events

    def process(self, detections: list[Detection]) -> list[dict[str, Any]]:
        # Pre-process detections to split group entries near the entry threshold
        split_detections: list[Detection] = []
        for det in detections:
            entry_zone = self._entry_zone_for_camera(det.camera_id)
            w_box = det.bbox[2] - det.bbox[0]
            h_box = det.bbox[3] - det.bbox[1]
            aspect_ratio = w_box / h_box if h_box > 0 else 0
            
            # If in entry zone and aspect ratio is abnormally wide (group entry)
            if entry_zone and aspect_ratio > 0.85 and point_in_polygon(foot_point(det.bbox), entry_zone.polygon):
                num_people = max(2, int(round(w_box / (h_box * 0.45))))
                slice_w = w_box / num_people
                for i in range(num_people):
                    vx1 = det.bbox[0] + i * slice_w
                    vx2 = vx1 + slice_w
                    vbbox = (vx1, det.bbox[1], vx2, det.bbox[3])
                    split_detections.append(
                        Detection(
                            camera_id=det.camera_id,
                            track_id=f"{det.track_id}_split_{i}",
                            bbox=vbbox,
                            confidence=det.confidence,
                            frame_index=det.frame_index,
                            timestamp=det.timestamp,
                            is_staff=det.is_staff,
                            embedding=det.embedding,
                        )
                    )
            else:
                split_detections.append(det)
        detections = split_detections

        events: list[dict[str, Any]] = []
        active_billing: set[str] = set()
        group_size_by_camera: dict[str, int] = {}
        self._frame_claimed_visitors = set()
        for det in detections:
            group_size_by_camera[det.camera_id] = group_size_by_camera.get(det.camera_id, 0) + 1

        # First pass updates states and collects live billing occupancy.
        for det in detections:
            state = self._visitor_for(det)
            state.is_staff = state.is_staff or det.is_staff
            state.last_seen_at = det.timestamp
            state.last_camera_track_key = f"{det.camera_id}:{det.track_id}"
            self._update_appearance(state, det)
            zone = zone_for_bbox(self.layout, det.camera_id, det.bbox)
            if zone and zone.zone_id == "BILLING" and not state.is_staff:
                active_billing.add(state.visitor_id)

        queue_depth = len(active_billing)

        latest_ts = max((det.timestamp for det in detections), default=None)
        if latest_ts is not None:
            events.extend(self.flush_stale_queue(latest_ts))

        for det in detections:
            state = self._visitor_for(det)
            zone = zone_for_bbox(self.layout, det.camera_id, det.bbox)
            previous_zone = state.current_zone
            camera_group_size = group_size_by_camera.get(det.camera_id)
            sku_zone = sku_zone_for_bbox(self.layout, det.camera_id, det.bbox, zone)
            entry_zone = zone if zone and zone.zone_id == "ENTRY" else self._entry_zone_for_camera(det.camera_id)

            if entry_zone is not None:
                event_type = self._entry_direction_event(state, det, entry_zone)
                if event_type:
                    event = self._event(state, det, event_type, None)
                    event["metadata"]["group_size"] = camera_group_size
                    events.append(event)

            if previous_zone is not None and (zone is None or zone.zone_id != previous_zone):
                prev_zone = next((z for z in self.layout.zones if z.zone_id == previous_zone), None)
                dwell_ms = 0
                if state.zone_entered_at:
                    dwell_ms = int((det.timestamp - state.zone_entered_at).total_seconds() * 1000)
                event = self._event(state, det, "ZONE_EXIT", prev_zone, dwell_ms=dwell_ms, sku_zone=state.current_sku_zone)
                event["metadata"]["group_size"] = camera_group_size
                events.append(event)
                if previous_zone == "BILLING" and not state.is_staff:
                    state.billing_left_at = det.timestamp
                state.current_zone = None
                state.current_sku_zone = None
                state.zone_entered_at = None
                state.last_dwell_emit_at = None

            if zone and zone.zone_id != previous_zone:
                event = self._event(state, det, "ZONE_ENTER", zone, sku_zone=sku_zone)
                event["metadata"]["group_size"] = camera_group_size
                events.append(event)
                state.current_zone = zone.zone_id
                state.current_sku_zone = sku_zone
                state.zone_entered_at = det.timestamp
                state.last_dwell_emit_at = det.timestamp
                if zone.zone_id == "BILLING" and not state.is_staff and queue_depth > 0 and not state.queue_joined:
                    event = self._event(state, det, "BILLING_QUEUE_JOIN", zone, queue_depth=queue_depth, sku_zone=sku_zone)
                    event["metadata"]["group_size"] = camera_group_size
                    events.append(event)
                    state.queue_joined = True
                    state.billing_left_at = None
                    state.billing_abandon_emitted = False

            elif zone and state.current_zone == zone.zone_id and state.zone_entered_at:
                due_from = state.last_dwell_emit_at or state.zone_entered_at
                while det.timestamp - due_from >= self.dwell_delta:
                    due_from = due_from + self.dwell_delta
                    dwell_ms = int((due_from - state.zone_entered_at).total_seconds() * 1000)
                    event = self._event(state, det, "ZONE_DWELL", zone, dwell_ms=dwell_ms, sku_zone=sku_zone)
                    event["metadata"]["group_size"] = camera_group_size
                    events.append(event)
                    state.last_dwell_emit_at = due_from

            if state.exited and state not in self.retired_entry_visitors:
                self.retired_entry_visitors.append(state)

            state.last_bbox = det.bbox
        return events
