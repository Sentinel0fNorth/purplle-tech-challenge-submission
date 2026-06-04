from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Literal

from .emit import EventEmitter
from .layout import load_layout
from .reid import crop_embedding
from .staff import StaffClassifier
from .tracker import Detection, RetailStateMachine

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


@dataclass
class VideoInput:
    store_id: str
    camera_id: str
    path: str
    start_time: datetime


def _open_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is not available. Install project requirements first; if you see a NumPy/OpenCV ABI "
            "error, use the pinned numpy<2 requirement in requirements.txt."
        )
    return cv2


def probe_video_fps(path: str, fallback: float) -> float:
    if not path or cv2 is None:
        return fallback
    cap = cv2.VideoCapture(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return fps if fps > 0 else fallback
    finally:
        cap.release()


def read_manifest(path: str) -> list[VideoInput]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("videos", data if isinstance(data, list) else [])
    return [
        VideoInput(
            store_id=item["store_id"],
            camera_id=item["camera_id"],
            path=item["path"],
            start_time=parse_ts(item.get("start_time", "2026-03-03T14:00:00Z")),
        )
        for item in items
    ]


def simulate_detections(store_id: str, camera_id: str, start: datetime, fps: float = 15.0) -> Iterable[tuple[object | None, Detection]]:
    # Synthetic path: enters, dwells in skincare, joins billing queue, exits. Useful for demos/CI.
    for i in range(0, int(fps * 80)):
        t = start + timedelta(seconds=i / fps)
        if i < fps * 10:
            bbox = (860, 820 - i * 1.5, 960, 1040 - i * 1.5)  # ENTRY zone
        elif i < fps * 45:
            bbox = (360, 420, 470, 760)  # SKINCARE zone
        elif i < fps * 65:
            bbox = (1300, 430, 1420, 800)  # BILLING zone
        else:
            bbox = (870, 700 - (i - fps * 65) * 1.8, 960, 990 - (i - fps * 65) * 1.8)
        yield None, Detection(camera_id=camera_id, track_id="sim_1", bbox=bbox, confidence=0.88, frame_index=i, timestamp=t)


def run_yolo_video(
    video: VideoInput,
    layout_path: str,
    api_url: str | None,
    out: str | None,
    model_name: str,
    fps: float,
    conf: float,
    imgsz: int,
    device: str | None,
    tracker: str,
    max_frames: int | None = None,
    truncate_out: bool = True,
) -> int:
    from ultralytics import YOLO  # type: ignore

    layouts = load_layout(layout_path)
    layout = layouts[video.store_id]
    video_fps = probe_video_fps(video.path, fps)
    machine = RetailStateMachine(layout=layout, fps=video_fps)
    staff = StaffClassifier(layout.staff_uniform_rgb)
    emitter = EventEmitter(api_url=api_url, jsonl_path=out, truncate=truncate_out)
    model = YOLO(model_name)
    emitted = 0
    track_kwargs = {
        "source": video.path,
        "stream": True,
        "persist": True,
        "classes": [0],
        "tracker": tracker,
        "conf": conf,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device:
        track_kwargs["device"] = device
    results = model.track(**track_kwargs)
    for frame_index, result in enumerate(results):
        if max_frames is not None and frame_index >= max_frames:
            break
        frame = getattr(result, "orig_img", None)
        timestamp = video.start_time + timedelta(seconds=frame_index / video_fps)
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.id is None:
            machine.process([])
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        for bbox, conf_value, tid in zip(xyxy, confs, ids):
            bbox_tuple = tuple(float(v) for v in bbox)
            is_staff, staff_conf = staff.classify(frame, bbox_tuple)
            detections.append(
                Detection(
                    camera_id=video.camera_id,
                    track_id=str(tid),
                    bbox=bbox_tuple,
                    confidence=float(min(conf_value, staff_conf if is_staff else conf_value)),
                    frame_index=frame_index,
                    timestamp=timestamp,
                    is_staff=is_staff,
                    embedding=crop_embedding(frame, bbox_tuple),
                )
            )
        events = machine.process(detections)
        emitter.emit_many(events)
        emitted += len(events)
    emitter.flush()
    return emitted


def _motion_boxes(frame, subtractor, min_area: int) -> list[tuple[float, float, float, float, float]]:
    cv = _open_cv2()
    mask = subtractor.apply(frame)
    mask = cv.medianBlur(mask, 5)
    _, mask = cv.threshold(mask, 200, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=2)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[float, float, float, float, float]] = []
    for contour in contours:
        area = float(cv.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv.boundingRect(contour)
        if h < 45 or w < 18:
            continue
        confidence = max(0.35, min(0.75, area / 18000.0))
        boxes.append((float(x), float(y), float(x + w), float(y + h), confidence))
    return boxes


def _assign_motion_tracks(
    previous: dict[str, tuple[float, float, float, float]],
    boxes: list[tuple[float, float, float, float, float]],
    next_track_id: int,
    max_distance: float,
) -> tuple[list[tuple[str, tuple[float, float, float, float], float]], dict[str, tuple[float, float, float, float]], int]:
    assigned: list[tuple[str, tuple[float, float, float, float], float]] = []
    new_previous: dict[str, tuple[float, float, float, float]] = {}
    used: set[str] = set()

    def center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    for x1, y1, x2, y2, confidence in boxes:
        bbox = (x1, y1, x2, y2)
        cx, cy = center(bbox)
        best_id = None
        best_dist = max_distance
        for track_id, prev_bbox in previous.items():
            if track_id in used:
                continue
            px, py = center(prev_bbox)
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist < best_dist:
                best_id = track_id
                best_dist = dist
        if best_id is None:
            best_id = f"motion_{next_track_id}"
            next_track_id += 1
        used.add(best_id)
        new_previous[best_id] = bbox
        assigned.append((best_id, bbox, confidence))
    return assigned, new_previous, next_track_id


def run_motion_video(
    video: VideoInput,
    layout_path: str,
    api_url: str | None,
    out: str | None,
    fps: float,
    sample_every: int,
    min_area: int,
    max_frames: int | None = None,
    truncate_out: bool = True,
) -> int:
    cv = _open_cv2()
    layouts = load_layout(layout_path)
    layout = layouts[video.store_id]
    video_fps = probe_video_fps(video.path, fps)
    machine = RetailStateMachine(layout=layout, fps=video_fps)
    staff = StaffClassifier(layout.staff_uniform_rgb)
    emitter = EventEmitter(api_url=api_url, jsonl_path=out, truncate=truncate_out)
    cap = cv.VideoCapture(video.path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video.path}")
    subtractor = cv.createBackgroundSubtractorMOG2(history=450, varThreshold=32, detectShadows=True)
    previous: dict[str, tuple[float, float, float, float]] = {}
    next_track_id = 1
    emitted = 0
    frame_index = 0
    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if sample_every > 1 and frame_index % sample_every != 0:
                frame_index += 1
                continue
            timestamp = video.start_time + timedelta(seconds=frame_index / video_fps)
            raw_boxes = _motion_boxes(frame, subtractor, min_area=min_area)
            tracks, previous, next_track_id = _assign_motion_tracks(previous, raw_boxes, next_track_id, max_distance=90.0)
            detections: list[Detection] = []
            for track_id, bbox, confidence in tracks:
                is_staff, staff_conf = staff.classify(frame, bbox)
                detections.append(
                    Detection(
                        camera_id=video.camera_id,
                        track_id=track_id,
                        bbox=bbox,
                        confidence=float(min(confidence, staff_conf if is_staff else confidence)),
                        frame_index=frame_index,
                        timestamp=timestamp,
                        is_staff=is_staff,
                        embedding=crop_embedding(frame, bbox),
                    )
                )
            events = machine.process(detections)
            emitter.emit_many(events)
            emitted += len(events)
            frame_index += 1
    finally:
        cap.release()
    emitter.flush()
    return emitted


def run_simulation(
    video: VideoInput,
    layout_path: str,
    api_url: str | None,
    out: str | None,
    fps: float,
    truncate_out: bool = True,
) -> int:
    layouts = load_layout(layout_path)
    layout = layouts[video.store_id]
    machine = RetailStateMachine(layout=layout, fps=fps)
    emitter = EventEmitter(api_url=api_url, jsonl_path=out, truncate=truncate_out)
    emitted = 0
    for _, detection in simulate_detections(video.store_id, video.camera_id, video.start_time, fps=fps):
        events = machine.process([detection])
        emitter.emit_many(events)
        emitted += len(events)
    emitter.flush()
    return emitted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process retail CCTV clips into Store Intelligence events.")
    parser.add_argument("--layout", default="data/sample/store_layout.json")
    parser.add_argument("--manifest", help="JSON manifest with videos: store_id,camera_id,path,start_time")
    parser.add_argument("--store-id", default="STORE_BLR_002")
    parser.add_argument("--camera-id", default="CAM_ENTRY_01")
    parser.add_argument("--video", help="Single video path")
    parser.add_argument("--start-time", default="2026-03-03T14:00:00Z")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--out", default="data/generated_events.jsonl")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--detector", choices=["auto", "yolo", "motion"], default="auto")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO person detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument("--device", help="YOLO device, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config.")
    parser.add_argument("--sample-every", type=int, default=1, help="For motion detector, process every Nth frame.")
    parser.add_argument("--motion-min-area", type=int, default=900, help="Minimum moving blob area for motion detector.")
    parser.add_argument("--max-frames", type=int, help="Development smoke-test limit per video.")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--simulate", action="store_true", help="Run deterministic synthetic detections; no video/model needed.")
    args = parser.parse_args(argv)

    if args.manifest:
        videos = read_manifest(args.manifest)
    else:
        videos = [VideoInput(args.store_id, args.camera_id, args.video or "", parse_ts(args.start_time))]

    total = 0
    detector: Literal["yolo", "motion"] = "yolo"
    if args.detector == "motion":
        detector = "motion"
    elif args.detector == "auto":
        try:
            import ultralytics  # noqa: F401  # type: ignore
        except Exception:
            detector = "motion"

    for video_index, video in enumerate(videos):
        truncate_out = video_index == 0
        if args.simulate or not video.path:
            total += run_simulation(video, args.layout, args.api_url, args.out, args.fps, truncate_out=truncate_out)
        elif detector == "motion":
            total += run_motion_video(
                video,
                args.layout,
                args.api_url,
                args.out,
                args.fps,
                sample_every=max(1, args.sample_every),
                min_area=args.motion_min_area,
                max_frames=args.max_frames,
                truncate_out=truncate_out,
            )
        else:
            total += run_yolo_video(
                video,
                args.layout,
                args.api_url,
                args.out,
                args.model,
                args.fps,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                tracker=args.tracker,
                max_frames=args.max_frames,
                truncate_out=truncate_out,
            )
    print(f"emitted_events={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
