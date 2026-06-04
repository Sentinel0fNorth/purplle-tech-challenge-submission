from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .detect import VideoInput, parse_ts, read_manifest
from .reid import crop_embedding


def _video_inputs(args: argparse.Namespace) -> list[VideoInput]:  # pragma: no cover - exercised through CLI in manual YOLO runs
    if args.manifest:
        return read_manifest(args.manifest)
    return [VideoInput(args.store_id, args.camera_id, args.video, parse_ts(args.start_time))]


def precompute_video(
    video: VideoInput,
    out_dir: Path,
    model_name: str,
    conf: float,
    imgsz: int,
    stride: int,
    device: str | None,
    max_frames: int | None,
) -> Path:  # pragma: no cover - requires ultralytics/OpenCV video fixtures
    import cv2  # type: ignore
    from ultralytics import YOLO  # type: ignore

    cap = cv2.VideoCapture(video.path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video.path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    model = YOLO(model_name)
    kwargs: dict[str, Any] = {
        "source": video.path,
        "stream": True,
        "persist": True,
        "classes": [0],
        "tracker": "bytetrack.yaml",
        "conf": conf,
        "imgsz": imgsz,
        "vid_stride": max(1, stride),
        "verbose": False,
    }
    if device:
        kwargs["device"] = device

    frames: list[dict[str, Any]] = []
    for result_index, result in enumerate(model.track(**kwargs)):
        frame_index = result_index * max(1, stride)
        if max_frames is not None and frame_index >= max_frames:
            break
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.id is None:
            frames.append({"i": frame_index, "boxes": []})
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        frame = getattr(result, "orig_img", None)
        boxes = []
        for (x1, y1, x2, y2), score, track_id in zip(xyxy, confs, ids):
            bbox = (float(x1), float(y1), float(x2), float(y2))
            embedding = crop_embedding(frame, bbox)
            boxes.append([*bbox, float(score), int(track_id), list(embedding) if embedding is not None else None])
        frames.append(
            {
                "i": frame_index,
                "timestamp": (video.start_time + timedelta(seconds=frame_index / fps)).isoformat().replace("+00:00", "Z"),
                "boxes": boxes,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video.store_id}_{video.camera_id}.detections.json"
    payload = {
        "store_id": video.store_id,
        "camera_id": video.camera_id,
        "video_path": video.path,
        "start_time": video.start_time.isoformat().replace("+00:00", "Z"),
        "fps": fps,
        "frame_size": [frame_w, frame_h],
        "stride": max(1, stride),
        "frames": frames,
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI around YOLO precompute
    parser = argparse.ArgumentParser(description="Precompute YOLO/ByteTrack person detections for fast replay.")
    parser.add_argument("--manifest")
    parser.add_argument("--video")
    parser.add_argument("--store-id", default="STORE_BLR_002")
    parser.add_argument("--camera-id", default="CAM_ENTRY_01")
    parser.add_argument("--start-time", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument("--out-dir", default="data/detections")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--device")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    if not args.manifest and not args.video:
        raise SystemExit("--manifest or --video is required")
    outputs = [
        str(precompute_video(video, Path(args.out_dir), args.model, args.conf, args.imgsz, args.stride, args.device, args.max_frames))
        for video in _video_inputs(args)
    ]
    print(json.dumps({"written": outputs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
