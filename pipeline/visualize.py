from __future__ import annotations

import argparse
from pathlib import Path

from .detect import read_manifest
from .layout import load_layout, zone_for_bbox

try:
    import cv2  # type: ignore
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV and NumPy are required for visualization. Run `pip install -r requirements.txt`.") from exc


ZONE_COLORS = {
    "ENTRY": (60, 220, 60),
    "BILLING": (60, 60, 240),
    "SKINCARE": (240, 170, 60),
    "MAKEUP": (220, 80, 220),
    "HAIRCARE": (60, 220, 220),
}


def _draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + baseline + 4, y)
    cv2.rectangle(frame, (x, y - th - baseline - 6), (x + tw + 8, y + 4), color, -1)
    cv2.putText(frame, text, (x + 4, y - baseline - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_zones(frame, layout, camera_id: str) -> None:
    overlay = frame.copy()
    for zone in layout.zones:
        if zone.camera_ids and camera_id not in zone.camera_ids:
            continue
        if not zone.polygon:
            continue
        pts = np.array(zone.polygon, dtype=np.int32)
        color = ZONE_COLORS.get(zone.zone_id, (200, 200, 80))
        cv2.polylines(frame, [pts], True, color, 2)
        cv2.fillPoly(overlay, [pts], color)
        x, y = pts[0]
        _draw_label(frame, zone.zone_id, int(x), int(y), color)
    cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)


def render_video(
    video_path: str,
    store_id: str,
    camera_id: str,
    layout_path: str,
    model_name: str,
    out_path: str,
    conf: float,
    imgsz: int,
    max_frames: int | None,
    show: bool,
) -> int:
    from ultralytics import YOLO  # type: ignore

    layouts = load_layout(layout_path)
    layout = layouts[store_id]
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    cap.release()

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    processed = 0
    results = model.track(
        source=video_path,
        stream=True,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )
    try:
        for frame_index, result in enumerate(results):
            if max_frames is not None and frame_index >= max_frames:
                break
            frame = result.orig_img.copy()
            _draw_zones(frame, layout, camera_id)

            boxes = getattr(result, "boxes", None)
            if boxes is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(xyxy)
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else list(range(len(xyxy)))
                for bbox, score, track_id in zip(xyxy, confs, ids):
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    bbox_tuple = tuple(float(v) for v in bbox)
                    zone = zone_for_bbox(layout, camera_id, bbox_tuple)
                    color = ZONE_COLORS.get(zone.zone_id if zone else "", (80, 180, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    foot = (int((x1 + x2) / 2), int(y2))
                    cv2.circle(frame, foot, 4, color, -1)
                    zone_name = zone.zone_id if zone else "NO_ZONE"
                    _draw_label(frame, f"id={track_id} {score:.2f} {zone_name}", x1, y1 - 6, color)

            writer.write(frame)
            processed += 1
            if show:
                cv2.imshow("store intelligence detections", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        writer.release()
        if show:
            cv2.destroyAllWindows()
    return processed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render YOLO detections, track IDs, foot points, and retail zones onto video.")
    parser.add_argument("--layout", default="data/real_store_layout.json")
    parser.add_argument("--manifest", default="data/video_manifest.json")
    parser.add_argument("--camera-id", help="Render only one camera from the manifest.")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--out-dir", default="data/annotated")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=300, help="Frame limit. Use 0 to render the full video.")
    parser.add_argument("--show", action="store_true", help="Open a live OpenCV preview window. Press q to close.")
    args = parser.parse_args(argv)

    videos = read_manifest(args.manifest)
    if args.camera_id:
        videos = [video for video in videos if video.camera_id == args.camera_id]
    if not videos:
        raise RuntimeError("No matching videos found in manifest.")

    max_frames = None if args.max_frames == 0 else args.max_frames
    for video in videos:
        out_path = Path(args.out_dir) / f"{video.store_id}_{video.camera_id}_annotated.mp4"
        frames = render_video(
            video_path=video.path,
            store_id=video.store_id,
            camera_id=video.camera_id,
            layout_path=args.layout,
            model_name=args.model,
            out_path=str(out_path),
            conf=args.conf,
            imgsz=args.imgsz,
            max_frames=max_frames,
            show=args.show,
        )
        print(f"rendered_frames={frames} output={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
