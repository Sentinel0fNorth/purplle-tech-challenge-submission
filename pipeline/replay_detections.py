from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .emit import EventEmitter
from .layout import load_layout
from .tracker import Detection, RetailStateMachine


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def replay_file(path: Path, layout_path: str, api_url: str | None, out: str | None, truncate: bool) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    layouts = load_layout(layout_path)
    layout = layouts[payload["store_id"]]
    fps = float(payload.get("fps") or 15.0)
    start = _parse_ts(payload.get("start_time", "2026-03-03T14:00:00Z"))
    machine = RetailStateMachine(layout=layout, fps=fps)
    emitter = EventEmitter(api_url=api_url, jsonl_path=out, truncate=truncate)
    emitted = 0
    for frame in payload.get("frames", []):
        frame_index = int(frame.get("i") or 0)
        timestamp = _parse_ts(frame["timestamp"]) if frame.get("timestamp") else start + timedelta(seconds=frame_index / fps)
        detections = [
            Detection(
                camera_id=payload["camera_id"],
                track_id=str(box[5]),
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                confidence=float(box[4]),
                frame_index=frame_index,
                timestamp=timestamp,
                embedding=tuple(float(v) for v in box[6]) if len(box) > 6 and box[6] is not None else None,
            )
            for box in frame.get("boxes", [])
        ]
        events = machine.process(detections)
        emitter.emit_many(events)
        emitted += len(events)
    if payload.get("frames"):
        last = payload["frames"][-1]
        last_index = int(last.get("i") or 0)
        last_ts = _parse_ts(last["timestamp"]) if last.get("timestamp") else start + timedelta(seconds=last_index / fps)
        stale = machine.flush_stale_queue(last_ts + timedelta(minutes=6))
        emitter.emit_many(stale)
        emitted += len(stale)
    emitter.flush()
    return emitted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay precomputed detections into structured retail events.")
    parser.add_argument("--detections", required=True, help="Detection JSON file or directory of *.detections.json files.")
    parser.add_argument("--layout", default="data/sample/store_layout.json")
    parser.add_argument("--api-url", default="")
    parser.add_argument("--out", default="data/replayed_events.jsonl")
    args = parser.parse_args(argv)

    source = Path(args.detections)
    files = sorted(source.glob("*.detections.json")) if source.is_dir() else [source]
    total = 0
    for index, file_path in enumerate(files):
        total += replay_file(file_path, args.layout, args.api_url or None, args.out, truncate=index == 0)
    print(f"emitted_events={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
