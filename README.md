# Store Intelligence API + CCTV Event Pipeline

This repository is a complete, deployable solution for the retail CCTV challenge: raw clips are converted into structured behavioral events, events are ingested into a production-aware REST API, and live metrics can be viewed through API endpoints or a terminal dashboard.

## 5-command setup

```bash
# 1. clone / unzip the repo
cd store-intelligence

# 2. start the API
docker compose up --build

# 3. in another terminal, install the CV/ML dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. run the model layer on the supplied Store 1 / Store 2 videos
python -m pipeline.detect --layout data/real_store_layout.json --manifest data/video_manifest.json --api-url "" --out data/real_events.jsonl --detector auto

# Optional deterministic smoke test
python -m pipeline.detect --simulate --layout data/sample/store_layout.json --api-url "" --out data/generated_events.jsonl

# 5. query metrics
curl http://localhost:8000/stores/STORE_BLR_002/metrics

# 6. open the live terminal dashboard
python -m pipeline.dashboard --api-url http://localhost:8000 --store-id STORE_BLR_002
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/events/ingest` | Ingest up to 500 events, validate rows, deduplicate by `event_id`, and allow partial success. |
| GET | `/stores/{store_id}/metrics` | Unique visitors, conversion rate, dwell by zone, queue depth, abandonment rate. |
| GET | `/stores/{store_id}/funnel` | Entry → Zone Visit → Billing Queue → Purchase session funnel. |
| GET | `/stores/{store_id}/heatmap` | Zone frequency, average dwell, normalized 0-100 heatmap score. |
| GET | `/stores/{store_id}/anomalies` | Queue spike, conversion drop, and dead-zone anomalies. |
| GET | `/health` | DB status, last event per store, and stale-feed warnings. |

Optional `start` and `end` query parameters accept ISO-8601 UTC timestamps.

## Event schema

The detection layer emits challenge-compliant JSON events:

```json
{
  "event_id": "71a8f9c2-b5e4-4d9a-9b1c-3f2d1e4a5b6c",
  "store_id": "STORE_BLR_002",
  "camera_id": "CAM_ENTRY_01",
  "visitor_id": "VIS_c8a2f1",
  "event_type": "ZONE_DWELL",
  "timestamp": "2026-03-03T14:22:10Z",
  "zone_id": "SKINCARE",
  "dwell_ms": 30000,
  "is_staff": false,
  "confidence": 0.82,
  "metadata": {
    "queue_depth": null,
    "sku_zone": "MOISTURISER",
    "session_seq": 5
  }
}
```

### Event contract checklist

- `event_id` is generated as a unique UUID v4 by the pipeline.
- `event_type` is one of the exact uppercase challenge values: `ENTRY`, `EXIT`, `REENTRY`, `ZONE_ENTER`, `ZONE_EXIT`, `ZONE_DWELL`, `BILLING_QUEUE_JOIN`, `BILLING_QUEUE_ABANDON`.
- `timestamp` is emitted as a timezone-aware UTC ISO-8601 string with `Z`.
- `store_id` comes from the manifest/layout and must match the `store_id` used in the POS CSV for conversion correlation.
- `visitor_id` is stable across cameras when the cross-camera Re-ID match passes appearance, time, and camera-transition gates.
- `REENTRY` reuses the original `visitor_id` after a confirmed exit and return.
- `metadata.queue_depth` is an integer on billing queue join events.
- `is_staff` is set from the staff classifier and staff events are filtered out of customer metrics.
- `metadata.sku_zone` uses exact configured SKU/brand labels. Layouts can define `sku_subzones` with brand-level polygons, for example `Maybelline`, `Faces Canada`, and `Renee`, so POS `brand_name` values can be matched directly when the store calibration provides those labels.

## Running against the provided Store 1 / Store 2 videos

The checked-in `data/video_manifest.json` points at the video folders placed at the workspace root:

- `../Store 1-20260602T101818Z-3-001ec38db8/Store 1`
- `../Store 2-20260602T101819Z-3-001099f208/Store 2`

`data/real_store_layout.json` contains starter zone polygons and camera roles for those clips. Treat polygons as calibration data: after inspecting detections, adjust them to the actual shelf zones, entry threshold, and billing queue area.

Run only the CV/ML layer and write events to JSONL:

```bash
python -m pipeline.detect \
  --layout data/real_store_layout.json \
  --manifest data/video_manifest.json \
  --api-url "" \
  --out data/real_events.jsonl \
  --detector auto
```

For quick calibration loops, add `--max-frames 300` to process only the first few seconds of each clip.

To visually inspect what the model is seeing, render an annotated video with YOLO boxes, track IDs, foot points, and configured retail zones:

```bash
python -m pipeline.visualize \
  --layout data/real_store_layout.json \
  --manifest data/video_manifest.json \
  --camera-id STORE1_CAM3_ENTRY \
  --model yolov8n.pt \
  --out-dir data/annotated \
  --max-frames 300 \
  --imgsz 640
```

The rendered MP4 is written to `data/annotated`. Add `--show` if you want an OpenCV preview window while it renders.

For faster repeated demos on CPU, precompute YOLO/ByteTrack detections once and replay them through the same event state machine:

```bash
python -m pipeline.precompute_detections \
  --manifest data/video_manifest.json \
  --out-dir data/detections \
  --stride 5 \
  --model yolov8n.pt

python -m pipeline.replay_detections \
  --detections data/detections \
  --layout data/real_store_layout.json \
  --api-url http://localhost:8000 \
  --out data/replayed_events.jsonl
```

The replay path is useful for dashboards and judging walkthroughs because it avoids rerunning neural inference while preserving the normal zone, queue, dwell, and entry/exit event logic.

Run the same model layer while streaming events into the API:

```bash
python -m pipeline.detect \
  --layout data/real_store_layout.json \
  --manifest data/video_manifest.json \
  --api-url http://localhost:8000 \
  --out data/real_events.jsonl \
  --detector auto
```

`--detector auto` uses Ultralytics YOLO + ByteTrack when `ultralytics` is installed. If YOLO is unavailable, it falls back to an OpenCV motion detector so the full event path can still run locally on the supplied clips. For final scoring, prefer YOLO:

```bash
python -m pipeline.detect \
  --layout data/real_store_layout.json \
  --manifest data/video_manifest.json \
  --api-url http://localhost:8000 \
  --out data/real_events.jsonl \
  --detector yolo \
  --model yolov8n.pt \
  --conf 0.25 \
  --imgsz 960
```

If you receive different challenge videos, copy `data/video_manifest.example.json` to `data/video_manifest.json`, replace each `url`, `path`, and `start_time`, then run:

```bash
python -m pipeline.download_videos data/video_manifest.json
python -m pipeline.detect \
  --layout data/real_store_layout.json \
  --manifest data/video_manifest.json \
  --api-url http://localhost:8000 \
  --out data/generated_events.jsonl \
  --detector yolo \
  --model yolov8n.pt
```

If the challenge provides a richer `store_layout.json`, replace `data/sample/store_layout.json` with it or pass its path through `--layout`.

## Detection notes

The real-video path uses Ultralytics YOLO tracking with ByteTrack. If `ultralytics` is unavailable, `--detector auto` falls back to a motion-segmentation baseline; if video links have not arrived yet, `--simulate` runs a deterministic synthetic path that exercises entry, zone dwell, billing queue, and exit logic. The event generator itself is independent of the detector, which makes it safe to swap YOLOv8 for YOLOv11, RT-DETR, or another person detector.

Staff detection currently uses a configurable torso-color heuristic from `store_layout.json`. It is isolated in `pipeline/staff.py` so it can be replaced with a FashionCLIP/VLM classifier after the actual uniforms are visible in the clips.

Entry and exit events use a confirmed virtual line crossing rather than a single-frame threshold, which reduces jitter around door boundaries. Billing abandon events are delayed until the queue grace window elapses, so a shopper briefly leaving the billing polygon is not immediately counted as abandoned. Event metadata also includes `group_size` when available from the active camera frame.

Cross-camera Re-ID is implemented in the tracker. Each detection carries a multi-region crop appearance embedding built from full-body, upper/mid/lower clothing histograms, dominant color, texture edges, and box-shape cues. New camera-local tracks are matched against recent visitor profiles using appearance similarity, camera transition hints, and time decay; weak appearance matches are rejected even if the camera sequence is plausible. Cached detections produced by `pipeline.precompute_detections` also store embeddings, so replayed demos use the same Re-ID path as live YOLO runs.

For deployments that can install a stronger Re-ID model, `pipeline.reid.set_deep_embedding_provider(...)` or the `STORE_INTEL_REID_PROVIDER=module:function` environment variable can append OSNet/FastReID-style embeddings to the handcrafted descriptor without changing the rest of the event pipeline.

## Tests and coverage

```bash
pytest
```

The tests cover ingestion idempotency, partial success, metrics, POS conversion, zero-purchase stores, re-entry funnel deduplication, anomalies, and the empty-store pipeline edge case. Prompt blocks are included at the top of each test file as required by the challenge.

## Production readiness

- `docker compose up` starts the API.
- SQLite persistence is used by default and mounted in the `store_intel_data` volume.
- Every request emits a structured JSON log with trace ID, endpoint, latency, status code, and ingest event count when applicable.
- Duplicate event batches are safe to replay because `event_id` is the database primary key.
- Database errors return structured HTTP 503 responses, not raw stack traces.
- `/health` reports stale feeds when a store has not emitted events in more than 10 minutes.

## Documentation

- `docs/DESIGN.md` explains the architecture and AI-assisted decisions.
- `docs/CHOICES.md` covers detection model choice, event schema rationale, and API architecture rationale.
