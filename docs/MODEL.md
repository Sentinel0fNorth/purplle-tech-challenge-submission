# MODEL.md - Complete Retail CCTV Model Layer

## Purpose

The model layer converts raw retail CCTV footage into structured store-intelligence events. It is designed for the Purplle retail challenge requirements: count real shoppers, track movement across cameras, identify dwell and queue behavior, exclude staff, preserve confidence, and emit a challenge-compliant event stream that the API can ingest and score.

The model layer is intentionally separate from the API. Video inference can run as a batch process, a near-real-time worker, or a precompute/replay workflow, while the FastAPI service only consumes validated JSON events.

## End-to-End Flow

```text
CCTV mp4 files
  -> YOLO person detector
  -> ByteTrack same-camera tracking
  -> crop appearance embedding
  -> cross-camera Re-ID visitor matching
  -> zone and SKU polygon mapping
  -> retail event state machine
  -> JSONL and/or POST /events/ingest
  -> metrics, funnel, heatmap, anomalies
```

The main entry points are:

- `pipeline.detect`: runs detection/tracking/event generation directly from videos.
- `pipeline.precompute_detections`: runs YOLO/ByteTrack once and stores cached detections plus embeddings.
- `pipeline.replay_detections`: replays cached detections through the same state machine.
- `pipeline.visualize`: renders annotated videos with boxes, track IDs, foot points, and retail zones.

## Inputs

### Video Manifest

`data/video_manifest.json` defines the clips:

```json
{
  "videos": [
    {
      "store_id": "STORE_1",
      "camera_id": "STORE1_CAM1_ZONE",
      "path": "../Store 1-.../Store 1/CAM 1 - zone.mp4",
      "start_time": "2026-06-02T10:00:00Z"
    }
  ]
}
```

The `store_id` must match the store layout and should match the POS export when conversion correlation is needed.

### Store Layout

`data/real_store_layout.json` and `data/sample/store_layout.json` define:

- store IDs
- camera IDs and roles
- staff uniform reference colors
- zone polygons
- exact SKU or brand sub-zone polygons

Zone matching uses the bottom-center foot point of the person box, not the box center. This is important in CCTV footage because a shopper's torso may visually overlap a shelf while their feet are still in the aisle or another zone.

Example SKU sub-zone:

```json
{
  "zone_id": "MAKEUP",
  "sku_zone": "Maybelline",
  "camera_ids": ["CAM_FLOOR_01"],
  "polygon": [[720, 320], [1120, 320], [1170, 820], [680, 820]],
  "sku_subzones": [
    {"sku_zone": "Maybelline", "polygon": [[680, 320], [850, 320], [860, 820], [680, 820]]},
    {"sku_zone": "Faces Canada", "polygon": [[850, 320], [1010, 320], [1020, 820], [860, 820]]},
    {"sku_zone": "Renee", "polygon": [[1010, 320], [1170, 320], [1170, 820], [1020, 820]]}
  ]
}
```

When a person lands inside a SKU sub-zone, `metadata.sku_zone` uses the exact brand string. This is meant to align directly with POS `brand_name` values.

## Detection

The real-video path uses Ultralytics YOLO:

```python
YOLO(model_name).track(
    source=video.path,
    stream=True,
    persist=True,
    classes=[0],
    tracker="bytetrack.yaml",
    conf=0.25,
    imgsz=960
)
```

Only class `0` is used, which is the person class in COCO-style YOLO models. The default checked-in model is `yolov8n.pt`, chosen because it is fast enough for CPU machines while still detecting people reliably in the challenge clips.

Important knobs:

- `--model`: detector weights, default `yolov8n.pt`
- `--conf`: person detection confidence threshold, default `0.25`
- `--imgsz`: inference image size, default `960`
- `--device`: optional device like `cpu`, `0`, or `cuda:0`
- `--tracker`: Ultralytics tracker config, default `bytetrack.yaml`
- `--max-frames`: optional frame cap for calibration runs

## Same-Camera Tracking

Same-camera identity is handled by ByteTrack through Ultralytics. ByteTrack keeps a persistent local `track_id` inside each camera clip. The state machine receives detections with:

- `camera_id`
- `track_id`
- `bbox`
- `confidence`
- `frame_index`
- `timestamp`
- `is_staff`
- `embedding`

Camera-local tracks are not assumed to be globally valid. A local key such as `STORE1_CAM1_ZONE:7` is only used as the first identity clue. Cross-camera visitor identity is handled separately.

## Cross-Camera Multi-Target Re-ID

Cross-camera Re-ID is implemented in `pipeline.reid` and `pipeline.tracker`.

### Appearance Descriptor

For each person crop, `crop_embedding(frame, bbox)` builds a normalized embedding using:

- full-body HSV color histogram
- upper-body clothing histogram
- torso clothing histogram
- lower-body clothing histogram
- dominant color statistics
- edge and texture orientation descriptor
- normalized bounding-box shape and position cues

This is lightweight and dependency-friendly. It does not require a heavy Re-ID model, so the project remains runnable on reviewer machines.

### Optional Deep Re-ID Hook

The handcrafted descriptor can be extended with a stronger OSNet/FastReID-style embedding provider:

```python
from pipeline.reid import set_deep_embedding_provider

set_deep_embedding_provider(my_reid_provider)
```

Or through an environment variable:

```bash
STORE_INTEL_REID_PROVIDER=module:function
```

The provider must accept `(frame, bbox)` and return a numeric embedding. The returned vector is normalized and concatenated with the handcrafted descriptor.

### Matching Logic

When a new camera-local track appears, `RetailStateMachine` tries to match it against recent visitor profiles.

The match must pass:

- minimum appearance similarity: `reid_min_appearance`, default `0.78`
- total Re-ID score threshold: `reid_threshold`, default `0.82`
- time window: `reid_time_window_seconds`, default `900`
- same-frame uniqueness: one visitor cannot be claimed by multiple new tracks in the same frame
- same-camera guard: a visitor is not re-matched to a second active track in the same camera

The final score is:

```text
appearance cosine similarity
  + camera transition bonus
  + time proximity bonus or penalty
```

Transition hints include:

- entry -> floor/zone: positive
- floor/zone -> billing: positive
- billing -> entry: positive
- same camera: negative

If a match passes, the same `visitor_id` is reused across cameras. If no match passes, a new `visitor_id` is created. This avoids forcing bad matches in visually ambiguous moments.

### Metadata

Events include Re-ID observability fields:

```json
{
  "metadata": {
    "reid_matches": 1,
    "reid_score": 0.91
  }
}
```

This makes it possible to audit when a visitor ID crossed camera boundaries.

## Staff Filtering

Staff classification is implemented in `pipeline.staff.StaffClassifier`.

The current production-safe implementation is a uniform color heuristic:

- crop the person's torso region
- compare dominant torso colors to configured `staff_uniform_rgb`
- return `(is_staff, staff_confidence)`

If `is_staff` is true:

- the event still gets emitted
- `is_staff` is set to `true`
- customer metrics exclude that visitor
- queue depth excludes staff

This keeps the data visible for debugging while preventing staff from inflating visitor, queue, dwell, or conversion metrics.

## Zone and SKU Mapping

Zone detection happens in `pipeline.layout.zone_for_bbox`. The model checks the foot point of the box against configured zone polygons.

SKU/brand detection happens in `pipeline.layout.sku_zone_for_bbox`. It first checks `sku_subzones`, then falls back to the broader zone's `sku_zone`.

This supports two levels of analytics:

- broad retail zone heatmap, such as `MAKEUP`, `SKINCARE`, `BILLING`
- exact POS brand matching, such as `Maybelline`, `Faces Canada`, `Renee`

The exact brand strings must be calibrated to match POS `brand_name` values.

## Event State Machine

The core business logic is in `pipeline.tracker.RetailStateMachine`.

It converts frame-level detections into challenge events:

- `ENTRY`
- `EXIT`
- `REENTRY`
- `ZONE_ENTER`
- `ZONE_EXIT`
- `ZONE_DWELL`
- `BILLING_QUEUE_JOIN`
- `BILLING_QUEUE_ABANDON`

### Entry and Exit

Entry/exit uses a virtual door line with confirmation frames.

Default:

- `line_confirm_frames=3`

A person must be on the inside or outside side of the line for enough consecutive frames before an entry or exit is emitted. This reduces boundary jitter.

If a shopper exits and later crosses back inside with the same visitor identity, the model emits `REENTRY` and reuses the original `visitor_id`.

### Zone Enter and Exit

When the person's foot point enters a new zone:

- close the previous zone with `ZONE_EXIT`
- emit `ZONE_ENTER` for the new zone
- store `zone_entered_at`
- store the active SKU/brand label

When the person leaves all zones, the previous zone is closed with `ZONE_EXIT`.

### Dwell

The model emits `ZONE_DWELL` every configured dwell interval while a person stays in the same zone.

Default:

- `dwell_seconds=30`

The dwell event uses the elapsed time since zone entry.

### Queue Join

Queue depth is computed from active, non-staff visitors in the `BILLING` zone.

When a non-staff visitor first enters `BILLING`, the model emits:

```json
{
  "event_type": "BILLING_QUEUE_JOIN",
  "zone_id": "BILLING",
  "metadata": {
    "queue_depth": 2
  }
}
```

`metadata.queue_depth` is always an integer when the join event is emitted.

### Queue Abandon

Queue abandonment is delayed instead of emitted immediately when a visitor leaves billing.

Default:

- `queue_abandon_seconds=300`

If the visitor left billing and remains unresolved for the grace period, `flush_stale_queue` emits `BILLING_QUEUE_ABANDON`.

This avoids falsely marking someone as abandoned when they briefly leave the polygon or when POS correlation later shows they purchased.

## Event Schema Contract

The pipeline emits the problem-statement schema:

```json
{
  "event_id": "uuid-v4",
  "store_id": "STORE_1",
  "camera_id": "STORE1_CAM1_ZONE",
  "visitor_id": "VIS_ab12cd34",
  "event_type": "ZONE_ENTER",
  "timestamp": "2026-06-02T10:00:00Z",
  "zone_id": "MAKEUP",
  "dwell_ms": 0,
  "is_staff": false,
  "confidence": 0.91,
  "metadata": {
    "queue_depth": null,
    "sku_zone": "Faces Canada",
    "session_seq": 2,
    "group_size": 3,
    "reid_matches": 1,
    "reid_score": 0.89
  }
}
```

Contract guarantees:

- `event_id` is UUID v4.
- `event_type` uses exact uppercase challenge values.
- `timestamp` is UTC ISO-8601 with `Z`.
- `store_id` comes from the manifest/layout.
- `visitor_id` is stable across cameras when Re-ID passes.
- `REENTRY` reuses the existing visitor ID after a confirmed exit and return.
- `metadata.queue_depth` is an integer on billing queue join events.
- `metadata.sku_zone` can be exact brand text when brand polygons are configured.
- `is_staff` is propagated from the staff classifier.

## Running the Model

### Full Real-Video Detection

```bash
python -m pipeline.detect \
  --layout data/real_store_layout.json \
  --manifest data/video_manifest.json \
  --api-url= \
  --out data/real_events.jsonl \
  --detector yolo \
  --model yolov8n.pt \
  --conf 0.25 \
  --imgsz 960
```

Use `--api-url http://localhost:8000` to stream events into the API.

### Single Camera

```bash
python -m pipeline.detect \
  --layout data/real_store_layout.json \
  --store-id STORE_1 \
  --camera-id STORE1_CAM1_ZONE \
  --video "../Store 1-20260602T101818Z-3-001ec38db8/Store 1/CAM 1 - zone.mp4" \
  --start-time 2026-06-02T10:00:00Z \
  --api-url= \
  --out data/store1_cam1_zone_full_events.jsonl \
  --detector yolo \
  --model yolov8n.pt \
  --conf 0.25 \
  --imgsz 960
```

### Simulation Mode

```bash
python -m pipeline.detect \
  --simulate \
  --layout data/sample/store_layout.json \
  --api-url= \
  --out data/generated_events.jsonl
```

Simulation is useful for CI and endpoint checks when real videos are not available.

### Precompute and Replay

Precompute detections once:

```bash
python -m pipeline.precompute_detections \
  --manifest data/video_manifest.json \
  --out-dir data/detections \
  --stride 5 \
  --model yolov8n.pt
```

Replay cached detections:

```bash
python -m pipeline.replay_detections \
  --detections data/detections \
  --layout data/real_store_layout.json \
  --api-url http://localhost:8000 \
  --out data/replayed_events.jsonl
```

Cached detections include embeddings, so replay uses the same cross-camera Re-ID logic as live YOLO inference.

### Visualizer

```bash
python -m pipeline.visualize \
  --layout data/real_store_layout.json \
  --manifest data/video_manifest.json \
  --camera-id STORE1_CAM1_ZONE \
  --model yolov8n.pt \
  --out-dir data/annotated \
  --conf 0.25 \
  --imgsz 960 \
  --max-frames 0
```

The visualizer draws:

- zone polygons
- YOLO boxes
- ByteTrack IDs
- foot points
- inferred zone labels

## Model Outputs

The model can output:

- JSONL events
- API ingestion batches
- cached detection JSON files
- annotated videos

JSONL files are the primary audit artifact. They let a reviewer inspect exact events without running the API.

## API Integration

The API ingests model events through:

```http
POST /events/ingest
```

Payload shape:

```json
{
  "events": [
    {"event_id": "..."}
  ]
}
```

The API validates, deduplicates by `event_id`, stores metadata, and exposes:

- `/stores/{store_id}/metrics`
- `/stores/{store_id}/funnel`
- `/stores/{store_id}/heatmap`
- `/stores/{store_id}/anomalies`
- `/health`

## Validation and Tests

Run:

```bash
pytest -q
```

The test suite covers:

- event schema validation
- Resource Center sample-event adaptation
- raw POS CSV normalization
- metric and funnel behavior
- heatmap and anomaly behavior
- empty-store safety
- dwell event timing
- confirmed entry line crossing
- delayed billing abandon
- cross-camera multi-target Re-ID
- stale Re-ID rejection
- weak-appearance Re-ID rejection
- exact SKU sub-zone mapping
- REENTRY reuse
- staff flag propagation
- cached detection replay

Current verified status from the latest run:

- 25 tests passed
- coverage gate passed above 70 percent

## Calibration Procedure

For each new store or camera set:

1. Run visualizer on a short clip:

```bash
python -m pipeline.visualize --camera-id CAMERA_ID --max-frames 300
```

2. Inspect whether foot points land in correct zones.

3. Adjust `store_layout.json` polygons.

4. Add brand-level `sku_subzones` for POS matching.

5. Run detection for a short frame limit:

```bash
python -m pipeline.detect --max-frames 300 ...
```

6. Inspect emitted JSONL:

- visitor IDs
- zone transitions
- queue depth
- staff flag
- SKU labels
- Re-ID scores

7. Tune:

- `--conf`
- `--imgsz`
- `reid_threshold`
- `reid_min_appearance`
- `reid_time_window_seconds`
- staff uniform RGB values
- zone and SKU polygons

8. Run full video after calibration.

## Known Limitations

The current Re-ID descriptor is strong for a dependency-light implementation but is not equivalent to a trained person Re-ID network. It can still struggle when:

- two shoppers wear very similar colors
- heavy occlusion hides clothing
- cameras have very different lighting or color temperature
- a person changes appearance by carrying/removing a bag or jacket
- camera timestamps are badly misaligned

The system handles uncertainty conservatively. If Re-ID confidence is not high enough, it creates a new visitor rather than forcing a bad match.

The SKU-zone mapping quality depends on polygon calibration. Exact brand matching works only when the layout includes brand-level polygons whose strings match POS brand names.

Staff classification is currently a uniform-color heuristic. It is reliable when staff uniforms are visually consistent, but it should be upgraded to a classifier if uniforms vary heavily or staff wear casual layers.

## Upgrade Path

High-impact future upgrades:

- Add OSNet/FastReID embeddings through `STORE_INTEL_REID_PROVIDER`.
- Add camera timestamp synchronization if clips are not aligned.
- Learn camera transition priors from historical tracks.
- Add homography calibration for better cross-camera spatial reasoning.
- Add face-hidden/group-size modeling if required by future scoring.
- Replace staff heuristic with FashionCLIP or a small uniform classifier.
- Materialize per-store/day metrics if scaling beyond local challenge volume.

The architecture is prepared for these upgrades because detection, Re-ID, event generation, ingestion, and analytics are separated behind small interfaces.
