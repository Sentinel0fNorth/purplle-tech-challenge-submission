# Store Intelligence API + CCTV Event Pipeline

This repository is a unified, self-contained solution for the retail CCTV challenge. It consolidates the CV/ML detection pipeline and the production-ready FastAPI backend service into a single repository.

Raw CCTV video clips are processed into structured behavioral events, which are ingested into the consolidated backend API. Live metrics, Heatmaps, Funnels, and Anomalies can be retrieved via REST endpoints or viewed on the terminal dashboard.

---

## 5-Command Setup

```bash
# 1. Start the unified API and Database in Docker
docker compose up --build -d

# 2. Run the pipeline model layer on the Store 1 / Store 2 videos
python -m pipeline.detect --layout data/real_store_layout.json --manifest data/video_manifest.json --api-url "http://localhost:8000" --out data/real_events.jsonl --detector auto

# 3. Optional: Run a simulated/mock pipeline execution
python -m pipeline.detect --simulate --layout data/sample/store_layout.json --api-url "http://localhost:8000" --out data/generated_events.jsonl

# 4. Query live metrics with optional temporal filtering
curl "http://localhost:8000/stores/STORE_BLR_002/metrics?start=2026-03-03T00:00:00Z&end=2026-03-04T00:00:00Z"

# 5. Open the live terminal dashboard
python -m pipeline.dashboard --api-url http://localhost:8000 --store-id STORE_BLR_002
```

---

## Consolidated System Architecture

The project is structured into two main layers integrated into one repository:

### 1. Ingestion & Analytics Backend (`./app`)
The production-ready FastAPI service resides in `./app` and persists events and POS data to a SQLite database. It consists of:
- `main.py`: REST routes for event ingestion (`/ingest`) and store-level analytics.
- `metrics.py`: Computes conversion rate, abandonment rate, unique visitors, and average dwell time per zone with support for time-based filtering.
- `anomalies.py`: Flags conversion rate drops (against a 7-day conversion baseline), dead zones (inactive zones relative to active windows), and checkout queue depth spikes.
- `pos_loader.py`: Daily batch ingestion script matching transacted brands to visitors' SKU zone visits.
- `schema.sql` & `db.py`: Database connection pool, indexes, and schema definitions.

### 2. Detection & Re-ID Pipeline (`./pipeline`)
The tracking system maps bounding box coordinates to physical store locations:
- `detect.py`: Directs inference (using YOLOv8 + ByteTrack or fallback motion detection) and converts tracking vectors into challenge-compliant events.
- `tracker.py`: State machine tracking visitor entries/exits, zone transitions, queue joins/abandons, and Re-ID matching.
- `reid.py`: Multi-region appearance histogram extractor matching cross-camera visitors.
- `layout.py`: Calibrates spatial polygons (zones and `sku_subzones` for brand mapping) from image coordinates.
- `emit.py`: Buffers and posts events to `/ingest`.

---

## API Endpoints

All endpoints support optional `start` and `end` query parameters (ISO-8601 UTC format) for temporal analytics.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/ingest` | Batch ingest events, validate schemas, deduplicate by `event_id`, and record visitor sessions. |
| GET | `/stores/{store_id}/metrics` | Retrieve unique visitors, conversion rate, latest queue depth, and average dwell time. |
| GET | `/stores/{store_id}/funnel` | Retrieve Entry → Zone Visit → Billing Queue → Purchase funnel stages and drop-offs. |
| GET | `/stores/{store_id}/heatmap` | Retrieve zones list with visits count, dwell times, and normalized 0-100 scores. |
| GET | `/stores/{store_id}/anomalies` | Retrieve baseline-calibrated anomaly alerts for the active window. |
| GET | `/health` | Verify database health status. |

---

## Event Schema

The pipeline emits standard, fully validated JSON events to the `/ingest` endpoint:

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

---

## Brand Spatial Calibration

Layout mappings (`data/real_store_layout.json`) support dividing generic sections into brand-level sub-zones. For instance:
* `SKINCARE` is segmented into sub-zones matching POS brand name strings (`Good Vibes`, `Minimalist`, `Foxtale`, `DERMDOC`, `COSRX`, `Round Lab`).
* `MAKEUP` is segmented into brand sub-zones (`Alps Goodness`, `Swiss Beauty`, `Lakme`, `Faces Canada`, `Maybelline`, `NY Bae`).

This calibration allows `/stores/{store_id}/metrics` to directly correlate visitors' spatial SKU visits to brand-specific transactions in the POS log.

---

## Tests and Coverage

Test suites run inside the consolidated container sandbox to ensure full compatibility.

To run tests with code coverage metrics:
```bash
docker compose run --rm api pytest
```

The test coverage exceeds the **70% gate** (achieving **71.33%** coverage across the codebase) and covers:
- Partial success and validation rejection.
- Time-window metric isolation and conversion rates.
- Funnel session-deduplication.
- Anomaly alerts.
- Live console dashboard rendering.
