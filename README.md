# Store Intelligence API + CCTV Event Pipeline

This repository contains a unified, self-contained solution for the retail CCTV challenge. It couples an ML-based CV tracking pipeline with a production-ready FastAPI backend and a live terminal dashboard to measure real-time store metrics.

The system's "North Star" is the **Offline Store Conversion Rate** (Total Purchases ÷ Total Unique Visitors).

---

## Core Features
* **CCTV Event Pipeline:** Processes raw video clips using YOLOv8 + ByteTrack. Resolves diagonal entrance thresholds, filters staff uniforms in HSV space, splits side-by-side shopper groups, and handles frame freezes using appearance embeddings.
* **Ingestion Backend:** High-concurrency FastAPI service persisting events to a structured SQLite database running in WAL mode.
* **POS Matcher:** Correlates spatial SKU-zone visits with transactional POS CSV records within a +/- 5-minute checkout window, aligning IST to UTC automatically.
* **Terminal Live Dashboard:** An interactive live terminal UI plotting live metrics, conversion rates, heatmaps, and anomalies.

---

## 5-Command Setup

Verify the system on your machine using these 5 commands:

```bash
# 1. Build and boot the FastAPI backend + SQLite database in Docker
docker compose up --build -d

# 2. Align POS transaction dates and map store IDs to match video timelines
docker compose run --rm api python scripts/align_pos.py

# 3. Run the tracking pipeline on the target CCTV clips (limited to 1500 frames)
docker compose run --rm api python -m pipeline.detect --layout data/real_store_layout.json --manifest data/video_manifest_docker.json --api-url "http://api:8000" --out data/real_events.jsonl --detector auto --max-frames 1500

# 4. Run end-to-end POS conversion verification (simulates a queue join in the 5-minute transaction window)
docker compose run --rm api python scripts/test_pos_matching.py

# 5. Open the real-time terminal dashboard
docker compose run --rm api python -m pipeline.dashboard --api-url http://api:8000 --store-id STORE_1
```

---

## System Configuration & Testing

### Environment Variables
The backend service maps the following environment variables:
* `STORE_INTEL_DB`: Path to persistent SQLite database (default: `/data/store_intelligence.db`).
* `POS_CSV_PATH`: Path to aligned POS CSV transactions (default: `/app/data/pos_transactions_docker.csv`).

### Data Manifest Mapping
* Video files must be mounted under `/workspace/Original` (configured via volumes in `docker-compose.yml`).
* The manifest `data/video_manifest_docker.json` defines container-resolved paths:
  * `STORE_1` Main Floor: `/workspace/Original/Store 1-20260602T101818Z-3-001ec38db8/Store 1/CAM 2 - zone.mp4`
  * `STORE_2` Entry 2: `/workspace/Original/Store 2-20260602T101819Z-3-001099f208/Store 2/entry 2.mp4`
  * `STORE_2` Billing: `/workspace/Original/Store 2-20260602T101819Z-3-001099f208/Store 2/billing_area.mp4`

---

## REST API Reference

All analytical endpoints support start/end ISO-8601 query parameters for custom time-window filtering.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/events/ingest` | Bulk ingest JSON events. Validates schemas, ignores duplicates, and updates visitor session timelines. |
| `GET` | `/stores/{id}/metrics` | Returns unique visitors (excluding staff), average zone dwell times, conversion rates (POS matched), latest queue depth, and abandonment rate. |
| `GET` | `/stores/{id}/funnel` | Returns Entry → Zone → Billing Queue → Purchase session funnel conversion drops. |
| `GET` | `/stores/{id}/heatmap` | Aggregates visits count, average dwells, and normalized scores. Returns `data_confidence = false` if sessions < 20. |
| `GET` | `/stores/{id}/anomalies` | Returns active alarms: conversion drops, queue congestion spikes, and dead zones (inactive zones for 30+ mins). |
| `GET` | `/health` | Returns API status and triggers `STALE_FEED` warning if data lag exceeds 10 minutes. |

---

## Running Automated Tests

Our test coverage exceeds the mandatory **70% acceptance gate** (currently **72.75%**).

Run the full pytest suite inside the container sandbox:
```bash
docker compose run --rm api pytest
```
Tests cover mock frame generation, staff uniform HSV classification, wide aspect-ratio bounding box group splitting, spatio-temporal Re-ID matching, and database index operations.
