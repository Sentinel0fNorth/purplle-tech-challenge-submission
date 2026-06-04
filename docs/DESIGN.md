# DESIGN.md — Store Intelligence System

## Plain-language architecture overview

This repository implements the full path from retail CCTV clips to queryable store intelligence. The system is split into two deployable pieces: a detection/event-production pipeline and a FastAPI Intelligence API. The split is intentional. Video processing is CPU/GPU-heavy and may run as a batch job or near-real-time worker, while the API must remain responsive to dashboards and scoring tests. The pipeline emits structured JSON events, and the API ingests those events through `POST /events/ingest`.

The API is started by `docker compose up` and stores data in SQLite by default. SQLite is used because the challenge scoring harness is local, deterministic, and easy to reset. The storage layer is isolated in `app/database.py`, so replacing SQLite with PostgreSQL mostly means rewriting connection setup and SQL dialect differences, not changing endpoint behavior. Each event is keyed by `event_id`, making ingestion idempotent. Replaying the same batch returns a deduplicated count and does not double-count metrics.

The event schema follows the challenge catalogue: `ENTRY`, `EXIT`, `REENTRY`, `ZONE_ENTER`, `ZONE_EXIT`, `ZONE_DWELL`, `BILLING_QUEUE_JOIN`, and `BILLING_QUEUE_ABANDON`. Timestamps are stored in UTC ISO-8601 format. Staff events are stored but excluded from customer metrics. The API computes unique visitors, conversion rate, dwell by zone, queue depth, abandonment rate, funnel stages, heatmap cells, anomalies, and health status. POS conversion is estimated by the required time-window method: a visitor present in the billing zone during the 5 minutes before a POS transaction is treated as converted.

## Pipeline overview

`pipeline.detect` supports two modes. Simulation mode is deterministic and is used for acceptance demos and CI when no challenge videos are available. Real video mode uses Ultralytics YOLO tracking with ByteTrack (`tracker="bytetrack.yaml"`) when `ultralytics` is installed. The model outputs person detections with track IDs; `pipeline.tracker.RetailStateMachine` converts those detections into business events. This state machine is model-agnostic, so a future RT-DETR, YOLOv11, DeepSORT, or OSNet Re-ID component can feed the same class.

Spatial mapping happens through `store_layout.json`. The bottom-center foot point of the bounding box is tested against each configured zone polygon. This avoids the common mistake of using the bounding-box center, which can put a shopper's torso in one polygon while their feet are still in another. Queue depth is computed from active non-staff visitors in the `BILLING` polygon. Staff classification is implemented as a deterministic torso-color heuristic because the final uniform palette is unknown until the real clips arrive. The class is deliberately isolated in `pipeline/staff.py` so a FashionCLIP implementation can replace it without touching event generation.

## API endpoint behavior

`POST /events/ingest` accepts either a raw event array or `{ "events": [...] }`, validates each row independently, stores valid events, deduplicates duplicates, and returns partial success for malformed events. The request limit is 500 events per batch. The structured logging middleware writes one JSON log per request with `trace_id`, endpoint, latency, status code, and ingest event count.

`GET /stores/{id}/metrics` resolves a time window. If the caller does not pass `start` and `end`, it first tries the current UTC day. If the store has no events today, it falls back to the latest day present for that store. This keeps the API useful for historical challenge clips whose timestamps may not match the machine's current date while still behaving like a real-time API when live events are flowing.

`GET /stores/{id}/funnel` counts sessions, not raw events. Re-entry events for an existing `visitor_id` therefore do not inflate the entry stage. `GET /stores/{id}/heatmap` returns zone visits, dwell, and a normalized 0-100 score. `GET /stores/{id}/anomalies` detects billing queue spikes, conversion drops versus a seven-day baseline when enough history exists, and dead zones with no visits in the last 30 minutes of the active window. `GET /health` checks the database and flags stale feeds when a store has not emitted events for more than 10 minutes.

## AI-Assisted Decisions

### 1. Tracking architecture

An LLM initially suggested a combined detector-and-embedding approach because it looked elegant and fast. I kept the architectural comparison but chose a ByteTrack-style tracking-by-detection path for the first deployable version because the problem statement stresses occlusion, crowded billing queues, and low-confidence detections. ByteTrack's practical advantage is that it keeps weak boxes in the association process instead of discarding them too early. The current implementation exposes this as a YOLO + ByteTrack integration, with the event state machine kept independent from the detector.

### 2. Test architecture

An LLM was used to draft adversarial tests for the state machine, especially empty-store periods. The generated tests were too generic, so they were changed to match the 15fps temporal assumptions and to assert that 10,000 consecutive empty frames do not produce null events or crashes. Similar prompt blocks are included at the top of every test file, as requested by the challenge.

### 3. Staff detection

An LLM suggested a vision-language classifier using staff-uniform prompts. That is a strong final direction, especially with FashionCLIP, but it introduces heavy model dependencies and cannot be tuned without seeing the real uniforms. I isolated staff classification behind `StaffClassifier` and shipped a deterministic heuristic fallback. After the video links arrive, this module can be upgraded to FashionCLIP using prompts such as "a retail employee wearing a branded staff uniform" versus "a customer wearing casual clothes" while preserving the API and schema.

## Operational notes

The repository is designed to run locally with five commands, but it is production-aware: idempotency is tested, database failures produce structured 503 responses, request logs are structured JSON, and the detection producer can retry event delivery. The terminal dashboard is intentionally simple: it proves the pipeline and API are connected by polling live metrics while the detection pipeline emits events.
