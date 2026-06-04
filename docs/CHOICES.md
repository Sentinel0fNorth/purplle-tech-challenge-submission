# CHOICES.md — Key Engineering Decisions

## Decision 1: Detection model and tracking strategy

### Options considered

I considered three detection/tracking approaches: YOLO with ByteTrack, a JDE-style model that produces detections and Re-ID embeddings in one pass, and a simpler contour or optical-flow based tracker. Optical flow was rejected because the videos are anonymized retail CCTV with lighting changes, overlapping shoppers, staff movement, and stationary billing queues. It would be fragile and would not provide a reliable confidence score for the required event schema.

The JDE-style option is attractive because it combines detection and appearance features in one model. An AI assistant suggested this as the first choice because it reduces inference overhead and can help cross-camera identity matching. I did not choose it for the first deployable build because the challenge's hardest cases are partial occlusion and billing queue buildup. In those moments, detection confidence drops. If detection and embedding quality both degrade together, the system can fragment tracks and miss queue occupancy.

### Chosen approach

I chose YOLO tracking with ByteTrack as the default real-video path and a deterministic simulation mode for demos and CI. ByteTrack is a good fit because it associates low-confidence detections to existing tracks instead of throwing them away. That behavior aligns directly with the challenge requirement that confidence should degrade gracefully rather than fail silently. The repository uses `ultralytics` when available and keeps the event logic in a model-agnostic `RetailStateMachine` so that YOLOv11, RT-DETR, or an OSNet feature bank can be added later without rewriting the API.

### What remains after video links arrive

Once the real videos are available, the next calibration step is to run detection on each camera, inspect false positives/false negatives, and adjust model confidence, track buffer, polygon layouts, and staff classification. The code is ready for that because video inputs are supplied by a manifest, not hardcoded paths.

## Decision 2: Event schema design

### Options considered

I considered a minimal event schema with only `visitor_id`, `event_type`, and `timestamp`; a deeply nested schema that stored full bounding boxes and model internals; and the current challenge-aligned schema with metadata. A minimal schema is easy to ingest but pushes too much reconstruction work onto the API. A very rich schema helps debugging but makes scoring brittle and increases payload size.

An AI assistant suggested a flat event schema with chronological sorting as the main way to reconstruct sessions. I kept the flat readability but added a required `metadata.session_seq` ordinal. This field makes each visitor's session order explicit even when events from multiple cameras arrive in slightly different order. It also helps answer follow-up questions because the source of funnel reconstruction is visible in the event data.

### Chosen approach

The chosen schema mirrors the problem statement: globally unique `event_id`, `store_id`, `camera_id`, `visitor_id`, `event_type`, UTC `timestamp`, `zone_id`, `dwell_ms`, `is_staff`, `confidence`, and metadata fields for `queue_depth`, `sku_zone`, and `session_seq`. The API stores metadata as JSON but also extracts queue depth, SKU zone, and sequence into indexed columns for analytics. Low-confidence detections are not suppressed by the API. They are stored with their actual confidence so downstream filtering can be changed without rerunning the videos.

### Why this helps the North Star metric

The North Star metric is offline conversion rate. That metric breaks if entries are double-counted, staff are counted as shoppers, or purchases cannot be correlated to billing sessions. The event schema gives the API enough information to exclude staff, deduplicate sessions, and connect POS transactions to billing-zone presence.

## Decision 3: API architecture and persistence

### Options considered

I considered WebSockets, Kafka/Redpanda, direct database writes from the pipeline, and batched HTTP POST ingestion. WebSockets offer low latency but make idempotent replay and local scoring more complex. Kafka is production-grade but too heavy for a take-home challenge and would complicate `docker compose up`. Direct database writes from the pipeline were rejected because they tightly couple CV processing to analytics storage and make it harder for a backend engineer to own the API independently.

An AI assistant initially suggested WebSockets for the live dashboard because they feel more "real time." I overrode that suggestion and used REST ingestion plus polling for the dashboard. The challenge allows a terminal dashboard, and the more important production property is fault tolerance. A batch of JSON events with stable UUIDs can be retried safely after a timeout or database outage.

### Chosen approach

The API uses FastAPI and SQLite. `POST /events/ingest` accepts up to 500 events per batch, validates each event independently, and returns a structured result containing accepted, deduplicated, and rejected counts. The SQLite primary key on `event_id` enforces idempotency at the storage layer, not only in application code. The service has structured logging middleware and a `/health` endpoint with stale-feed warnings.

### Upgrade path

At 40 stores, the first likely bottleneck is SQLite write concurrency and query aggregation. The code is intentionally organized so PostgreSQL can replace SQLite behind `app/database.py`, and metric queries can later be materialized by store/day if needed. The external API contract does not need to change.
