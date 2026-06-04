# CHOICES.md — Engineering Decisions & Trade-offs

This document details the core engineering decisions, structural trade-offs, and critical critiques of AI advice during the development of the Store Intelligence system.

---

## 1. Object Detection & Tracking Model Selection

### Options Considered
1. **YOLOv8x (Extra Large) + BoT-SORT:** High-parameter model with advanced state-tracking.
2. **YOLOv8n (Nano) + ByteTrack:** Low-parameter model optimized for CPU execution.
3. **OpenCV Background Subtraction (MOG2):** Lightweight motion pixel difference detector.

### What AI Suggested
The AI recommended using **YOLOv8x** to maximize mean Average Precision (mAP) for person detection and **BoT-SORT** to track shoppers, arguing that higher confidence would prevent ID-switching in cluttered retail stores.

### What We Chose and Why
We chose **YOLOv8n + ByteTrack** (with a fallback to OpenCV Motion Detection if Ultralytics is unavailable).
* **Speed vs Precision:** Retail environments use standard commodity hardware. YOLOv8x runs at less than 1 frame per second on CPUs, causing frame drops that break tracking. YOLOv8n runs 3x faster while retaining sufficient person-detection accuracy under standard indoor store lighting.
* **Occlusion Tracking:** Shoppers are frequently hidden by banners, displays, and other customers. BoT-SORT drops tracks as soon as detection confidence drops. ByteTrack retains associations for low-confidence detections, allowing the tracker to stitch the visitor's path continuously when they emerge from behind a shelf.
* **Static Camera Optimization:** BoT-SORT includes camera motion compensation (egomotion), which is highly expensive and useless for static CCTV cameras. ByteTrack is a simpler, linear Kalman-filter based association algorithm, making it far more efficient.

---

## 2. Event Schema Design Rationale

### Options Considered
1. **Hierarchical Session Schema:** The pipeline gathers all events locally and sends a single, large JSON payload summarizing the customer's journey when they exit.
2. **Flat Event Stream (Selected):** The pipeline emits individual events (e.g. `ENTRY`, `ZONE_ENTER`, `BILLING_QUEUE_JOIN`, `EXIT`) as they occur.
3. **Normalized Relational Ingestion:** The pipeline POSTs to separate endpoints depending on event type (e.g., `/events/entry`, `/events/dwell`).

### What AI Suggested
The AI suggested using a **Hierarchical Session Schema** to minimize HTTP requests. The pipeline would buffer a visitor's shelf movements and queue joins, emitting a single integrated timeline upon exit.

### What We Chose and Why
We chose the **Flat Event Stream** payload.
* **Fault Tolerance:** If a camera crashes, a power cut occurs, or the CV pipeline is terminated, a hierarchical system loses the entire shopper's history. A flat stream ensures that events are ingested and saved to the SQLite database immediately. Even if a session exits abnormally, we retain their entry, skincare visits, and queue joins.
* **Real-Time Visibility:** Hierarchical sessions only inform the database *after* a customer has left the store. Flat event streams allow us to fetch live parameters (like the current `queue_depth` in `/stores/STORE_1/metrics` or active anomalies) while customers are still shopping.
* **Unified API Receiver:** Having separate endpoints for entries, exits, and dwells increases route maintenance overhead. A single endpoint `/events/ingest` accepts arrays of generic payloads and parses them database-side, simplifying client emission logic.

---

## 3. API Database & Storage Choice

### Options Considered
1. **FastAPI + PostgreSQL + Redis Streams:** Full production relational database and in-memory event stream.
2. **FastAPI + SQLite (WAL Mode) (Selected):** Zero-configuration filesystem database.
3. **Node.js Express + MongoDB:** Document store suited for loose event payloads.

### What AI Suggested
The AI recommended a clustered **PostgreSQL** instance to handle persistent event logs, coupled with a **Redis Cache** to manage active store occupancy states and queue depth variables.

### What We Chose and Why
We chose **FastAPI + SQLite in Write-Ahead Logging (WAL) mode**.
* **Zero-Touch Acceptance Gate:** The mandatory evaluation criteria requires the API to start instantly using `docker compose up` with zero manual steps. Running PostgreSQL and Redis containers introduces boot sequence race conditions where FastAPI starts before the database initializes, requiring complex entrypoint scripts. SQLite is serverless and boots instantly.
* **High Ingestion Throughput:** By configuring SQLite with `PRAGMA journal_mode=WAL` and `PRAGMA synchronous=NORMAL`, we isolate writes. SQLite WAL mode supports concurrent readers and writes without database locking errors.
* **Decoupled State Management:** Standard SQL queries on a indexed database (indexing `store_id`, `timestamp`, and `event_type`) are fast enough to calculate metrics on-the-fly. This removes the need for Redis, keeping the container memory footprint under 150MB.

---

## 4. Algorithmic Calibration Choices

### Diagonal Line Crossing vs Horizontal Midlines
Angled entry cameras view doorways diagonally. A horizontal threshold line causes visitors to cross too early or too late. We calculate diagonal equations ($y = mx + c$) to represent the physical boundary, confirm coordinates over 3 frames to filter noise, and ignore diagonal boundary oscillations.

### Torso-HSV vs Deep Uniform Classifiers
Running deep uniform classification models on shopper crops requires extensive training datasets and slows CPU processing. Since staff uniforms are solid, consistent colors (dark blue for Store 1, pink for Store 2), measuring BGR torso crop distances in HSV color space yields high accuracy with zero training overhead.
