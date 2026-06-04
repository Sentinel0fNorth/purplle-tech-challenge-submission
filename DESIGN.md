# DESIGN.md — Store Intelligence Architecture & Decisions

This document details the architectural design, core state tracking algorithms, layout spatial calibrations, and AI-assisted engineering decisions made during the development of the Apex Retail Store Intelligence System.

---

## 1. System Architecture

The codebase follows a decoupled, event-driven architecture designed to separate heavy computer vision state-tracking from high-throughput analytical query processing:

```
                      ┌──────────────────────┐
                      │   Raw CCTV Videos    │
                      └──────────┬───────────┘
                                 │
                                 ▼ (Inference on CPU)
                  ┌──────────────────────────────┐
                  │   YOLOv8 + ByteTrack (ML)    │
                  └──────────┬───────────────────┘
                             │
                             ▼ (Centroid Footpoints)
                  ┌──────────────────────────────┐
                  │    Retail State Machine      │
                  └──────────┬───────────────────┘
                             │
                             ▼ (HTTP POST JSON events)
            ┌──────────────────────────────────┐
            │  Ingestion API (FastAPI Router)  │
            └────────────────┬─────────────────┘
                             │
                             ▼ (Write-Ahead Log)
                  ┌──────────────────────────────┐
                  │     SQLite WAL Database      │
                  └──────────────────────────────┘
```

The system is split into three core modules:
1. **ML Detection & Tracking Pipeline (`pipeline/`):** Processes video frames, tracks persons using YOLOv8-Nano + ByteTrack, filters out store staff uniforms using HSV torso crop distance, maps foot centroids to layout polygons, and resolves diagonal threshold crossings. It runs a state machine to emit events like `ENTRY`, `EXIT`, `ZONE_ENTER`, `ZONE_EXIT`, `ZONE_DWELL`, and `BILLING_QUEUE_JOIN`.
2. **FastAPI Analytics Backend (`app/`):** Validates incoming events via Pydantic, deduplicates using unique `event_id` keys, persists records in SQLite, maps transaction products to layout zones during query-time, and exposes metrics, heatmap, funnel, and anomalies endpoints.
3. **Live Terminal UI (`pipeline/dashboard.py`):** Periodically polls FastAPI REST endpoints and renders real-time occupancy statistics, conversion rates, and queue indicators inside a console window.

---

## 2. Store Layout Spatial Calibration

Physical retail shelves are calibrated using 2D image-coordinate polygons in `data/real_store_layout.json`. 
* **Zone Definitions:**
  * **Departments (`zone_id`):** Cover major zones such as `SKINCARE`, `HAIRCARE`, and `MAKEUP`.
  * **Brand Bays (`sku_zone`):** Define smaller brand-specific polygons (e.g., `Good Vibes`, `Minimalist`, `Lakme`) nested within department boundaries.
* **Coordinate Mapping:** A shopper's position is mapped using their **footpoint centroid** (bottom-middle of the bounding box: $x_{foot} = (x_1 + x_2)/2, y_{foot} = y_2$). A ray-casting point-in-polygon algorithm determines which department and brand-bay coordinates contain the shopper's feet, preventing spatial bleeding caused by simple bounding box center calculations.

---

## 3. Core State Tracking & Edge Case Logic

### A. Diagonal Threshold Crossing (`ENTRY` & `EXIT`)
CCTV entry cameras (e.g. `STORE1_CAM3_ENTRY`) view the doorway from an angle, making traditional horizontal line thresholds inaccurate.
* **Equation Calibration:** We define door boundaries using diagonal line segments ($y = mx + c$).
* **Line Debouncing:** A crossing is detected when a visitor's coordinate changes sign relative to the line ($y - (mx + c)$). To prevent boundary oscillation (the ping-pong effect), an entry or exit is only emitted if the coordinates remain on the crossed side for 3 consecutive frames.

### B. Torso HSV Staff Filtering
Store staff walk through all zones regularly, distorting customer conversion metrics. 
* **Torso Crop:** The pipeline crops the upper-middle region of each detected bounding box.
* **HSV Histogram Distance:** Instead of complex models, we measure the Euclidean distance in HSV color space against calibrated uniform profiles (dark blue/black for Store 1, pink for Store 2). Matches are flagged as `is_staff = 1` and excluded from API metrics.

### C. Bounding Box Group Splitting
When groups of shoppers enter together, YOLO frequently groups them into a single wide bounding box, causing undercounting.
* **Aspect-Ratio Trigger:** If a bounding box is inside an entry zone and has an abnormally wide aspect ratio ($\text{Width} / \text{Height} > 0.85$), the state machine triggers group splitting.
* **Dynamic Slicing:** It divides the wide box into $N$ individual vertical strips, registering virtual visitor IDs and generating distinct entry events to preserve footfall accuracy.

### D. Spatio-Temporal Re-ID & Frame Freezes
If a camera frame drops or freezes, a visitor might disappear and reappear at the billing counter with a different track ID.
* **Appearance Embeddings:** The tracker extracts visual color histogram descriptors.
* **Freeze Bypass:** If a shopper reappears within a short window, a high cosine similarity score ($>0.90$) overrides travel time rules and merges the tracks into a single `visitor_id` session, avoiding double-counting in the funnel.

---

## 4. AI-Assisted Decisions

During system design, we interacted with LLMs to evaluate trade-offs. Below are the key places where we critiques and altered AI-suggested designs:

### Decision 1: Shopper Group Splitting
* **What AI Suggested:** The AI proposed using a deep-learning posture estimation model (e.g., MediaPipe Pose or YOLOv8-pose) to detect individual heads or skeletal keypoints to split crowded groups in the entry threshold.
* **Our Override:** We overrode this advice. Running keypoint models inside a CPU-only docker container would grind inference speed to a crawl. Instead, we implemented a rule-based aspect-ratio bounding box slice heuristic ($\text{W}/\text{H} > 0.85$ splits the box vertically by a multiplier of $H \times 0.45$). This is extremely fast, uses zero extra memory, and successfully splits side-by-side entries.

### Decision 2: Spatial Calibration Decoupling
* **What AI Suggested:** The AI recommended hardcoding logical catalog maps (e.g., brand-to-category associations) inside the CV tracking script to output category labels directly.
* **Our Agreement:** We agreed with the AI's warning that product inventory layouts change often. We decoupled this: the pipeline only emits spatial coordinate labels (`sku_zone` as the brand bay). The API database maps transactions and categories dynamically during SQL queries by joining POS logs with spatial tracking events.

### Decision 3: Event Ingestion Queue
* **What AI Suggested:** The AI strongly recommended implementing a heavy message broker (like RabbitMQ or Redis Streams) to buffer incoming events from `/events/ingest` before database writes.
* **Our Override:** We overrode the AI. A candidate submission must boot instantly with zero external dependencies. Adding RabbitMQ increases configuration overhead. Instead, we configured the SQLite database in **Write-Ahead Logging (WAL)** mode with `journal_mode=WAL` and `synchronous=NORMAL`. This allows high-concurrency ingestion and thread-safe batch writes directly through FastAPI, satisfying performance requirements with zero setup footprint.
