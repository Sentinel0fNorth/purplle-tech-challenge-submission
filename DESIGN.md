# DESIGN.md — Store Intelligence System

## System Overview
The Apex Retail Store Intelligence System is an end-to-end pipeline designed to bridge the data blind spot in physical retail stores. It processes raw CCTV footage from retail outlets, translates movement into structured business events, and exposes real-time intelligence via a REST API.

---

## 1. System Architecture

The codebase is split into three decoupled components:

1. **Detection & Tracking Pipeline (`pipeline/`):**
   * **Object Detection:** Detects persons on a frame-by-frame basis using YOLOv8n.
   * **Tracker:** Maintains shopper trajectories across frames using ByteTrack.
   * **Staff Filter:** Performs real-time HSV-space uniform filtering on detected shopper crops.
   * **State Machine:** Processes detections to generate retail events (e.g. entry, zone transitions, queue joins, dwell alerts, exits).
   
2. **API & Metric Server (`app/`):**
   * **FastAPI Backend:** Handles events ingestion (`/ingest`) and provides clean analytical endpoints for business metrics, conversion funnels, occupancy heatmaps, and rule-based anomaly detection.
   * **SQLite Database:** Stores incoming events, parsed session lifetimes, and imported point-of-sale (POS) records under WAL mode for high concurrency.

3. **Terminal Dashboard (`pipeline/dashboard.py`):**
   * A terminal UI displaying live metrics fetched from the FastAPI endpoints.

```
                  ┌──────────────────────┐
                  │   Raw CCTV Videos    │
                  └──────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   YOLOv8 + ByteTrack (ML)    │
              └──────────┬───────────────────┘
                         │
                         ▼
              ┌──────────────────────────────┐
              │   Retail State Machine       │
              └──────────┬───────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │ Ingest Client -> REST API Server │
        └────────────────┬─────────────────┘
                         │
                         ▼
              ┌──────────────────────────────┐
              │     SQLite WAL Database      │
              └──────────────────────────────┘
```

---

## 2. Store Layout Calibration

The store physical architecture is defined in [real_store_layout.json](data/real_store_layout.json). Rather than logical categories, layout definitions correspond to physical shelf coordinates:

* **Department Zones (`zone_id`):** Represent major shelf clusters (e.g., `SKINCARE`, `MAKEUP`, `HAIRCARE`).
* **Brand Subzones (`sku_zone`):** Define segments dedicated to specific brands (e.g., `COSRX`, `Lakme`, `Maybelline`).
* **Centroid Point-in-Polygon Mapping:** The state machine uses a ray-casting point-in-polygon algorithm to map a visitor’s bottom-most bounding box coordinate (footpoint) to the corresponding physical shelf layout area.

---

## 3. Core State Logic & Edge Cases

### A. Entrance Line Crossing
* **Diagonal Threshold Support:** Due to angled CCTV camera viewpoints (such as `STORE1_CAM3_ENTRY`), traditional horizontal midline thresholds lead to premature triggers. We support arbitrary diagonal line segments ($y = mx + c$) to trace the physical door threshold.
* **Directional Orientation:** A configurable `entry_line_side` operator determines whether an entry is classified as `"above"` or `"below"` the diagonal line.
* **Line Confirmation:** An entry or exit event is only emitted when a visitor's footpoint is tracked on the designated side for a consecutive number of confirmation frames (default: 3).

### B. Group Entry Splitting
When a group of 2–4 people enter the doorway simultaneously, they are often tracked by YOLO as a single wide bounding box.
* The state machine checks if a track's bounding box is inside the `ENTRY` zone and has an abnormally wide aspect ratio ($>0.85$).
* It dynamically splits the bounding box into $N$ individual vertical slices, assigning virtual sub-track IDs to accurately register the correct visitor count.

### C. Staff Filtering
* Rather than tracking staff as normal shoppers (which deflates conversion metrics), a dedicated HSV uniform classifier evaluates the color histogram of the visitor's torso.
* Store 1 staff (dark blue/black) and Store 2 staff (pink) are filtered out automatically at the pipeline layer, and their events are flagged as `is_staff = 1` in the database.

### D. Spatio-Temporal Re-ID & Frame Freezes
* Shoppers transition between different cameras (e.g., Entry -> Zone -> Billing). The state machine computes cosine similarity on visual appearance embeddings.
* If a frame freezes or drops and the shopper suddenly appears at the billing counter under a different track, a high visual similarity score (>0.90) overrides standard travel time limits to stitch the tracks into a single visitor session.

---

## 4. API Schemas & Data Model

* **`/ingest`:** Bulk ingests behavioural events, ignoring duplicates and logging validation errors.
* **`/stores/{store_id}/metrics`:** Calculates key performance indicators (KPIs) like footfall, conversion rate (correlated with POS data within a 5-minute billing window), average dwell time, and billing queue depth.
* **`/stores/{store_id}/funnel`:** Aggregates visitor counts progressing from `ENTRY` -> `ZONE_ENTER` -> `BILLING_QUEUE_JOIN` -> `PAYMENT`.
* **`/stores/{store_id}/anomalies`:** Evaluates rule-based alerts including crowd spikes, long customer dwell times, rapid store exit groups, and queue congestion.
