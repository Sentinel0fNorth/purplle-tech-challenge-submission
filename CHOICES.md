# CHOICES.md — Engineering Decisions & Trade-offs

This document details the engineering choices and design trade-offs chosen during implementation.

---

## 1. Object Detection & Tracking Choices

### YOLOv8n (Nano) over YOLOv8x (Extra-Large)
* **Rationale:** The application targets real-time deployment on standard retail commodity hardware (often lacking high-end GPUs). YOLOv8n runs over $3\times$ faster than YOLOv8x on CPU while maintaining an acceptable mAP for person tracking in well-lit indoor store conditions. High inference frame rate is critical to feed ByteTrack stable continuous detections.

### ByteTrack over SORT
* **Rationale:** SORT relies solely on high-confidence bounding boxes. In retail stores with high shelves and display banners, shoppers are regularly partially occluded. ByteTrack preserves low-confidence detection association history, preventing tracking fragmentation when a shopper is briefly hidden by displays or other customers.

---

## 2. Spatial Mapping & Zone Modeling

### Footpoint Coordinates over Bounding Box Centers
* **Rationale:** Using the center of the bounding box to assign a visitor to a zone causes visual distortion. If a customer stands at the edge of a shelf, their bounding box center might fall outside the zone or into an adjacent column. Tracing the bottom-middle coordinate of the bounding box (their feet) represents their actual physical location on the floor plane.

### Decoupled Physical vs Logical Categories
* **Rationale:** Product inventory catalogs mapping brands to categories (e.g. `Alps Goodness` to Skincare) can change dynamically. Rather than hardcoding product rules in the state machine, the layout maps the visual shelf boundaries as they appear in the video stream. The database and API handle category mapping dynamically by joining transactional POS logs with spatial tracking data.

---

## 3. Algorithmic Solutions for Edge Cases

### HSV toros-based Staff Filtering
* **Rationale:** A deep learning uniform classifier requires extensive training data. Instead, we use HSV color-space torso cropping. Store staff uniform colors are highly consistent (dark blue/black for Store 1, pink for Store 2). Measuring BGR-converted HSV Euclidean distance allows us to filter staff with zero-shot accuracy, protecting conversion metrics.

### Aspect-Ratio Based Group Splitting
* **Rationale:** When multiple people enter side-by-side, trackers merge them into a single wide detection box. To avoid undercounting entries, the state machine detects wide aspect ratios ($>0.85$) inside entry zones and splits the box into vertical sub-slices. This represents a robust rule-based solution that does not require heavy multi-object parsing models.

### Diagonal Threshold Lines
* **Rationale:** Angled entry cameras show the entrance threshold as a diagonal line. A horizontal midline threshold splits the doorway space unevenly, triggering early/late entries depending on which side a customer walks in. Diagonal boundary line equations ($y = mx + c$) allow us to trace the physical door threshold accurately.

---

## 4. API & Database Storage Choices

### SQLite (WAL Mode) over Postgres/Redis
* **Rationale:** A containerized microservice stack must start instantly with `docker compose up` without waiting for database initialization scripts or credentials. SQLite is a lightweight, zero-configuration database. By enabling Write-Ahead Logging (WAL) and synchronous normal mode, we achieve high-speed concurrent ingestion and quick query times, satisfying the requirements without adding external service overhead.
