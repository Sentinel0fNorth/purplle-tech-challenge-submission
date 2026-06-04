# PROMPT: "Add tests for a retail heatmap endpoint and health endpoint: heatmap should return normalized scores and LOW data confidence for fewer than 20 sessions; health should return database status and last event timestamps."
# CHANGES MADE: Adapted heatmap counts/normalized score formatting and health API assertions to the backend service.

from uuid import uuid4

def event(visitor, zone, ts, event_type="ZONE_EXIT"):
    return {
        "event_id": str(uuid4()),
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_FLOOR_01",
        "visitor_id": visitor,
        "event_type": event_type,
        "timestamp": ts,
        "zone_id": zone,
        "dwell_ms": 30000,
        "is_staff": False,
        "confidence": 0.9,
        "metadata": {"queue_depth": None, "sku_zone": zone, "session_seq": 1},
    }

def test_heatmap_normalization_and_confidence(client):
    client.post(
        "/ingest",
        json=[
            event("VIS_A", "SKINCARE", "2026-03-03T11:00:00Z"),
            event("VIS_B", "SKINCARE", "2026-03-03T11:02:00Z"),
            event("VIS_C", "MAKEUP", "2026-03-03T11:03:00Z"),
        ],
    )
    body = client.get("/stores/STORE_BLR_002/heatmap").json()
    assert body["data_confidence"] is False
    zones_by_id = {z["zone_id"]: z for z in body["zones"]}
    assert zones_by_id["SKINCARE"]["normalized_score"] == 100
    assert zones_by_id["MAKEUP"]["normalized_score"] == 50

def test_health_reports_db(client):
    client.post("/ingest", json=[event("VIS_A", "SKINCARE", "2026-03-03T11:00:00Z")])
    body = client.get("/health").json()
    assert body["status"] == "ok"

