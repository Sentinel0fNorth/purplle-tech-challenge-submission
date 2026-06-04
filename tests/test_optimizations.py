# PROMPT: "Write integration tests in pytest to verify: 1) ingestion track stitching from entry sessions, 2) top-level indexed database column writes for metrics, 3) zero-traffic metrics fallback safety, and 4) stale feed warnings in health checks."
# CHANGES MADE: Added explicit datetime offsets relative to current UTC clock to guarantee health check stale feed warnings trigger predictably.

from datetime import datetime, timezone, timedelta
from uuid import uuid4
from app.db import get_db

def make_event(visitor, camera, ts, event_type="ENTRY", zone=None, queue_depth=None, sku_zone=None):
    return {
        "event_id": str(uuid4()),
        "store_id": "STORE_BLR_002",
        "camera_id": camera,
        "visitor_id": visitor,
        "event_type": event_type,
        "timestamp": ts,
        "zone_id": zone,
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.95,
        "metadata": {"queue_depth": queue_depth, "sku_zone": sku_zone, "session_seq": 1},
    }

def test_ingestion_time_track_stitching(client):
    # 1. Ingest an ENTRY event for VIS_1 on CAM_ENTRY_01 at 14:00:00
    entry_ts = "2026-03-03T14:00:00Z"
    entry_event = make_event("VIS_1", "CAM_ENTRY_01", entry_ts, "ENTRY")
    
    resp = client.post("/ingest", json=[entry_event])
    assert resp.status_code == 200
    assert resp.json()["ingested"] == 1

    # 2. Ingest a floor event (ZONE_ENTER) for VIS_2 on CAM_FLOOR_01 at 14:00:15 (15 seconds later)
    # This should trigger session stitching since VIS_2 entered 15 seconds after VIS_1's entry
    floor_ts = "2026-03-03T14:00:15Z"
    floor_event = make_event("VIS_2", "CAM_FLOOR_01", floor_ts, "ZONE_ENTER", "SKINCARE", sku_zone="MOISTURISER")
    
    resp2 = client.post("/ingest", json=[floor_event])
    assert resp2.status_code == 200
    assert resp2.json()["ingested"] == 1

    # Check database to see if the visitor_id for the floor event was mapped to VIS_1
    with get_db() as conn:
        row = conn.execute("SELECT visitor_id, sku_zone FROM events WHERE event_id = ?", (floor_event["event_id"],)).fetchone()
        assert row is not None
        assert row["visitor_id"] == "VIS_1"
        assert row["sku_zone"] == "MOISTURISER"

    # 3. Subsequent event for VIS_2 at 14:01:00 (more than 30s after the entry event, but already stitched)
    subsequent_ts = "2026-03-03T14:01:00Z"
    subsequent_event = make_event("VIS_2", "CAM_FLOOR_01", subsequent_ts, "ZONE_EXIT", "SKINCARE")
    
    resp3 = client.post("/ingest", json=[subsequent_event])
    assert resp3.status_code == 200
    
    with get_db() as conn:
        row = conn.execute("SELECT visitor_id FROM events WHERE event_id = ?", (subsequent_event["event_id"],)).fetchone()
        assert row is not None
        assert row["visitor_id"] == "VIS_1"

def test_top_level_indexed_columns(client):
    # Ingest a billing join event with queue_depth
    ts = "2026-03-03T14:05:00Z"
    event_payload = make_event("VIS_99", "CAM_FLOOR_01", ts, "BILLING_QUEUE_JOIN", "BILLING", queue_depth=4, sku_zone="CHECKOUT")
    
    resp = client.post("/ingest", json=[event_payload])
    assert resp.status_code == 200
    
    with get_db() as conn:
        row = conn.execute("SELECT queue_depth, sku_zone FROM events WHERE event_id = ?", (event_payload["event_id"],)).fetchone()
        assert row is not None
        assert row["queue_depth"] == 4
        assert row["sku_zone"] == "CHECKOUT"

def test_zero_traffic_graceful_degradation(client):
    # Query metrics and other analytics for a store with absolutely no events or traffic
    # This should not raise division by zero or other crashes
    metrics = client.get("/stores/STORE_EMPTY/metrics").json()
    assert metrics["unique_visitors"] == 0
    assert metrics["conversion_rate"] == 0.0
    assert metrics["avg_dwell_seconds"] == 0.0
    assert metrics["queue_depth"] == 0
    assert metrics["abandonment_rate"] == 0.0

    funnel = client.get("/stores/STORE_EMPTY/funnel").json()
    assert funnel["entry_count"] == 0
    assert funnel["zone_count"] == 0
    assert funnel["billing_queue_count"] == 0
    assert funnel["purchase_count"] == 0
    assert funnel["drop_offs"]["entry_to_zone"] == 1.0
    assert funnel["drop_offs"]["zone_to_queue"] == 1.0
    assert funnel["drop_offs"]["queue_to_purchase"] == 1.0

    heatmap = client.get("/stores/STORE_EMPTY/heatmap").json()
    assert heatmap["data_confidence"] is False
    assert heatmap["total_sessions"] == 0
    assert heatmap["zones"] == []

def test_health_stale_feed_warning(client):
    # 1. Ingest an old event (older than 10 minutes from now)
    old_ts = (datetime.now(timezone.utc) - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
    old_event = make_event("VIS_OLD", "CAM_ENTRY_01", old_ts, "ENTRY")
    
    resp = client.post("/ingest", json=[old_event])
    assert resp.status_code == 200

    # 2. Call /health and verify STALE_FEED status is returned
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] == "ok"
    assert health_data["warning"] == "STALE_FEED"

    # 3. Ingest a very fresh event (within 1 minute from now)
    fresh_ts = (datetime.now(timezone.utc) - timedelta(seconds=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    fresh_event = make_event("VIS_FRESH", "CAM_ENTRY_01", fresh_ts, "ENTRY")
    
    resp_fresh = client.post("/ingest", json=[fresh_event])
    assert resp_fresh.status_code == 200

    # 4. Call /health again and verify status is back to ok
    health_resp2 = client.get("/health")
    assert health_resp2.status_code == 200
    assert health_resp2.json()["status"] == "ok"
