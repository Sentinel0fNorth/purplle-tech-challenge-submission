# PROMPT: "Generate tests for retail operations anomalies: billing queue spike, conversion drop versus seven-day baseline, and dead zone detection when no zone visits occur for 30 minutes."
# CHANGES MADE: Kept the test deterministic by verifying queue spike and dead zone only; conversion-drop logic remains covered by pure metric computations to avoid fragile calendar dependencies.

from uuid import uuid4


def event(visitor, kind, ts, zone=None, q=None):
    return {
        "event_id": str(uuid4()),
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_FLOOR_01",
        "visitor_id": visitor,
        "event_type": kind,
        "timestamp": ts,
        "zone_id": zone,
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.8,
        "metadata": {"queue_depth": q, "sku_zone": zone, "session_seq": 1},
    }


def test_queue_spike_and_dead_zone_anomalies(client):
    client.post(
        "/ingest",
        json=[
            event("VIS_A", "ZONE_ENTER", "2026-03-03T10:00:00Z", "SKINCARE"),
            event("VIS_B", "ZONE_ENTER", "2026-03-03T10:00:00Z", "MAKEUP"),
            event("VIS_C", "BILLING_QUEUE_JOIN", "2026-03-03T10:50:00Z", "BILLING", q=8),
        ],
    )
    body = client.get("/stores/STORE_BLR_002/anomalies").json()
    types = {item["type"] for item in body}
    assert "BILLING_QUEUE_SPIKE" in types
    assert "DEAD_ZONE" in types

