# PROMPT: "Create tests for retail store metrics that exclude staff, handle zero-purchase stores, compute conversion rate from POS transactions within a 5-minute billing window, and return average dwell time per zone."
# CHANGES MADE: Added a direct SQLite POS insert to mirror the backend schema and verified the new metric formats.

from uuid import uuid4
from app.db import get_db

def mk(visitor, event_type, ts, zone=None, staff=False, dwell=0, q=None, seq=1):
    return {
        "event_id": str(uuid4()),
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_BILLING_01" if zone == "BILLING" else "CAM_FLOOR_01",
        "visitor_id": visitor,
        "event_type": event_type,
        "timestamp": ts,
        "zone_id": zone,
        "dwell_ms": dwell,
        "is_staff": staff,
        "confidence": 0.91,
        "metadata": {"queue_depth": q, "sku_zone": "CHECKOUT" if zone == "BILLING" else "MOISTURISER", "session_seq": seq},
    }

def test_metrics_exclude_staff_and_correlate_pos(client):
    events = [
        mk("VIS_A", "ENTRY", "2026-03-03T14:00:00Z"),
        mk("VIS_A", "ZONE_DWELL", "2026-03-03T14:00:45Z", "SKINCARE", dwell=30000),
        mk("VIS_A", "BILLING_QUEUE_JOIN", "2026-03-03T14:01:00Z", "BILLING", q=2),
        mk("VIS_B", "ENTRY", "2026-03-03T14:02:00Z"),
        mk("VIS_STAFF", "ENTRY", "2026-03-03T14:02:05Z", staff=True),
        mk("VIS_STAFF", "BILLING_QUEUE_JOIN", "2026-03-03T14:02:10Z", "BILLING", staff=True, q=3),
        # Emulating exits to ensure zone_exit is counted for avg_dwell
        mk("VIS_A", "ZONE_EXIT", "2026-03-03T14:00:50Z", "SKINCARE", dwell=30000),
    ]
    assert client.post("/ingest", json=events).status_code == 200
    with get_db() as conn:
        conn.execute(
            "INSERT INTO pos_transactions VALUES (?, ?, ?, ?, ?)",
            ("TXN_TEST", "STORE_BLR_002", "2026-03-03T14:01:00Z", 1000.0, "MOISTURISER"),
        )

    response = client.get("/stores/STORE_BLR_002/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["unique_visitors"] == 2
    assert body["conversion_rate"] == 0.5
    assert body["queue_depth"] == 3
    assert body["avg_dwell_seconds"] == 30.0

def test_zero_purchase_store(client):
    client.post("/ingest", json=[mk("VIS_A", "ENTRY", "2026-03-03T14:00:00Z")])
    body = client.get("/stores/STORE_BLR_002/metrics").json()
    assert body["conversion_rate"] == 0.0

