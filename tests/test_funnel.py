# PROMPT: "Write funnel tests for an offline retail analytics API. Ensure the session, not raw events, is the counting unit and a REENTRY event for the same visitor does not double-count entries."
# CHANGES MADE: Adapted funnel stage counts and drop-offs to the backend service API response models.

from uuid import uuid4
from app.db import get_db

def e(visitor, kind, ts, zone=None, seq=1):
    return {
        "event_id": str(uuid4()),
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": visitor,
        "event_type": kind,
        "timestamp": ts,
        "zone_id": zone,
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.83,
        "metadata": {"queue_depth": 1 if kind == "BILLING_QUEUE_JOIN" else None, "sku_zone": zone, "session_seq": seq},
    }

def test_funnel_session_deduplicates_reentry(client):
    events = [
        e("VIS_A", "ENTRY", "2026-03-03T14:00:00Z", seq=1),
        e("VIS_A", "ZONE_ENTER", "2026-03-03T14:00:20Z", "SKINCARE", seq=2),
        e("VIS_A", "BILLING_QUEUE_JOIN", "2026-03-03T14:02:00Z", "BILLING", seq=3),
        e("VIS_A", "EXIT", "2026-03-03T14:06:00Z", seq=4),
        e("VIS_A", "REENTRY", "2026-03-03T14:07:00Z", seq=5),
    ]
    client.post("/ingest", json=events)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO pos_transactions VALUES (?, ?, ?, ?, ?)",
            ("TXN_FUNNEL", "STORE_BLR_002", "2026-03-03T14:02:00Z", 500.0, "SKINCARE"),
        )
    body = client.get("/stores/STORE_BLR_002/funnel").json()
    assert body["entry_count"] == 1
    assert body["zone_count"] == 1
    assert body["billing_queue_count"] == 1
    assert body["purchase_count"] == 1
    assert body["drop_offs"]["entry_to_zone"] == 0.0
    assert body["drop_offs"]["zone_to_queue"] == 0.0
    assert body["drop_offs"]["queue_to_purchase"] == 0.0

