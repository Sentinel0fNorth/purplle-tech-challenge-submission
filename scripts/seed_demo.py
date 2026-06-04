from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from app.database import init_db, load_pos_csv
from app.ingestion import ingest_payload


def ev(visitor: str, seq: int, event_type: str, ts: str, zone: str | None = None, dwell: int = 0, staff: bool = False, q: int | None = None):
    return {
        "event_id": str(uuid4()),
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_ENTRY_01" if zone in (None, "ENTRY") else "CAM_FLOOR_01",
        "visitor_id": visitor,
        "event_type": event_type,
        "timestamp": ts,
        "zone_id": zone,
        "dwell_ms": dwell,
        "is_staff": staff,
        "confidence": 0.87,
        "metadata": {"queue_depth": q, "sku_zone": "CHECKOUT" if zone == "BILLING" else "MOISTURISER", "session_seq": seq},
    }


def main() -> int:
    init_db()
    load_pos_csv("data/sample/pos_transactions.csv")
    events = [
        ev("VIS_demo1", 1, "ENTRY", "2026-03-03T14:00:00Z"),
        ev("VIS_demo1", 2, "ZONE_ENTER", "2026-03-03T14:00:15Z", "SKINCARE"),
        ev("VIS_demo1", 3, "ZONE_DWELL", "2026-03-03T14:00:45Z", "SKINCARE", dwell=30000),
        ev("VIS_demo1", 4, "ZONE_EXIT", "2026-03-03T14:00:50Z", "SKINCARE", dwell=35000),
        ev("VIS_demo1", 5, "BILLING_QUEUE_JOIN", "2026-03-03T14:00:55Z", "BILLING", q=3),
        ev("VIS_demo2", 1, "ENTRY", "2026-03-03T14:02:00Z"),
        ev("VIS_staff1", 1, "ZONE_ENTER", "2026-03-03T14:02:05Z", "BILLING", staff=True),
        ev("VIS_demo2", 2, "ZONE_ENTER", "2026-03-03T14:02:25Z", "MAKEUP"),
        ev("VIS_demo2", 3, "ZONE_EXIT", "2026-03-03T14:03:00Z", "MAKEUP", dwell=35000),
        ev("VIS_demo2", 4, "EXIT", "2026-03-03T14:03:30Z"),
        ev("VIS_demo2", 5, "REENTRY", "2026-03-03T14:05:10Z"),
    ]
    result = ingest_payload(events)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
