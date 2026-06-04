# PROMPT: "Generate Pytest tests for a FastAPI event ingestion endpoint that validates a batch of retail CCTV events, deduplicates by event_id, supports partial success for malformed rows, and verifies idempotency when the same payload is posted twice."
# CHANGES MADE: Tightened the event schema to match the Purplle challenge fields, asserted IngestResponse schema, and added a duplicate replay assertion.

from uuid import uuid4

def event(event_id=None, visitor="VIS_1", event_type="ENTRY", zone=None, seq=1):
    return {
        "event_id": str(event_id or uuid4()),
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": visitor,
        "event_type": event_type,
        "timestamp": "2026-03-03T14:00:00Z",
        "zone_id": zone,
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.9,
        "metadata": {"queue_depth": None, "sku_zone": zone, "session_seq": seq},
    }

def test_ingest_duplicates_and_idempotency(client):
    stable_id = str(uuid4())
    payload = [event(stable_id)]

    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["ingested"] == 1
    assert body["duplicates"] == 0

    replay = client.post("/ingest", json=[event(stable_id)])
    assert replay.status_code == 200
    assert replay.json()["ingested"] == 0
    assert replay.json()["duplicates"] == 1

def test_resource_center_sample_event_is_adapted(client):
    payload = [
        {
            "event_type": "zone_entered",
            "store_code": "store_1076",
            "camera_id": "CAM_01",
            "track_id": 42,
            "event_timestamp": "2026-04-10T10:05:00Z",
            "zone_name": "SKINCARE",
            "group_size": 2,
            "confidence": 0.77,
        }
    ]
    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["ingested"] == 1
    assert len(body["errors"]) == 0

def test_resource_center_queue_event_maps_wait_and_depth(client):
    payload = [
        {
            "event_type": "queue_abandoned",
            "store_id": "ST1008",
            "camera_id": "CAM_01",
            "queue_event_id": "queue-row-1",
            "queue_exit_ts": "2026-04-10T10:10:00Z",
            "queue_position_at_join": 4,
            "wait_seconds": 95.5,
        }
    ]
    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["ingested"] == 1
    assert len(body["errors"]) == 0

def test_raw_line_item_pos_csv_is_normalized(client, tmp_path, monkeypatch):
    from app.db import get_db
    from app.pos_loader import load_pos_data

    csv_path = tmp_path / "raw_pos.csv"
    csv_path.write_text(
        "store_id,order_id,order_date,order_time,brand_name,total_amount\n"
        "ST1008,INV1,10-04-2026,10:01:00,Maybelline,100.50\n"
        "ST1008,INV1,10-04-2026,10:01:00,Faces Canada,99.50\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("POS_CSV_PATH", str(csv_path))
    load_pos_data()
    with get_db() as conn:
        row = conn.execute("SELECT store_id, order_id, total_amount, brands FROM pos_transactions").fetchone()
    assert row["store_id"] == "ST1008"
    assert row["order_id"] == "INV1"
    assert row["total_amount"] == 200.0
    assert "Maybelline" in row["brands"]
    assert "Faces Canada" in row["brands"]

