import requests
import uuid
import sys
import sqlite3

import os

# FastAPI server URL
api_url = os.getenv("API_URL", "http://api:8000")

def get_db_path():
    import os
    return os.getenv("STORE_INTEL_DB") or "/data/store_intelligence.db"

def main():
    print("Connecting to local database to find an aligned POS transaction...")
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # Get one transaction for STORE_1
        txn = conn.execute(
            "SELECT * FROM pos_transactions WHERE store_id = 'STORE_1' ORDER BY timestamp LIMIT 1"
        ).fetchone()
        conn.close()
    except Exception as e:
        print(f"Error querying SQLite database directly: {e}")
        print("Falling back to simulated transaction timestamp '2026-06-02T12:15:05Z' and brand 'Faces Canada'")
        txn = {
            "timestamp": "2026-06-02T12:15:05Z",
            "brands": "Faces Canada"
        }

    if not txn:
        print("No POS transactions found in database! Make sure to align and start the server first.")
        sys.exit(1)

    txn_ts = txn["timestamp"]
    txn_brands = txn["brands"].split(",") if txn["brands"] else ["Faces Canada"]
    brand_to_match = txn_brands[0].strip()

    print(f"Found transaction at {txn_ts} for brand '{brand_to_match}'.")
    print("Preparing simulated visitor events to match this transaction...")

    visitor_id = f"VIS_MATCH_{uuid.uuid4().hex[:6]}"
    
    # CCTV events use ISO-8601 UTC formats.
    # The transaction timestamp is e.g. 2026-06-02T12:15:05Z (which is UTC).
    # We create a billing queue join event 2 minutes before the transaction (e.g. 12:13:05).
    from datetime import datetime, timedelta, timezone
    
    # Parse transaction time
    txn_dt = datetime.fromisoformat(txn_ts.replace("Z", "+00:00"))
    
    entry_ts = (txn_dt - timedelta(minutes=10)).strftime('%Y-%m-%dT%H:%M:%SZ')
    zone_enter_ts = (txn_dt - timedelta(minutes=8)).strftime('%Y-%m-%dT%H:%M:%SZ')
    queue_join_ts = (txn_dt + timedelta(minutes=2)).strftime('%Y-%m-%dT%H:%M:%SZ')

    events = [
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_1",
            "camera_id": "STORE1_CAM3_ENTRY",
            "visitor_id": visitor_id,
            "event_type": "ENTRY",
            "timestamp": entry_ts,
            "confidence": 0.95,
            "is_staff": False,
            "metadata": {
                "session_seq": 1
            }
        },
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_1",
            "camera_id": "STORE1_CAM2_ZONE",
            "visitor_id": visitor_id,
            "event_type": "ZONE_ENTER",
            "timestamp": zone_enter_ts,
            "zone_id": "MAKEUP",
            "confidence": 0.90,
            "is_staff": False,
            "metadata": {
                "sku_zone": brand_to_match,
                "session_seq": 2
            }
        },
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_1",
            "camera_id": "STORE1_CAM5_BILLING",
            "visitor_id": visitor_id,
            "event_type": "BILLING_QUEUE_JOIN",
            "timestamp": queue_join_ts,
            "confidence": 0.88,
            "is_staff": False,
            "metadata": {
                "queue_depth": 2,
                "sku_zone": "CHECKOUT",
                "session_seq": 3
            }
        }
    ]

    print(f"Ingesting events for visitor {visitor_id} to endpoint /events/ingest...")
    try:
        response = requests.post(f"{api_url}/events/ingest", json=events)
        if response.status_code != 200:
            print(f"Ingestion failed with status {response.status_code}: {response.text}")
            sys.exit(1)
        print("Ingestion response:", response.json())
        
        print(f"Querying metrics for STORE_1 in time window around {txn_ts}...")
        # Query metrics with start/end bounds to isolate our test session
        start_bound = (txn_dt - timedelta(minutes=15)).strftime('%Y-%m-%dT%H:%M:%SZ')
        end_bound = (txn_dt + timedelta(minutes=10)).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        metrics_url = f"{api_url}/stores/STORE_1/metrics?start={start_bound}&end={end_bound}"
        print(f"GET {metrics_url}")
        metrics_response = requests.get(metrics_url)
        
        if metrics_response.status_code != 200:
            print(f"Metrics query failed with status {metrics_response.status_code}: {metrics_response.text}")
            sys.exit(1)
            
        metrics = metrics_response.json()
        print("Returned metrics:", metrics)
        
        print("\nChecking conversion rate correctness...")
        rate = metrics.get("conversion_rate", 0.0)
        unique_visitors = metrics.get("unique_visitors", 0)
        print(f"Unique visitors: {unique_visitors}, Conversion Rate: {rate}")
        
        if unique_visitors > 0 and rate > 0.0:
            print("\nSUCCESS: POS transaction matched and conversion rate is non-zero!")
        else:
            print("\nFAILURE: Conversion rate is 0.0 or unique visitors is 0. Verify matching window rules.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error testing POS matching: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
