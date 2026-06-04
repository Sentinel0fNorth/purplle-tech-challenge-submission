import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from .db import get_db

def get_anomalies(store_id: str, start: str | None = None, end: str | None = None) -> List[Dict[str, Any]]:
    from .metrics import resolve_window, get_metrics
    window_start, window_end = resolve_window(store_id, start, end)
    start_str = window_start.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = window_end.strftime('%Y-%m-%dT%H:%M:%SZ')
    thirty_mins_ago = (window_end - timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    anomalies = []
    with get_db() as conn:
        # 1. Queue Spike
        q_row = conn.execute(
            """
            SELECT metadata, timestamp FROM events 
            WHERE store_id = ? AND event_type = 'BILLING_QUEUE_JOIN'
              AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            (store_id, start_str, end_str)
        ).fetchone()
        
        if q_row:
            m = json.loads(q_row['metadata'])
            depth = m.get('queue_depth', 0)
            if depth > 5:
                anomalies.append({
                    "type": "BILLING_QUEUE_SPIKE",
                    "severity": "WARN",
                    "description": f"Queue depth reached {depth}.",
                    "suggested_action": "Open another billing counter."
                })
        
        # 2. Dead Zone
        zones = conn.execute(
            "SELECT DISTINCT zone_id FROM events WHERE store_id = ? AND zone_id IS NOT NULL AND zone_id NOT IN ('BILLING', 'ENTRY')", 
            (store_id,)
        ).fetchall()
        for z in zones:
            zone_id = z['zone_id']
            recent = conn.execute(
                """
                SELECT COUNT(*) FROM events 
                WHERE store_id = ? AND zone_id = ? AND event_type = 'ZONE_ENTER' 
                  AND timestamp >= ? AND timestamp < ?
                """,
                (store_id, zone_id, thirty_mins_ago, end_str)
            ).fetchone()[0]
            if recent == 0:
                anomalies.append({
                    "type": "DEAD_ZONE",
                    "severity": "CRITICAL",
                    "description": f"No visits in zone {zone_id} for 30 minutes.",
                    "suggested_action": "Check camera feed or product layout."
                })
                
        # 3. Conversion Drop vs 7-day baseline
        current_metrics = get_metrics(store_id, start_str, end_str)
        current_rate = current_metrics["conversion_rate"]
        baselines = []
        for i in range(1, 8):
            day_start = window_start - timedelta(days=i)
            day_end = day_start + timedelta(days=1)
            m = get_metrics(store_id, day_start.strftime('%Y-%m-%dT%H:%M:%SZ'), day_end.strftime('%Y-%m-%dT%H:%M:%SZ'))
            if m["unique_visitors"] > 0:
                baselines.append(m["conversion_rate"])
        if baselines:
            avg_baseline = sum(baselines) / len(baselines)
            if avg_baseline > 0 and current_rate < avg_baseline * 0.5:
                anomalies.append({
                    "type": "CONVERSION_DROP",
                    "severity": "WARN" if current_rate >= avg_baseline * 0.25 else "CRITICAL",
                    "description": f"Conversion rate dropped to {current_rate:.2%} vs 7-day average of {avg_baseline:.2%}.",
                    "suggested_action": "Check staffing, checkout bottlenecks, or store traffic trends."
                })
                
    return anomalies

