import json
from datetime import datetime, timedelta, time, timezone
from typing import Dict, Any
from .db import get_db

def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def resolve_window(store_id: str, start: str | None = None, end: str | None = None) -> tuple[datetime, datetime]:
    supplied_start = parse_dt(start)
    supplied_end = parse_dt(end)
    if supplied_start and supplied_end:
        return supplied_start, supplied_end

    now = datetime.now(timezone.utc)
    current_start = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
    current_end = current_start + timedelta(days=1)

    if not supplied_start and not supplied_end:
        conn = get_db()
        try:
            current_start_str = current_start.strftime('%Y-%m-%dT%H:%M:%SZ')
            current_end_str = current_end.strftime('%Y-%m-%dT%H:%M:%SZ')
            current_count = conn.execute(
                "SELECT COUNT(*) AS c FROM events WHERE store_id=? AND timestamp>=? AND timestamp<?",
                (store_id, current_start_str, current_end_str),
            ).fetchone()["c"]
            if current_count:
                return current_start, current_end

            latest = conn.execute(
                "SELECT MAX(timestamp) AS ts FROM events WHERE store_id=?",
                (store_id,),
            ).fetchone()["ts"]
            if latest:
                latest_dt = parse_dt(latest) or now
                day_start = datetime.combine(latest_dt.date(), time.min, tzinfo=timezone.utc)
                return day_start, day_start + timedelta(days=1)
        except Exception:
            pass
        finally:
            conn.close()
        return current_start, current_end

    if supplied_start and not supplied_end:
        return supplied_start, supplied_start + timedelta(days=1)
    if supplied_end and not supplied_start:
        return supplied_end - timedelta(days=1), supplied_end
    return current_start, current_end

def correlate_pos(store_id: str, window_start: datetime, window_end: datetime) -> int:
    start_str = window_start.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = window_end.strftime('%Y-%m-%dT%H:%M:%SZ')
    with get_db() as conn:
        billing_events = conn.execute(
            """
            SELECT visitor_id, timestamp, metadata
            FROM events
            WHERE store_id = ? AND event_type = 'BILLING_QUEUE_JOIN' AND is_staff = 0
              AND timestamp >= ? AND timestamp < ?
            """, (store_id, start_str, end_str)
        ).fetchall()
        
        converted_visitor_ids = set()
        for bev in billing_events:
            vid = bev['visitor_id']
            b_time = datetime.fromisoformat(bev['timestamp'].replace('Z', '+00:00'))
            start_time = (b_time - timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            txns = conn.execute(
                """
                SELECT brands FROM pos_transactions
                WHERE store_id = ? AND timestamp >= ? AND timestamp <= ?
                """, (store_id, start_time, bev['timestamp'])
            ).fetchall()
            
            if not txns:
                continue
                
            zones = conn.execute(
                """
                SELECT sku_zone FROM events
                WHERE store_id = ? AND visitor_id = ? AND event_type = 'ZONE_ENTER'
                  AND timestamp >= ? AND timestamp < ?
                """, (store_id, vid, start_str, end_str)
            ).fetchall()
            
            visited_brands = set()
            for z in zones:
                if z['sku_zone']:
                    visited_brands.add(z['sku_zone'].lower())
            
            matched = False
            for t in txns:
                if not t['brands']:
                    continue
                txn_brands = [b.strip().lower() for b in t['brands'].split(',')]
                if any(b in visited_brands for b in txn_brands) or not visited_brands:
                    matched = True
                    break
            
            if matched:
                converted_visitor_ids.add(vid)
                
        return len(converted_visitor_ids)

def get_metrics(store_id: str, start: str | None = None, end: str | None = None) -> Dict[str, Any]:
    try:
        window_start, window_end = resolve_window(store_id, start, end)
        start_str = window_start.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = window_end.strftime('%Y-%m-%dT%H:%M:%SZ')
        with get_db() as conn:
            row = conn.execute(
                """
                SELECT COUNT(DISTINCT visitor_id) as count FROM visitor_sessions 
                WHERE store_id = ? AND is_staff = 0
                  AND entry_time >= ? AND entry_time < ?
                """,
                (store_id, start_str, end_str)
            ).fetchone()
            unique_visitors = row['count'] if row else 0
            
            dwell_row = conn.execute(
                """
                SELECT AVG(dwell_ms) as avg_dwell FROM events 
                WHERE store_id = ? AND event_type = 'ZONE_EXIT' AND is_staff = 0
                  AND timestamp >= ? AND timestamp < ? AND zone_id IS NOT NULL
                """,
                (store_id, start_str, end_str)
            ).fetchone()
            avg_dwell = float(dwell_row['avg_dwell']) / 1000 if dwell_row and dwell_row['avg_dwell'] else 0.0
            
            q_row = conn.execute(
                """
                SELECT queue_depth FROM events 
                WHERE store_id = ? AND event_type = 'BILLING_QUEUE_JOIN'
                  AND timestamp >= ? AND timestamp < ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (store_id, start_str, end_str)
            ).fetchone()
            queue_depth = q_row['queue_depth'] if q_row and q_row['queue_depth'] is not None else 0
                
            converted = correlate_pos(store_id, window_start, window_end)
            conversion_rate = (converted / unique_visitors) if unique_visitors > 0 else 0.0
            
            # Calculate abandonment rate
            joins = conn.execute(
                """
                SELECT DISTINCT visitor_id FROM events
                WHERE store_id=? AND timestamp>=? AND timestamp<? AND is_staff=0
                  AND event_type='BILLING_QUEUE_JOIN'
                """,
                (store_id, start_str, end_str),
            ).fetchall()
            abandons = conn.execute(
                """
                SELECT DISTINCT visitor_id FROM events
                WHERE store_id=? AND timestamp>=? AND timestamp<? AND is_staff=0
                  AND event_type='BILLING_QUEUE_ABANDON'
                """,
                (store_id, start_str, end_str),
            ).fetchall()
            total_joins = len({r["visitor_id"] for r in joins})
            total_abandons = len({r["visitor_id"] for r in abandons})
            abandonment_rate = round(total_abandons / total_joins, 4) if total_joins > 0 else 0.0
            
            return {
                "unique_visitors": unique_visitors,
                "conversion_rate": round(conversion_rate, 4),
                "avg_dwell_seconds": round(avg_dwell, 2),
                "queue_depth": queue_depth,
                "abandonment_rate": abandonment_rate
            }
    except Exception:
        return {
            "unique_visitors": 0,
            "conversion_rate": 0.0,
            "avg_dwell_seconds": 0.0,
            "queue_depth": 0,
            "abandonment_rate": 0.0
        }

def get_funnel(store_id: str, start: str | None = None, end: str | None = None) -> Dict[str, Any]:
    try:
        window_start, window_end = resolve_window(store_id, start, end)
        start_str = window_start.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = window_end.strftime('%Y-%m-%dT%H:%M:%SZ')
        with get_db() as conn:
            entry_count = conn.execute(
                """
                SELECT COUNT(DISTINCT visitor_id) FROM visitor_sessions 
                WHERE store_id = ? AND is_staff = 0
                  AND entry_time >= ? AND entry_time < ?
                """,
                (store_id, start_str, end_str)
            ).fetchone()[0]
            
            zone_count = conn.execute(
                """
                SELECT COUNT(DISTINCT visitor_id) FROM events 
                WHERE store_id = ? AND event_type = 'ZONE_ENTER' AND is_staff = 0
                  AND timestamp >= ? AND timestamp < ?
                """,
                (store_id, start_str, end_str)
            ).fetchone()[0]
            
            queue_count = conn.execute(
                """
                SELECT COUNT(DISTINCT visitor_id) FROM events 
                WHERE store_id = ? AND event_type = 'BILLING_QUEUE_JOIN' AND is_staff = 0
                  AND timestamp >= ? AND timestamp < ?
                """,
                (store_id, start_str, end_str)
            ).fetchone()[0]
            
            converted = correlate_pos(store_id, window_start, window_end)
            purchase_count = min(converted, queue_count)
            
            return {
                "entry_count": entry_count,
                "zone_count": zone_count,
                "billing_queue_count": queue_count,
                "purchase_count": purchase_count,
                "drop_offs": {
                    "entry_to_zone": round(1 - (zone_count/entry_count if entry_count else 0), 2),
                    "zone_to_queue": round(1 - (queue_count/zone_count if zone_count else 0), 2),
                    "queue_to_purchase": round(1 - (purchase_count/queue_count if queue_count else 0), 2)
                }
            }
    except Exception:
        return {
            "entry_count": 0,
            "zone_count": 0,
            "billing_queue_count": 0,
            "purchase_count": 0,
            "drop_offs": {
                "entry_to_zone": 0.0,
                "zone_to_queue": 0.0,
                "queue_to_purchase": 0.0
            }
        }

def get_heatmap(store_id: str, start: str | None = None, end: str | None = None) -> Dict[str, Any]:
    try:
        window_start, window_end = resolve_window(store_id, start, end)
        start_str = window_start.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = window_end.strftime('%Y-%m-%dT%H:%M:%SZ')
        with get_db() as conn:
            sessions = conn.execute(
                """
                SELECT COUNT(DISTINCT visitor_id) FROM visitor_sessions 
                WHERE store_id = ? AND is_staff = 0
                  AND entry_time >= ? AND entry_time < ?
                """, 
                (store_id, start_str, end_str)
            ).fetchone()[0]
            data_confidence = sessions >= 20
            
            rows = conn.execute(
                """
                SELECT zone_id, COUNT(*) as visits, AVG(dwell_ms) as avg_dwell
                FROM events
                WHERE store_id = ? AND event_type = 'ZONE_EXIT' AND is_staff = 0 AND zone_id IS NOT NULL
                  AND timestamp >= ? AND timestamp < ?
                GROUP BY zone_id
                """, (store_id, start_str, end_str)
            ).fetchall()
            
            max_visits = max([r['visits'] for r in rows]) if rows else 1
            zones = []
            for r in rows:
                zones.append({
                    "zone_id": r['zone_id'],
                    "visits": r['visits'],
                    "avg_dwell_seconds": round(r['avg_dwell'] / 1000, 2),
                    "normalized_score": round((r['visits'] / max_visits) * 100)
                })
                
            return {
                "data_confidence": data_confidence,
                "total_sessions": sessions,
                "zones": zones
            }
    except Exception:
        return {
            "data_confidence": False,
            "total_sessions": 0,
            "zones": []
        }

