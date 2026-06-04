import sqlite3
from typing import List, Tuple, Dict, Any
from datetime import datetime, timezone
from .models import EventPayload
from .db import get_db

def parse_timestamp(value: str) -> datetime:
    value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def ingest_events(events: List[EventPayload]) -> Tuple[int, int, List[Dict[str, Any]]]:
    ingested = 0
    duplicates = 0
    errors = []
    
    with get_db() as conn:
        for event in events:
            try:
                # Track Stitching logic
                cursor = conn.execute(
                    "SELECT stitched_visitor_id FROM stitched_visitors WHERE store_id = ? AND original_visitor_id = ?",
                    (event.store_id, event.visitor_id)
                )
                row = cursor.fetchone()
                if row:
                    event.visitor_id = row["stitched_visitor_id"]
                else:
                    if event.event_type not in ("ENTRY", "REENTRY", "EXIT"):
                        active_session = conn.execute(
                            """
                            SELECT visitor_id, entry_time FROM visitor_sessions
                            WHERE store_id = ? AND is_staff = 0 AND (exit_time IS NULL OR exit_time = '') AND entry_time <= ?
                            ORDER BY entry_time DESC LIMIT 1
                            """,
                            (event.store_id, event.timestamp)
                        ).fetchone()
                        
                        if active_session:
                            entry_dt = parse_timestamp(active_session["entry_time"])
                            event_dt = parse_timestamp(event.timestamp)
                            diff_seconds = (event_dt - entry_dt).total_seconds()
                            if 0 <= diff_seconds <= 30.0:
                                stitched_id = active_session["visitor_id"]
                                conn.execute(
                                    """
                                    INSERT OR IGNORE INTO stitched_visitors (store_id, original_visitor_id, stitched_visitor_id)
                                    VALUES (?, ?, ?)
                                    """,
                                    (event.store_id, event.visitor_id, stitched_id)
                                )
                                event.visitor_id = stitched_id

                meta_json = event.metadata.model_dump_json()
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO events 
                    (event_id, store_id, camera_id, visitor_id, event_type, timestamp, zone_id, dwell_ms, is_staff, confidence, metadata, queue_depth, sku_zone)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id, event.store_id, event.camera_id, event.visitor_id, 
                        event.event_type, event.timestamp, event.zone_id, event.dwell_ms, 
                        event.is_staff, event.confidence, meta_json,
                        event.metadata.queue_depth, event.metadata.sku_zone
                    )
                )
                
                if cursor.rowcount == 0:
                    duplicates += 1
                    continue
                    
                ingested += 1
                
                # Materialize visitor sessions
                session_id = f"{event.store_id}:{event.visitor_id}"
                
                if event.event_type in ('ENTRY', 'REENTRY'):
                    conn.execute(
                        """
                        INSERT INTO visitor_sessions (session_id, visitor_id, store_id, entry_time, is_staff, session_seq)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET 
                            entry_time = min(entry_time, excluded.entry_time),
                            is_staff = excluded.is_staff
                        """,
                        (session_id, event.visitor_id, event.store_id, event.timestamp, event.is_staff, 1)
                    )
                elif event.event_type == 'EXIT':
                    conn.execute(
                        """
                        UPDATE visitor_sessions SET exit_time = max(IFNULL(exit_time, ''), ?)
                        WHERE session_id = ?
                        """,
                        (event.timestamp, session_id)
                    )
                else:
                    # Fallback for zone/billing events ensuring session exists
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO visitor_sessions (session_id, visitor_id, store_id, entry_time, is_staff, session_seq)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (session_id, event.visitor_id, event.store_id, event.timestamp, event.is_staff, 1)
                    )
                    
            except Exception as e:
                errors.append({"event_id": event.event_id, "reason": str(e)})
        
        conn.commit()
    
    return ingested, duplicates, errors
