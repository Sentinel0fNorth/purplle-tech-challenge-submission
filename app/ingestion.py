import sqlite3
from typing import List, Tuple, Dict, Any
from .models import EventPayload
from .db import get_db

def ingest_events(events: List[EventPayload]) -> Tuple[int, int, List[Dict[str, Any]]]:
    ingested = 0
    duplicates = 0
    errors = []
    
    with get_db() as conn:
        for event in events:
            try:
                meta_json = event.metadata.model_dump_json()
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO events 
                    (event_id, store_id, camera_id, visitor_id, event_type, timestamp, zone_id, dwell_ms, is_staff, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id, event.store_id, event.camera_id, event.visitor_id, 
                        event.event_type, event.timestamp, event.zone_id, event.dwell_ms, 
                        event.is_staff, event.confidence, meta_json
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
