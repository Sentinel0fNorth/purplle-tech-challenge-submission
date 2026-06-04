import sqlite3
import json

def print_events():
    conn = sqlite3.connect("retail.db")
    conn.row_factory = sqlite3.Row
    events = conn.execute("SELECT event_type, visitor_id, timestamp, metadata FROM events ORDER BY timestamp ASC").fetchall()
    print("=" * 60)
    print(f"Total Events in SQLite Database: {len(events)}")
    print("=" * 60)
    for r in events:
        print(f"  * {r['event_type']:22s} | Visitor: {r['visitor_id']} | Time: {r['timestamp']}")
    print("=" * 60)
    conn.close()

if __name__ == "__main__":
    print_events()
