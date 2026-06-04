import json
import sys

def parse_events(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    entries = []
    exits = []
    others = []
    unique_visitors = set()

    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            event_type = event.get("event_type")
            visitor_id = event.get("visitor_id")
            timestamp = event.get("timestamp")
            
            unique_visitors.add(visitor_id)
            
            if event_type in ("ENTRY", "REENTRY"):
                entries.append((visitor_id, event_type, timestamp))
            elif event_type == "EXIT":
                exits.append((visitor_id, timestamp))
            else:
                others.append((visitor_id, event_type, timestamp))
        except Exception as e:
            print("Failed to parse line:", line, e)

    print("=" * 60)
    print(f"File Analyzed: {file_path}")
    print(f"Total Unique Visitor IDs Identified: {len(unique_visitors)}")
    print(f"Total ENTRY/REENTRY Events: {len(entries)}")
    print("=" * 60)
    for vid, etype, ts in entries:
        print(f"  * {etype:7s} | Visitor: {vid} | Time: {ts}")
    print("=" * 60)
    print(f"Total EXIT Events: {len(exits)}")
    for vid, ts in exits:
        print(f"  * EXIT    | Visitor: {vid} | Time: {ts}")
    print("=" * 60)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/group_entry_events.jsonl"
    parse_events(path)
