CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    visitor_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    zone_id TEXT,
    dwell_ms INTEGER NOT NULL DEFAULT 0,
    is_staff BOOLEAN NOT NULL DEFAULT 0,
    confidence REAL NOT NULL,
    metadata TEXT,
    queue_depth INTEGER,
    sku_zone TEXT
);

CREATE TABLE IF NOT EXISTS visitor_sessions (
    session_id TEXT PRIMARY KEY,
    visitor_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    is_converted BOOLEAN NOT NULL DEFAULT 0,
    is_staff BOOLEAN NOT NULL DEFAULT 0,
    session_seq INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS pos_transactions (
    order_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    total_amount REAL NOT NULL,
    brands TEXT
);

CREATE TABLE IF NOT EXISTS stitched_visitors (
    store_id TEXT NOT NULL,
    original_visitor_id TEXT NOT NULL,
    stitched_visitor_id TEXT NOT NULL,
    PRIMARY KEY (store_id, original_visitor_id)
);

CREATE INDEX IF NOT EXISTS idx_events_store_ts ON events(store_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_store_type ON events(store_id, event_type);
CREATE INDEX IF NOT EXISTS idx_events_store_camera_ts ON events(store_id, camera_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_store_visitor ON visitor_sessions(store_id, visitor_id);
CREATE INDEX IF NOT EXISTS idx_pos_transactions_store_ts ON pos_transactions(store_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_queue_depth ON events(queue_depth);
CREATE INDEX IF NOT EXISTS idx_events_sku_zone ON events(sku_zone);

