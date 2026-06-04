from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import structlog
import time
import uuid
import sqlite3
from pydantic import BaseModel

from .db import init_db
from .pos_loader import load_pos_data
from .models import EventPayload
from .ingestion import ingest_events
from .metrics import get_metrics, get_funnel, get_heatmap
from .anomalies import get_anomalies

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    init_db()
    logger.info("Loading POS data...")
    load_pos_data()
    yield
    logger.info("Shutting down...")

app = FastAPI(title="Apex Retail Intelligence API", lifespan=lifespan)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    request.state.trace_id = trace_id
    
    start_time = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    
    store_id = getattr(request.state, "store_id", None)
    if not store_id:
        path_parts = request.url.path.strip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] == "stores":
            store_id = path_parts[1]
            
    event_count = getattr(request.state, "event_count", None)
    
    logger.info(
        "Request processed",
        trace_id=trace_id,
        store_id=store_id,
        endpoint=request.url.path,
        latency_ms=latency_ms,
        event_count=event_count,
        status_code=response.status_code
    )
    response.headers["X-Trace-ID"] = trace_id
    return response

@app.exception_handler(sqlite3.Error)
def sqlite_exception_handler(request: Request, exc: sqlite3.Error):
    logger.error("Database error occurred", exc_info=exc, path=request.url.path)
    return JSONResponse(
        status_code=503,
        content={
            "status": "error",
            "message": "Service Unavailable: Database is temporarily unavailable or locked.",
            "detail": str(exc)
        }
    )

class IngestResponse(BaseModel):
    status: str
    ingested: int
    duplicates: int
    errors: List[Dict[str, Any]]

@app.get("/health")
def health_check():
    try:
        from .db import get_db
        with get_db() as conn:
            row = conn.execute("SELECT MAX(timestamp) AS ts FROM events").fetchone()
            latest_ts = row["ts"] if row else None
            
        if latest_ts:
            from datetime import datetime, timezone
            from .metrics import parse_dt
            dt = parse_dt(latest_ts)
            if dt:
                now = datetime.now(timezone.utc)
                if (now - dt).total_seconds() > 600:  # 10 minutes
                    return {"status": "ok", "warning": "STALE_FEED", "latest_event": latest_ts}
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database unavailable: {str(e)}")

@app.post("/events/ingest", response_model=IngestResponse)
@app.post("/ingest", response_model=IngestResponse)
def ingest(events: List[EventPayload], request: Request):
    request.state.event_count = len(events)
    if events:
        request.state.store_id = events[0].store_id
    try:
        ingested, duplicates, errors = ingest_events(events)
        status = "success" if not errors else "partial_success"
        return IngestResponse(status=status, ingested=ingested, duplicates=duplicates, errors=errors)
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error during ingestion")

@app.get("/stores/{store_id}/metrics")
def store_metrics(store_id: str, request: Request, start: str | None = None, end: str | None = None):
    request.state.store_id = store_id
    return get_metrics(store_id, start, end)

@app.get("/stores/{store_id}/funnel")
def store_funnel(store_id: str, request: Request, start: str | None = None, end: str | None = None):
    request.state.store_id = store_id
    return get_funnel(store_id, start, end)

@app.get("/stores/{store_id}/heatmap")
def store_heatmap(store_id: str, request: Request, start: str | None = None, end: str | None = None):
    request.state.store_id = store_id
    return get_heatmap(store_id, start, end)

@app.get("/stores/{store_id}/anomalies")
def store_anomalies(store_id: str, request: Request, start: str | None = None, end: str | None = None):
    request.state.store_id = store_id
    return get_anomalies(store_id, start, end)
