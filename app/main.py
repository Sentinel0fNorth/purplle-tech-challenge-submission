from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
import structlog
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

class IngestResponse(BaseModel):
    status: str
    ingested: int
    duplicates: int
    errors: List[Dict[str, Any]]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestResponse)
def ingest(events: List[EventPayload]):
    try:
        ingested, duplicates, errors = ingest_events(events)
        status = "success" if not errors else "partial_success"
        return IngestResponse(status=status, ingested=ingested, duplicates=duplicates, errors=errors)
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error during ingestion")

@app.get("/stores/{store_id}/metrics")
def store_metrics(store_id: str, start: str | None = None, end: str | None = None):
    return get_metrics(store_id, start, end)

@app.get("/stores/{store_id}/funnel")
def store_funnel(store_id: str, start: str | None = None, end: str | None = None):
    return get_funnel(store_id, start, end)

@app.get("/stores/{store_id}/heatmap")
def store_heatmap(store_id: str, start: str | None = None, end: str | None = None):
    return get_heatmap(store_id, start, end)

@app.get("/stores/{store_id}/anomalies")
def store_anomalies(store_id: str, start: str | None = None, end: str | None = None):
    return get_anomalies(store_id, start, end)
