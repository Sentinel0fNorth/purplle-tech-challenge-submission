import uuid
import json
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator

NAMESPACE_EVENTS = uuid.UUID('12345678-1234-5678-1234-567812345678')

class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: Optional[int] = None

    model_config = {"extra": "allow"}

class EventPayload(BaseModel):
    event_id: Optional[str] = None
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: str
    timestamp: str
    zone_id: Optional[str] = None
    dwell_ms: int = 0
    is_staff: bool = False
    confidence: float = 1.0
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    @model_validator(mode='before')
    @classmethod
    def bilingual_adapter(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
            
        # Map store_id
        if 'store_id' not in data and 'store_code' in data:
            data['store_id'] = data['store_code']
            
        # Normalize store_id (e.g., store_1076 -> ST1076)
        if data.get('store_id', '').startswith('store_'):
            data['store_id'] = 'ST' + data['store_id'].split('_')[1]
            
        # Map timestamp
        if 'timestamp' not in data:
            data['timestamp'] = data.get('event_timestamp') or data.get('event_time') or data.get('queue_join_ts') or ""
        
        # Ensure timestamp has Z if it's naive ISO
        if data['timestamp'] and not data['timestamp'].endswith('Z'):
            data['timestamp'] = data['timestamp'] + 'Z'

        # Map visitor_id
        if 'visitor_id' not in data:
            vid = data.get('id_token') or data.get('track_id')
            if vid is not None:
                data['visitor_id'] = str(vid)
            else:
                data['visitor_id'] = "UNKNOWN"

        # Map event_type
        etype = data.get('event_type', '').upper()
        if etype == 'ZONE_ENTERED':
            etype = 'ZONE_ENTER'
        elif etype == 'ZONE_EXITED':
            etype = 'ZONE_EXIT'
        elif etype == 'QUEUE_COMPLETED':
            etype = 'BILLING_QUEUE_JOIN'
        elif etype == 'QUEUE_ABANDONED':
            etype = 'BILLING_QUEUE_ABANDON'
        data['event_type'] = etype

        # Populate metadata if not present
        if 'metadata' not in data:
            meta = {}
            if 'queue_position_at_join' in data:
                meta['queue_depth'] = data['queue_position_at_join']
            if 'zone_name' in data:
                meta['sku_zone'] = data['zone_name']
            data['metadata'] = meta

        # Generate event_id if missing
        if not data.get('event_id'):
            # Combine stable fields to generate a deterministic UUIDv5
            stable_string = f"{data.get('store_id')}:{data.get('camera_id')}:{data.get('visitor_id')}:{data.get('event_type')}:{data.get('timestamp')}"
            data['event_id'] = str(uuid.uuid5(NAMESPACE_EVENTS, stable_string))
            
        return data
        
    @model_validator(mode='after')
    def validate_constraints(self) -> 'EventPayload':
        if self.event_type in ('ENTRY', 'EXIT', 'REENTRY') and self.zone_id is not None:
            self.zone_id = None
        if self.event_type == 'ZONE_DWELL' and self.dwell_ms <= 0:
            # Enforce positive dwell for ZONE_DWELL
            pass
        return self
