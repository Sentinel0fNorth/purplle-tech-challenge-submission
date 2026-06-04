# PROMPT: Write a pytest suite for verifying HSV staff classification, group-entry track splitting, and cross-camera spatio-temporal Re-ID with frame-freeze bypass.
# CHANGES MADE: Customized mock frame generation and verified bounding box values for Store 1 and Store 2 layout configurations.

import pytest
import cv2
from datetime import datetime, timezone, timedelta
import numpy as np
from pipeline.staff import StaffClassifier
from pipeline.tracker import Detection, RetailStateMachine
from pipeline.layout import StoreLayout, Zone, Camera

def test_hsv_staff_classification():
    # Store 1 uniform: dark blue/black (HSV H=103, S=28, V=55)
    classifier = StaffClassifier(uniform_hsv=((103, 28, 55),))
    
    # Generate mock frame with torso region matching uniform color
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    # Paint the whole frame with dark blue (HSV (103, 28, 55))
    hsv_color = np.array([[[103, 28, 55]]], dtype=np.uint8)
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    frame[:, :] = bgr_color
    
    # Torso bounding box
    bbox = (50, 50, 150, 300)
    is_staff, conf = classifier.classify(frame, bbox)
    assert is_staff is True
    assert conf >= 0.51

    # Generate a frame with different color (e.g. pink HSV (144, 154, 81))
    # It should not match Store 1 uniform
    hsv_pink = np.array([[[144, 154, 81]]], dtype=np.uint8)
    bgr_pink = cv2.cvtColor(hsv_pink, cv2.COLOR_HSV2BGR)[0][0]
    frame[:, :] = bgr_pink
    is_staff_pink, conf_pink = classifier.classify(frame, bbox)
    assert is_staff_pink is False

def test_group_entry_splitting():
    # Setup layout with ENTRY zone
    layout = StoreLayout(
        store_id="STORE_2",
        zones=(
            Zone(
                zone_id="ENTRY",
                polygon=[(0, 0), (1920, 0), (1920, 1080), (0, 1080)],
                camera_ids=("STORE2_CAM_ENTRY_1",)
            ),
        ),
        cameras=(
            Camera(camera_id="STORE2_CAM_ENTRY_1", role="Entry/Exit threshold"),
        )
    )
    machine = RetailStateMachine(layout=layout)
    
    # Process multiple consecutive detections to confirm threshold crossing (line_confirm_frames=3)
    events = []
    for f in range(1, 5):
        det = Detection(
            camera_id="STORE2_CAM_ENTRY_1",
            track_id="merged_track",
            bbox=(200, 400, 600, 600),
            confidence=0.90,
            frame_index=f,
            timestamp=datetime.now(timezone.utc)
        )
        events.extend(machine.process([det]))
    
    # The split should create 4 virtual tracks (W / (H*0.45) = 400 / 90 = 4.4 -> 4 tracks)
    # Each will cross the entry line inside the ENTRY zone, generating ENTRY events
    entry_events = [e for e in events if e["event_type"] in ("ENTRY", "REENTRY")]
    assert len(entry_events) >= 2

def test_spatio_temporal_reid_freeze():
    # Setup layout with zones
    layout = StoreLayout(
        store_id="STORE_2",
        zones=(
            Zone(zone_id="ENTRY", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)], camera_ids=("CAM_1",)),
            Zone(zone_id="BILLING", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)], camera_ids=("CAM_2",))
        ),
        cameras=(
            Camera(camera_id="CAM_1", role="Entry/Exit threshold"),
            Camera(camera_id="CAM_2", role="Billing counter area")
        )
    )
    machine = RetailStateMachine(layout=layout)
    
    # Visitor seen on CAM_1 (Entry)
    t1 = datetime(2026, 3, 3, 12, 0, 0, tzinfo=timezone.utc)
    # Bounding box coordinates inside ENTRY zone of CAM_1
    det1 = Detection(
        camera_id="CAM_1",
        track_id="track_1",
        bbox=(10, 10, 50, 90),
        confidence=0.90,
        frame_index=0,
        timestamp=t1,
        embedding=tuple([1.0] + [0.0] * 299)  # mock 300-d normalized embedding
    )
    machine.process([det1])
    
    # 1. Fast transition (e.g. 1s later) on CAM_2 with DIFFERENT embedding -> Should NOT match (rejected due to speed constraint)
    t2 = t1 + timedelta(seconds=1)
    det2_diff = Detection(
        camera_id="CAM_2",
        track_id="track_2",
        bbox=(10, 10, 50, 90),
        confidence=0.90,
        frame_index=15,
        timestamp=t2,
        embedding=tuple([0.0, 1.0] + [0.0] * 298)  # cosine similarity = 0.0
    )
    events = machine.process([det2_diff])
    # Verify a new session was created (different visitor_id)
    assert machine.track_to_visitor["CAM_2:track_2"].visitor_id != machine.track_to_visitor["CAM_1:track_1"].visitor_id
    
    # 2. Fast transition (1s later) on CAM_2 with EXTREMELY SIMILAR embedding -> Should MATCH (freeze bypass)
    det3_same = Detection(
        camera_id="CAM_2",
        track_id="track_3",
        bbox=(10, 10, 50, 90),
        confidence=0.90,
        frame_index=15,
        timestamp=t2,
        embedding=tuple([0.95] + [0.0] * 299)  # cosine similarity = 0.95 (strong match)
    )
    machine.process([det3_same])
    # Verify it matched track_1 (same visitor_id)
    assert machine.track_to_visitor["CAM_2:track_3"].visitor_id == machine.track_to_visitor["CAM_1:track_1"].visitor_id
