# PROMPT: "Generate a Pytest suite for a finite state machine handling spatial bounding box transitions. Create an adversarial test case where the input array of bounding boxes remains completely empty for 10,000 consecutive frames, ensuring the system does not raise a KeyError or NullType exception."
# CHANGES MADE: Modified the generated mock array to match the specific 15fps temporal offset requirements of the Purplle challenge, and added assertions to verify that zero ZONE_DWELL events are emitted during the empty period.

from datetime import datetime, timedelta, timezone
from uuid import UUID

from pipeline.layout import load_layout
from pipeline.tracker import Detection, RetailStateMachine


def test_empty_store_period_does_not_emit_or_crash():
    layout = load_layout("data/sample/store_layout.json")["STORE_BLR_002"]
    machine = RetailStateMachine(layout=layout, fps=15)
    total = []
    for _ in range(10_000):
        total.extend(machine.process([]))
    assert total == []


def test_zone_dwell_emitted_after_30_seconds():
    layout = load_layout("data/sample/store_layout.json")["STORE_BLR_002"]
    machine = RetailStateMachine(layout=layout, fps=15)
    start = datetime(2026, 3, 3, 14, 0, 0, tzinfo=timezone.utc)
    det1 = Detection("CAM_FLOOR_01", "1", (360, 420, 470, 760), 0.9, 0, start)
    det2 = Detection("CAM_FLOOR_01", "1", (360, 420, 470, 760), 0.88, 450, start + timedelta(seconds=31))
    events = machine.process([det1]) + machine.process([det2])
    assert any(e["event_type"] == "ZONE_DWELL" and e["dwell_ms"] == 30000 for e in events)


def test_entry_requires_confirmed_line_crossing():
    layout = load_layout("data/sample/store_layout.json")["STORE_BLR_002"]
    machine = RetailStateMachine(layout=layout, fps=15)
    start = datetime(2026, 3, 3, 14, 0, 0, tzinfo=timezone.utc)
    bbox = (860, 650, 960, 900)
    events = []
    for frame in range(2):
        events.extend(machine.process([Detection("CAM_ENTRY_01", "1", bbox, 0.9, frame, start + timedelta(seconds=frame / 15))]))
    assert not any(e["event_type"] == "ENTRY" for e in events)

    events.extend(machine.process([Detection("CAM_ENTRY_01", "1", bbox, 0.9, 3, start + timedelta(seconds=3 / 15))]))
    entry = [e for e in events if e["event_type"] == "ENTRY"]
    assert len(entry) == 1
    assert entry[0]["metadata"]["group_size"] == 1


def test_billing_abandon_is_delayed_until_grace_window():
    layout = load_layout("data/sample/store_layout.json")["STORE_BLR_002"]
    machine = RetailStateMachine(layout=layout, fps=15)
    start = datetime(2026, 3, 3, 14, 0, 0, tzinfo=timezone.utc)
    billing = Detection("CAM_FLOOR_01", "1", (1300, 430, 1420, 800), 0.9, 0, start)
    skincare = Detection("CAM_FLOOR_01", "1", (360, 420, 470, 760), 0.9, 1, start + timedelta(seconds=1))

    events = machine.process([billing]) + machine.process([skincare])
    assert any(e["event_type"] == "BILLING_QUEUE_JOIN" for e in events)
    assert not any(e["event_type"] == "BILLING_QUEUE_ABANDON" for e in events)

    stale = machine.flush_stale_queue(start + timedelta(minutes=6))
    assert any(e["event_type"] == "BILLING_QUEUE_ABANDON" for e in stale)


def test_cross_camera_reid_preserves_multi_target_identities():
    layout = load_layout("data/sample/store_layout.json")["STORE_BLR_002"]
    machine = RetailStateMachine(layout=layout, fps=15, reid_threshold=0.8)
    start = datetime(2026, 3, 3, 14, 0, 0, tzinfo=timezone.utc)
    emb_a = (1.0, 0.0, 0.0, 0.0)
    emb_b = (0.0, 1.0, 0.0, 0.0)

    machine.process(
        [
            Detection("CAM_ENTRY_01", "entry_a", (860, 650, 960, 900), 0.9, 0, start, embedding=emb_a),
            Detection("CAM_ENTRY_01", "entry_b", (980, 650, 1080, 900), 0.9, 0, start, embedding=emb_b),
        ]
    )
    entry_a = machine.track_to_visitor["CAM_ENTRY_01:entry_a"].visitor_id
    entry_b = machine.track_to_visitor["CAM_ENTRY_01:entry_b"].visitor_id
    assert entry_a != entry_b

    events = machine.process(
        [
            Detection("CAM_FLOOR_01", "floor_a", (360, 420, 470, 760), 0.91, 15, start + timedelta(seconds=1), embedding=emb_a),
            Detection("CAM_FLOOR_01", "floor_b", (760, 420, 870, 760), 0.91, 15, start + timedelta(seconds=1), embedding=emb_b),
        ]
    )
    assert machine.track_to_visitor["CAM_FLOOR_01:floor_a"].visitor_id == entry_a
    assert machine.track_to_visitor["CAM_FLOOR_01:floor_b"].visitor_id == entry_b
    assert {event["visitor_id"] for event in events if event["event_type"] == "ZONE_ENTER"} == {entry_a, entry_b}
    assert all(event["metadata"]["reid_matches"] == 1 for event in events if event["camera_id"] == "CAM_FLOOR_01")


def test_cross_camera_reid_does_not_match_stale_visitors():
    layout = load_layout("data/sample/store_layout.json")["STORE_BLR_002"]
    machine = RetailStateMachine(layout=layout, fps=15, reid_threshold=0.8, reid_time_window_seconds=30)
    start = datetime(2026, 3, 3, 14, 0, 0, tzinfo=timezone.utc)
    emb = (1.0, 0.0, 0.0, 0.0)

    machine.process([Detection("CAM_ENTRY_01", "entry_a", (860, 650, 960, 900), 0.9, 0, start, embedding=emb)])
    entry_id = machine.track_to_visitor["CAM_ENTRY_01:entry_a"].visitor_id
    machine.process([Detection("CAM_FLOOR_01", "floor_a", (360, 420, 470, 760), 0.91, 15, start + timedelta(minutes=2), embedding=emb)])
    assert machine.track_to_visitor["CAM_FLOOR_01:floor_a"].visitor_id != entry_id


def test_cross_camera_reid_rejects_weak_appearance_even_with_transition_bonus():
    layout = load_layout("data/sample/store_layout.json")["STORE_BLR_002"]
    machine = RetailStateMachine(layout=layout, fps=15, reid_threshold=0.8, reid_min_appearance=0.75)
    start = datetime(2026, 3, 3, 14, 0, 0, tzinfo=timezone.utc)

    machine.process([Detection("CAM_ENTRY_01", "entry_a", (860, 650, 960, 900), 0.9, 0, start, embedding=(1.0, 0.0, 0.0))])
    entry_id = machine.track_to_visitor["CAM_ENTRY_01:entry_a"].visitor_id
    machine.process(
        [
            Detection(
                "CAM_FLOOR_01",
                "floor_a",
                (360, 420, 470, 760),
                0.91,
                15,
                start + timedelta(seconds=1),
                embedding=(0.1, 0.99, 0.0),
            )
        ]
    )
    assert machine.track_to_visitor["CAM_FLOOR_01:floor_a"].visitor_id != entry_id


def test_pipeline_events_follow_required_schema_identifiers_and_metadata():
    layout = load_layout("data/sample/store_layout.json")["STORE_BLR_002"]
    machine = RetailStateMachine(layout=layout, fps=15)
    start = datetime(2026, 3, 3, 14, 0, 0, tzinfo=timezone.utc)
    events = machine.process(
        [
            Detection("CAM_FLOOR_01", "brand_track", (900, 420, 980, 760), 0.91, 0, start),
            Detection("CAM_FLOOR_01", "staff_track", (1300, 430, 1420, 800), 0.88, 0, start, is_staff=True),
        ]
    )
    brand_event = next(event for event in events if event["visitor_id"] == machine.track_to_visitor["CAM_FLOOR_01:brand_track"].visitor_id)
    staff_event = next(event for event in events if event["visitor_id"] == machine.track_to_visitor["CAM_FLOOR_01:staff_track"].visitor_id)

    assert UUID(brand_event["event_id"]).version == 4
    assert brand_event["event_type"] == "ZONE_ENTER"
    assert brand_event["timestamp"].endswith("Z")
    assert brand_event["store_id"] == "STORE_BLR_002"
    assert brand_event["metadata"]["sku_zone"] == "Faces Canada"
    assert staff_event["is_staff"] is True


def test_billing_queue_join_has_integer_queue_depth():
    layout = load_layout("data/sample/store_layout.json")["STORE_BLR_002"]
    machine = RetailStateMachine(layout=layout, fps=15)
    start = datetime(2026, 3, 3, 14, 0, 0, tzinfo=timezone.utc)
    events = machine.process(
        [
            Detection("CAM_FLOOR_01", "queue_a", (1300, 430, 1420, 800), 0.9, 0, start),
            Detection("CAM_FLOOR_01", "queue_b", (1450, 430, 1560, 800), 0.9, 0, start),
        ]
    )
    joins = [event for event in events if event["event_type"] == "BILLING_QUEUE_JOIN"]
    assert joins
    assert all(isinstance(event["metadata"]["queue_depth"], int) for event in joins)
    assert {event["metadata"]["queue_depth"] for event in joins} == {2}


def test_reentry_reuses_original_visitor_id_and_emits_reentry():
    layout = load_layout("data/sample/store_layout.json")["STORE_BLR_002"]
    machine = RetailStateMachine(layout=layout, fps=15)
    start = datetime(2026, 3, 3, 14, 0, 0, tzinfo=timezone.utc)
    inside = (860, 650, 960, 900)
    outside = (860, 100, 960, 400)
    events = []
    for index in range(3):
        events.extend(machine.process([Detection("CAM_ENTRY_01", "door_track", inside, 0.9, index, start + timedelta(seconds=index))]))
    visitor_id = next(event["visitor_id"] for event in events if event["event_type"] == "ENTRY")

    for index in range(3, 6):
        events.extend(machine.process([Detection("CAM_ENTRY_01", "door_track", outside, 0.9, index, start + timedelta(seconds=index))]))
    for index in range(6, 9):
        events.extend(machine.process([Detection("CAM_ENTRY_01", "door_track", inside, 0.9, index, start + timedelta(seconds=index))]))

    reentries = [event for event in events if event["event_type"] == "REENTRY"]
    assert reentries
    assert reentries[-1]["visitor_id"] == visitor_id


def test_simulated_detection_cli_writes_jsonl(tmp_path):
    from pipeline.detect import main

    out = tmp_path / "events.jsonl"
    code = main([
        "--simulate",
        "--layout",
        "data/sample/store_layout.json",
        "--api-url",
        "",
        "--out",
        str(out),
    ])
    assert code == 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0
    assert any('"event_type":"BILLING_QUEUE_JOIN"' in line for line in lines)


def test_replay_precomputed_detections_writes_events(tmp_path):
    import json
    from pipeline.replay_detections import main

    detections = tmp_path / "sample.detections.json"
    detections.write_text(
        json.dumps(
            {
                "store_id": "STORE_BLR_002",
                "camera_id": "CAM_ENTRY_01",
                "start_time": "2026-03-03T14:00:00Z",
                "fps": 15,
                "frames": [
                    {"i": 0, "boxes": [[860, 650, 960, 900, 0.9, 1]]},
                    {"i": 1, "boxes": [[860, 650, 960, 900, 0.9, 1]]},
                    {"i": 2, "boxes": [[860, 650, 960, 900, 0.9, 1]]},
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "events.jsonl"
    code = main(["--detections", str(detections), "--layout", "data/sample/store_layout.json", "--api-url", "", "--out", str(out)])
    assert code == 0
    assert '"event_type":"ENTRY"' in out.read_text(encoding="utf-8")


def test_staff_classifier_uniform_palette():
    import numpy as np
    from pipeline.staff import StaffClassifier

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = [180, 62, 72]  # BGR equivalent of RGB (72, 62, 180)
    classifier = StaffClassifier(uniform_rgb=((72, 62, 180),), threshold=10)
    is_staff, confidence = classifier.classify(frame, (10, 10, 90, 90))
    assert is_staff is True
    assert confidence > 0.9


def test_reid_embedding_helpers():
    import numpy as np
    from pipeline.reid import blend_embedding, cosine_similarity, crop_embedding, embedding_distance, set_deep_embedding_provider

    assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == 1.0
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == 0.0
    assert embedding_distance((1.0, 0.0), (1.0, 0.0)) == 0.0
    blended = blend_embedding((1.0, 0.0), (0.0, 1.0), samples=3)
    assert blended is not None
    assert len(blended) == 2

    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame[20:100, 30:90] = [30, 120, 220]
    embedding = crop_embedding(frame, (30, 20, 90, 100))
    assert embedding is not None
    assert len(embedding) > 300
    assert crop_embedding(frame, (0, 0, 0, 0)) is None

    set_deep_embedding_provider(lambda _frame, _bbox: (1.0, 0.0, 0.0, 0.0))
    try:
        deep_embedding = crop_embedding(frame, (30, 20, 90, 100))
    finally:
        set_deep_embedding_provider(None)
    assert deep_embedding is not None
    assert len(deep_embedding) == len(embedding) + 4
