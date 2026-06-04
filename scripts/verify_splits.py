import sys
from datetime import datetime, timezone, timedelta
from pipeline.layout import load_layout
from pipeline.tracker import RetailStateMachine, Detection
from pipeline.staff import StaffClassifier
from ultralytics import YOLO

def verify_splits(video_path, layout_path="data/real_store_layout.json", max_frames=600):
    layouts = load_layout(layout_path)
    layout = layouts["STORE_2"]
    machine = RetailStateMachine(layout=layout, fps=15.0)
    staff = StaffClassifier(layout.staff_uniform_rgb, layout.staff_uniform_hsv)
    
    model = YOLO("yolov8n.pt")
    
    # Run YOLO tracking as a stream, just like in detect.py
    results = model.track(
        source=video_path,
        stream=True,
        persist=True,
        classes=[0],
        verbose=False,
        conf=0.25,
        imgsz=960,
        tracker="bytetrack.yaml"
    )
    
    for frame_idx, r in enumerate(results):
        if max_frames is not None and frame_idx >= max_frames:
            break
            
        frame = getattr(r, "orig_img", None)
        timestamp = datetime.now(timezone.utc) + timedelta(seconds=frame_idx / 15.0)
        
        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.id is None:
            machine.process([])
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        detections = []
        
        for bbox, conf_value, tid in zip(xyxy, confs, ids):
            bbox_tuple = tuple(float(v) for v in bbox)
            is_staff, staff_conf = staff.classify(frame, bbox_tuple)
            detections.append(
                Detection(
                    camera_id="STORE2_CAM_ENTRY_2",
                    track_id=str(tid),
                    bbox=bbox_tuple,
                    confidence=float(min(conf_value, staff_conf if is_staff else conf_value)),
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    is_staff=is_staff,
                )
            )
        
        machine.process(detections)

    split_keys = [k for k in machine.track_to_visitor.keys() if "_split_" in k]
    print("=" * 60)
    print(f"Total internal tracks tracked: {len(machine.track_to_visitor)}")
    print(f"Total split tracks identified: {len(split_keys)}")
    print("=" * 60)
    for key in split_keys:
        state = machine.track_to_visitor[key]
        print(f"Track Key: {key:25s} -> Assigned Visitor ID: {state.visitor_id}")
    print("=" * 60)

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "data/entry_2.mp4"
    verify_splits(video)
