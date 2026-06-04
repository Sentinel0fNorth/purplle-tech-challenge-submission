import sys
from datetime import datetime, timezone, timedelta
from pipeline.layout import load_layout
from pipeline.tracker import RetailStateMachine, Detection
from pipeline.staff import StaffClassifier
from pipeline.layout import foot_point, point_in_polygon
from ultralytics import YOLO

def debug_splits(video_path, layout_path="data/real_store_layout.json", max_frames=1000):
    layouts = load_layout(layout_path)
    layout = layouts["STORE_2"]
    machine = RetailStateMachine(layout=layout, fps=15.0)
    
    # Let's print the entry zone details
    entry_zone = next((z for z in layout.zones if z.zone_id == "ENTRY"), None)
    print("ENTRY zone polygon:", entry_zone.polygon if entry_zone else None)
    
    model = YOLO("yolov8n.pt")
    
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
            
        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.id is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        
        for bbox, tid in zip(xyxy, ids):
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            ar = w / h if h > 0 else 0
            foot = foot_point(bbox)
            in_polygon = point_in_polygon(foot, entry_zone.polygon) if entry_zone else False
            
            if ar > 0.80:
                print(f"Frame {frame_idx:3d} | Track {tid:2d} | AR={ar:.3f} | Foot: ({foot[0]:.1f}, {foot[1]:.1f}) | In Entry Poly: {in_polygon}")

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "data/entry_2.mp4"
    debug_splits(video)
