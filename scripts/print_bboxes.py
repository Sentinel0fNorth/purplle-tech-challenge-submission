import sys
from ultralytics import YOLO
import cv2

def analyze_video(video_path, model_path="yolov8n.pt", max_frames=300):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video {video_path}")
        return

    frame_idx = 0
    wide_boxes_count = 0
    total_boxes_count = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, classes=[0], verbose=False, conf=0.25)
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or boxes.id is None:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            
            for bbox, tid in zip(xyxy, ids):
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                ar = w / h if h > 0 else 0
                total_boxes_count += 1
                
                if ar > 0.70:
                    print(f"Frame {frame_idx:3d} | Track {tid:2d} | Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) | W={w:.1f}, H={h:.1f} | AR={ar:.3f}")
                    if ar > 0.85:
                        wide_boxes_count += 1

        frame_idx += 1

    cap.release()
    print("=" * 60)
    print(f"Total frames processed: {frame_idx}")
    print(f"Total person bounding boxes: {total_boxes_count}")
    print(f"Wide bounding boxes (AR > 0.85): {wide_boxes_count}")
    print("=" * 60)

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "data/entry_2.mp4"
    analyze_video(video)
