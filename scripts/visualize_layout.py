import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path

def find_video_file(filename, search_dirs):
    for s_dir in search_dirs:
        for root, dirs, files in os.walk(s_dir):
            if filename in files:
                return os.path.join(root, filename)
    return None

def draw_layout(layout_path="data/real_store_layout.json", output_dir="data/layout_viz"):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(layout_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    stores = data.get("stores", [data])
    
    # Directories on the host / container to search for videos
    search_dirs = [
        "/app/data", 
        "/workspace/Original", 
        "/workspace",
        "data",
        "../Original",
        "../../Original"
    ]
    
    # Colors for different zones
    colors = {
        "ENTRY": (0, 255, 255),       # Yellow
        "BILLING": (0, 0, 255),       # Red
        "SKINCARE": (255, 0, 0),      # Blue
        "MAKEUP": (255, 0, 255),      # Magenta
        "HAIRCARE": (0, 255, 0),      # Green
        "DEFAULT": (255, 255, 255)    # White
    }
    
    for store in stores:
        store_id = store.get("store_id")
        print(f"\nProcessing Store: {store_id}")
        
        # Build map of camera_id to its zones
        cam_to_zones = {}
        for zone in store.get("zones", []):
            zone_id = zone.get("zone_id")
            polygon = [(int(p[0]), int(p[1])) for p in zone.get("polygon", [])]
            for cam_id in zone.get("camera_ids", []):
                if cam_id not in cam_to_zones:
                    cam_to_zones[cam_id] = []
                cam_to_zones[cam_id].append((zone_id, polygon))
                
        # Process each camera
        for cam in store.get("cameras", []):
            cam_id = cam.get("camera_id")
            video_path_raw = cam.get("video_path")
            entry_line_y = cam.get("entry_line_y")
            
            print(f"  * Camera: {cam_id}")
            
            # Find the actual video file
            video_file = None
            if video_path_raw:
                filename = os.path.basename(video_path_raw)
                video_file = find_video_file(filename, search_dirs)
                
            frame = None
            if video_file:
                print(f"    Found video: {video_file}")
                cap = cv2.VideoCapture(video_file)
                if cap.isOpened():
                    # Seek to frame 150 (approx 10s) to get a bright, clear frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
                    ret, frame = cap.read()
                    cap.release()
            else:
                print(f"    Warning: Could not find video file for {video_path_raw}")
                
            # If no frame could be read, use a black canvas
            if frame is None:
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                print("    Creating blank canvas for layout drawing.")
                
            h, w = frame.shape[:2]
            
            # Draw zones for this camera
            zones = cam_to_zones.get(cam_id, [])
            for zone_id, poly in zones:
                if not poly:
                    continue
                color = colors.get(zone_id, colors["DEFAULT"])
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                
                # Draw translucent polygon overlay
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                
                # Draw border line
                cv2.polylines(frame, [pts], True, color, 3)
                
                # Write Zone ID label at centroid
                moments = cv2.moments(pts)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    cx, cy = poly[0]
                
                cv2.putText(frame, zone_id, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
                
            # Draw threshold line for Entry cameras
            is_entry_cam = "entry" in cam.get("role", "").lower() or "entry" in cam_id.lower()
            if is_entry_cam:
                entry_line = cam.get("entry_line")
                if entry_line is not None:
                    p1 = (int(entry_line[0][0]), int(entry_line[0][1]))
                    p2 = (int(entry_line[1][0]), int(entry_line[1][1]))
                    cv2.line(frame, p1, p2, (0, 0, 255), 4)
                    label_text = "ENTRY THRESHOLD (diagonal)"
                    cv2.putText(frame, label_text, (50, min(p1[1], p2[1]) - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    line_y = int(entry_line_y) if entry_line_y is not None else 540
                    # Draw threshold line across the frame width
                    cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 4)
                    # Draw label
                    label_text = f"ENTRY THRESHOLD (y={line_y})"
                    if entry_line_y is None:
                        label_text += " [DEFAULT MIDLINE]"
                    cv2.putText(frame, label_text, (50, line_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
                
            # Save the annotated layout frame
            out_path = os.path.join(output_dir, f"{store_id}_{cam_id}.png")
            cv2.imwrite(out_path, frame)
            print(f"    Saved layout visualization to: {out_path}")

if __name__ == "__main__":
    layout = sys.argv[1] if len(sys.argv) > 1 else "data/real_store_layout.json"
    draw_layout(layout)
