import cv2
import numpy as np

def analyze():
    img = cv2.imread("data/uploaded_markup.jpg")
    if img is None:
        print("Could not load image")
        return
        
    h, w, c = img.shape
    print(f"Image dimensions: {w}x{h}")
    
    # Scale factor from 1024x576 to 1920x1080 is 1.875
    scale = 1920.0 / w
    print(f"Scaling factor: {scale}")

    # Let's save a grid image with the scaled-down grid
    grid_img = img.copy()
    for x in range(300, w, 20):
        color = (0, 255, 0) if x % 100 == 0 else (200, 200, 200)
        cv2.line(grid_img, (x, 0), (x, h), color, 1)
        cv2.putText(grid_img, f"{x} ({int(x*scale)})", (x - 10, 30 + (x % 100 // 20) * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
    cv2.imwrite("data/layout_viz/markup_grid_scaled.png", grid_img)
    print("Saved scaled grid markup to data/layout_viz/markup_grid_scaled.png")

    # Let's search for the vertical lines in the scaled-down image
    # Scan from x=350 to w (which covers the MAKEUP area starting at 740/1.875 = 394)
    # y range from 100 to 450 (which covers the vertical lines in 1024x576)
    
    for color_name, (b_min, b_max, g_min, g_max, r_min, r_max) in {
        "yellow": (0, 100, 180, 256, 180, 256),
        "blue": (180, 256, 0, 150, 0, 100),
        "white": (220, 256, 220, 256, 220, 256),
        "black": (0, 30, 0, 30, 0, 30)
    }.items():
        y_start, y_end = 120, 450
        matches = []
        for x in range(350, w):
            col = img[y_start:y_end, x]
            cond = (col[:, 0] >= b_min) & (col[:, 0] <= b_max) & \
                   (col[:, 1] >= g_min) & (col[:, 1] <= g_max) & \
                   (col[:, 2] >= r_min) & (col[:, 2] <= r_max)
            count = np.sum(cond)
            # Adjust pixel threshold for the smaller y range
            if count > 20:
                matches.append((x, count))
                
        # Group adjacent columns
        if matches:
            groups = []
            curr_group = [matches[0]]
            for m in matches[1:]:
                if m[0] - curr_group[-1][0] <= 5:
                    curr_group.append(m)
                else:
                    groups.append(curr_group)
                    curr_group = [m]
            groups.append(curr_group)
            
            print(f"\nDetected {color_name} lines:")
            for g in groups:
                best_x = max(g, key=lambda x: x[1])[0]
                xs = [m[0] for m in g]
                scaled_best_x = int(best_x * scale)
                print(f"  x range: {xs[0]}-{xs[-1]} (best: {best_x}) -> Scaled: {scaled_best_x}")

if __name__ == "__main__":
    analyze()
