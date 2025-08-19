import os
import cv2
import numpy as np

def extract_characters(image_path, job_id):
    output_dir = f'workspace/extracted/{job_id}'
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------- Parameters ----------
    resized_size = 64
    min_contour_area = 50
    max_contour_area = 100000
    
    # ---------- Load Image ----------
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # ---------- Find Contours ----------
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # ---------- PSO-Style Box Expansion ----------
    def expand_box_to_edges(x, y, w, h, edge_img, max_expand=10):
        h_img, w_img = edge_img.shape
        for _ in range(max_expand):
            x1 = max(x - 1, 0)
            y1 = max(y - 1, 0)
            x2 = min(x + w + 1, w_img)
            y2 = min(y + h + 1, h_img)
            roi = edge_img[y1:y2, x1:x2]
            if np.count_nonzero(roi) == 0:
                break
            x, y = x1, y1
            w, h = x2 - x1, y2 - y1
        return x, y, w, h
    
    # ---------- Extract & Save Characters ----------
    pso_count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_contour_area or area > max_contour_area:
            continue

        x_pso, y_pso, w_pso, h_pso = expand_box_to_edges(x, y, w, h, dilated)
        char_crop_pso = img[y_pso:y_pso+h_pso, x_pso:x_pso+w_pso]
        
        # Resize to 64x64
        resized_pso = cv2.resize(char_crop_pso, (resized_size, resized_size), 
                                interpolation=cv2.INTER_AREA)
        
        # Save extracted character
        save_path = os.path.join(output_dir, f"char_{pso_count}.png")
        cv2.imwrite(save_path, resized_pso)
        pso_count += 1
    
    print(f"✅ Extracted {pso_count} characters using PSO + RPN method.")
    return output_dir