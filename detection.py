import os
import cv2
import numpy as np
from ultralytics import YOLO

def classify_characters(input_dir, job_id):
    output_dir = f'workspace/classified/{job_id}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLOv8 model
    model = YOLO('models/yolov8/best.pt')
    
    # Process each extracted character
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            
            # Preprocess: remove padding
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(thresh)
            
            if coords is None:
                continue  # skip completely black images
            
            # Crop to content
            x, y, w, h = cv2.boundingRect(coords)
            cropped = img[y:y+h, x:x+w]
            
            # Run inference on cropped image
            results = model(cropped)
            boxes = results[0].boxes
            
            if len(boxes) == 0:
                continue  # skip if no prediction
            
            class_id = int(boxes[0].cls.item())
            class_name = model.names[class_id]
            
            # Create class directory if needed
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Save to class directory
            output_path = os.path.join(class_dir, filename)
            cv2.imwrite(output_path, cropped)
    
    return output_dir