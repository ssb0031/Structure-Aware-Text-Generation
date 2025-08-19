import os
import cv2
import numpy as np
from PIL import Image

def create_masks(input_dir, job_id):
    output_dir = f'workspace/masks/{job_id}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each class directory
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if os.path.isdir(class_dir):
            output_class_dir = os.path.join(output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Process each image
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, filename)
                    
                    # Convert to binary mask
                    mask = image_to_binary_mask(img_path)
                    
                    # Save mask
                    output_path = os.path.join(output_class_dir, filename)
                    mask.save(output_path)
    
    return output_dir

def image_to_binary_mask(image_path):
    # Load image and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Threshold to create binary mask (invert black/white)
    _, binary_mask = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Convert to PIL image
    return Image.fromarray(binary_mask)