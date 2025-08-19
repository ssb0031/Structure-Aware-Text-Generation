import os
import cv2
import numpy as np
import cma

def augment_characters(input_dir, job_id):
    output_dir = f'workspace/augmented/{job_id}'
    os.makedirs(output_dir, exist_ok=True)
    
    # -------- CONFIG --------
    MAX_ROTATIONS = 4
    TARGET_ANGLES = [-8, -4, 8, 4]
    ANGLE_BOUNDS = [-15, 15]
    
    # -------- Rotate Image --------
    def rotate_image(img, angle):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    # -------- Fitness Function --------
    def evaluate_by_angle_distance(angles):
        angles = np.sort(angles)
        target = np.sort(TARGET_ANGLES)
        return np.mean(np.abs(angles - target))  # Mean absolute error
    
    # -------- Process each character class --------
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Create output directory for this class
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Process each character image in the class
        for file_name in os.listdir(class_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')) and "_cmaes_" not in file_name:
                img_path = os.path.join(class_path, file_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ Skipped unreadable image: {img_path}")
                    continue
                
                # --- CMA-ES Setup ---
                x0 = np.zeros(MAX_ROTATIONS)
                sigma0 = 6.0
                opts = {"bounds": ANGLE_BOUNDS, "maxiter": 100}
                es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
                
                # Run optimization
                while not es.stop():
                    solutions = es.ask()
                    fitnesses = [evaluate_by_angle_distance(sol) for sol in solutions]
                    es.tell(solutions, fitnesses)
                
                # Get best angles
                best_angles = es.result.xbest
                best_angles = [round(a, 1) for a in best_angles]
                
                # --- Save Rotated Images ---
                base_name = os.path.splitext(file_name)[0]
                for angle in best_angles:
                    rotated_img = rotate_image(img, angle)
                    save_name = f"{base_name}_cmaes_{angle:+.1f}.png"
                    save_path = os.path.join(output_class_dir, save_name)
                    cv2.imwrite(save_path, rotated_img)
    
    return output_dir