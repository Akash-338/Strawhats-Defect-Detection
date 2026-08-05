import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
# from models.morphology_fusion import MorphologyAwareYOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess(image: np.ndarray) -> np.ndarray:
    """Applies DSP and CLAHE preprocessing."""
    # Convert to LAB for CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    res = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return res

def detect_single(image_path: str, weights: str, conf: float, save_dir: str):
    logging.info(f"Loading image {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read {image_path}")
        
    preprocessed = preprocess(image)
    
    # model = MorphologyAwareYOLO(weights)
    # results = model.predict(preprocessed, conf=conf)
    
    # Mocking results
    logging.info("Running morphology fusion inference...")
    
    # Dummy morphology features
    morph_features = {
        "crack_1": {"Area": 150, "Aspect Ratio": 0.2, "Eccentricity": 0.98},
        "scratch_1": {"Area": 45, "Aspect Ratio": 0.8, "Eccentricity": 0.85}
    }
    
    print("\n--- Detected Morphology Features ---")
    for def_id, feats in morph_features.items():
        print(f"{def_id}: {feats}")
    print("------------------------------------\n")
    
    out_img = preprocessed.copy()
    # Draw mock bboxes
    cv2.rectangle(out_img, (50, 50), (200, 80), (0, 0, 255), 2)
    cv2.putText(out_img, "crack 0.92", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.putText(out_img, "Defects: 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"result_{Path(image_path).name}"
    
    cv2.imwrite(str(out_path), out_img)
    logging.info(f"Saved annotated image to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", type=str, default="runs/detect", help="Save directory")
    args = parser.parse_args()
    
    detect_single(args.image, args.weights, args.conf, args.save)
