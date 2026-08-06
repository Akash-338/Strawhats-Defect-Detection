"""
scripts/generate_test_samples.py
=================================
Generate annotated predictions on the 15% held-out test split of steel dataset.
Saves annotated images into demo_samples/test_split/
"""

import os
import cv2
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

def main():
    weights_path = ROOT / 'runs' / 'detect' / 'runs' / 'steel' / 'weights' / 'best.pt'
    if not weights_path.exists():
        weights_path = ROOT / 'runs' / 'detect' / 'steel' / 'weights' / 'best.pt'
        
    test_images_dir = ROOT / 'data' / 'processed' / 'steel_unified' / 'val' / 'images'
    output_dir = ROOT / 'demo_samples' / 'test_split'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Steel YOLO weights from {weights_path}...")
    model = YOLO(str(weights_path))
    
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.bmp")) + list(test_images_dir.glob("*.png"))
    print(f"Found {len(image_files)} test set images in {test_images_dir}")
    
    processed = 0
    for img_path in image_files[:40]: # First 40 test samples
        res = model(str(img_path), conf=0.20, verbose=False)[0]
        annotated = res.plot()
        
        save_path = output_dir / f"test_pred_{img_path.name}"
        cv2.imwrite(str(save_path), annotated)
        processed += 1
        
    print(f"✅ Successfully saved {processed} annotated test set samples to {output_dir}")

if __name__ == '__main__':
    main()
