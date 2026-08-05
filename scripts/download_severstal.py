"""
download_severstal.py
Downloads Severstal Steel Defect Detection from Kaggle, processes RLE masks,
extracts bounding boxes, crops patches, and outputs YOLO-format labels.

Kaggle API Setup Instructions:
1. Install Kaggle CLI: pip install kaggle
2. Get API token: Go to Kaggle.com -> Settings -> Create New Token (kaggle.json)
3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<User>\\.kaggle\\ (Windows)
4. Accept the competition rules on Kaggle before downloading.
"""

import argparse
import logging
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Severstal classes: 1: patches, 2: scratches, 3: crazing, 4: pitted_surface
# Target classes for YOLO: {0: crazing, 1: inclusion, 2: patches, 3: pitted_surface, 4: rolled_in_scale, 5: scratches}
CLASS_MAPPING = {
    1: 2,  # patches
    2: 5,  # scratches
    3: 0,  # crazing
    4: 3   # pitted_surface
}

def download_dataset(data_dir: Path) -> None:
    """Download Severstal dataset using Kaggle CLI."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "severstal-steel-defect-detection.zip"
    
    if not zip_path.exists():
        logger.info("Downloading Severstal from Kaggle...")
        try:
            subprocess.run([
                "kaggle", "competitions", "download", 
                "-c", "severstal-steel-defect-detection", 
                "-p", str(data_dir)
            ], check=True)
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            logger.info("Please ensure Kaggle API is set up and you have accepted the competition rules.")
            return

    logger.info("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc="Extracting"):
            zip_ref.extract(member, data_dir)

def rle_to_mask(rle: str, width: int = 1600, height: int = 256) -> np.ndarray:
    """Convert RLE string to binary mask."""
    mask = np.zeros(width * height, dtype=np.uint8)
    if pd.isna(rle) or not isinstance(rle, str):
        return mask.reshape((height, width))
    
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape((width, height)).T

def process_data(data_dir: Path, output_dir: Path) -> None:
    """Process RLE masks, find bboxes, crop images, and save YOLO labels."""
    csv_path = data_dir / "train.csv"
    img_dir = data_dir / "train_images"
    
    if not csv_path.exists() or not img_dir.exists():
        logger.error(f"Data not found in {data_dir}. Did download succeed?")
        return
        
    out_img_dir = output_dir / "images"
    out_lbl_dir = output_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    if 'ImageId_ClassId' in df.columns:
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        df['ClassId'] = df['ImageId_ClassId'].apply(lambda x: int(x.split('_')[1]))
        
    grouped = df.groupby('ImageId')
    
    logger.info("Processing images and extracting patches...")
    for img_id, group in tqdm(grouped, total=len(grouped)):
        img_path = img_dir / img_id
        if not img_path.exists():
            continue
            
        defects = group.dropna(subset=['EncodedPixels'])
        if defects.empty:
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        for idx, row in defects.iterrows():
            orig_class = int(row['ClassId'])
            if orig_class not in CLASS_MAPPING:
                continue
            
            yolo_class = CLASS_MAPPING[orig_class]
            mask = rle_to_mask(row['EncodedPixels'], w, h)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c_idx, contour in enumerate(contours):
                x, y, bw, bh = cv2.boundingRect(contour)
                if bw < 5 or bh < 5:
                    continue
                    
                cx = x + bw // 2
                cy = y + bh // 2
                
                patch_size = 200
                half_patch = patch_size // 2
                
                px1 = max(0, cx - half_patch)
                py1 = max(0, cy - half_patch)
                px2 = min(w, px1 + patch_size)
                py2 = min(h, py1 + patch_size)
                
                if px2 - px1 < patch_size:
                    if px1 == 0: px2 = min(w, patch_size)
                    else: px1 = max(0, px2 - patch_size)
                if py2 - py1 < patch_size:
                    if py1 == 0: py2 = min(h, patch_size)
                    else: py1 = max(0, py2 - patch_size)
                    
                patch = img[py1:py2, px1:px2]
                
                new_x = max(0, x - px1)
                new_y = max(0, y - py1)
                new_bw = min(bw, px2 - px1 - new_x)
                new_bh = min(bh, py2 - py1 - new_y)
                
                patch_h, patch_w = patch.shape[:2]
                if patch_h == 0 or patch_w == 0:
                    continue
                
                norm_cx = (new_x + new_bw / 2) / patch_w
                norm_cy = (new_y + new_bh / 2) / patch_h
                norm_w = new_bw / patch_w
                norm_h = new_bh / patch_h
                
                patch_name = f"{Path(img_id).stem}_{orig_class}_{c_idx}"
                cv2.imwrite(str(out_img_dir / f"{patch_name}.jpg"), patch)
                
                with open(out_lbl_dir / f"{patch_name}.txt", "w") as f:
                    f.write(f"{yolo_class} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Download and process Severstal dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/Severstal"), help="Download directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/severstal"), help="Output directory")
    args = parser.parse_args()
    
    download_dataset(args.data_dir)
    process_data(args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()
