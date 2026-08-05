"""
split_dataset.py
Performs a stratified 70/15/15 split (Train/Val/Test) of the dataset.
"""

import argparse
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_image_classes(label_path: Path) -> List[int]:
    """Extract unique classes present in a YOLO label file."""
    classes = set()
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
    return list(classes)

def stratified_split(input_dir: Path, output_dir: Path, train_ratio: float = 0.7, val_ratio: float = 0.15) -> None:
    """Stratify and split dataset."""
    img_dir = input_dir / "images"
    lbl_dir = input_dir / "labels"
    
    if not img_dir.exists() or not lbl_dir.exists():
        logger.error(f"Input directories not found in {input_dir}")
        return
        
    images = list(img_dir.glob("*.jpg"))
    
    class_to_images: Dict[int, List[Path]] = defaultdict(list)
    
    logger.info("Analyzing classes for stratification...")
    for img_path in tqdm(images, desc="Reading labels"):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        classes = get_image_classes(lbl_path)
        primary_class = classes[0] if classes else -1
        class_to_images[primary_class].append(img_path)
        
    splits = {"train": [], "val": [], "test": []}
    
    np.random.seed(42)
    for cls, imgs in class_to_images.items():
        np.random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits["train"].extend(imgs[:n_train])
        splits["val"].extend(imgs[n_train:n_train + n_val])
        splits["test"].extend(imgs[n_train + n_val:])
        
    logger.info("Copying files to splits...")
    for split_name, imgs in splits.items():
        out_img = output_dir / split_name / "images"
        out_lbl = output_dir / split_name / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(imgs, desc=f"Copying {split_name}"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            shutil.copy2(img_path, out_img / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, out_lbl / lbl_path.name)
                
    logger.info("Split Statistics:")
    for split_name, imgs in splits.items():
        logger.info(f"{split_name.capitalize()}: {len(imgs)} images")

def main() -> None:
    parser = argparse.ArgumentParser(description="Perform stratified 70/15/15 split.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input processed dataset directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/dataset"), help="Output dataset directory")
    args = parser.parse_args()
    
    stratified_split(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
