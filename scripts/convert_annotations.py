"""
convert_annotations.py
Parses VOC XML annotations from NEU-DET, converts to YOLO format,
and copies images and labels to a processed directory.
"""

import argparse
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASS_MAPPING = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "rolled_in_scale": 4,
    "scratches": 5
}

def convert_voc_to_yolo(xml_path: Path, output_label_path: Path) -> bool:
    """Parse VOC XML and write YOLO format txt file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find("size")
        if size is None:
            return False
            
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        
        with open(output_label_path, "w") as out_file:
            for obj in root.findall("object"):
                name = obj.find("name").text.lower()
                if name not in CLASS_MAPPING:
                    logger.warning(f"Unknown class {name} in {xml_path}")
                    continue
                    
                class_id = CLASS_MAPPING[name]
                bndbox = obj.find("bndbox")
                
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                
                cx = (xmin + xmax) / 2.0 / w
                cy = (ymin + ymax) / 2.0 / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))
                
                out_file.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        return True
    except Exception as e:
        logger.error(f"Error parsing {xml_path}: {e}")
        return False

def find_image(images_dir: Path, stem: str) -> Path | None:
    """Find image file across class subfolders and extensions."""
    for ext in ['.jpg', '.bmp', '.png', '.jpeg']:
        # Direct path
        direct = images_dir / f"{stem}{ext}"
        if direct.exists():
            return direct
        # Search in class subfolders
        for subfolder in images_dir.iterdir():
            if subfolder.is_dir():
                candidate = subfolder / f"{stem}{ext}"
                if candidate.exists():
                    return candidate
    return None

def process_annotations(data_dir: Path, output_dir: Path) -> None:
    """Convert annotations and copy images.
    
    Handles the actual NEU-DET Kaggle structure:
      data/NEU-DET/NEU-DET/{train,validation}/annotations/*.xml
      data/NEU-DET/NEU-DET/{train,validation}/images/{classname}/*.bmp
    """
    # Try nested structure first (Kaggle download)
    nested = data_dir / "NEU-DET"
    if nested.exists():
        data_dir = nested
    
    out_img_dir = output_dir / "images"
    out_lbl_dir = output_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    for split in ["train", "validation"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            # Fallback: try IMAGES/ANNOTATIONS flat structure
            images_dir = data_dir / "IMAGES"
            annotations_dir = data_dir / "ANNOTATIONS"
            if images_dir.exists() and annotations_dir.exists():
                xml_files = list(annotations_dir.glob("*.xml"))
                logger.info(f"Found {len(xml_files)} XML files (flat structure).")
                for xml_path in tqdm(xml_files, desc="Converting"):
                    img = find_image(images_dir, xml_path.stem)
                    if img is None:
                        continue
                    out_lbl = out_lbl_dir / f"{xml_path.stem}.txt"
                    if convert_voc_to_yolo(xml_path, out_lbl):
                        shutil.copy2(img, out_img_dir / f"{xml_path.stem}{img.suffix}")
                        success_count += 1
                break
            else:
                logger.warning(f"Split '{split}' not found, skipping.")
                continue
        
        annotations_dir = split_dir / "annotations"
        images_dir = split_dir / "images"
        
        if not annotations_dir.exists():
            logger.warning(f"No annotations dir in {split_dir}")
            continue
        
        xml_files = list(annotations_dir.glob("*.xml"))
        logger.info(f"[{split}] Found {len(xml_files)} XML files. Processing...")
        
        for xml_path in tqdm(xml_files, desc=f"Converting {split}"):
            img = find_image(images_dir, xml_path.stem)
            if img is None:
                logger.warning(f"Image not found for {xml_path.name}")
                continue
            
            out_lbl = out_lbl_dir / f"{xml_path.stem}.txt"
            if convert_voc_to_yolo(xml_path, out_lbl):
                shutil.copy2(img, out_img_dir / f"{xml_path.stem}{img.suffix}")
                success_count += 1
    
    logger.info(f"Successfully processed {success_count} files.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VOC XML to YOLO format.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/NEU-DET"), help="NEU-DET dataset directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/neu-det"), help="Output directory")
    args = parser.parse_args()
    
    process_annotations(args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()
