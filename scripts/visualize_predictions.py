"""
visualize_predictions.py
Draws bounding boxes and class labels on images from YOLO-format predictions.
"""

import argparse
import logging
from pathlib import Path

import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASSES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled_in_scale",
    5: "scratches"
}

COLORS = {
    0: (0, 0, 255),    # Red
    1: (0, 255, 0),    # Green
    2: (255, 0, 0),    # Blue
    3: (0, 255, 255),  # Yellow
    4: (255, 0, 255),  # Magenta
    5: (255, 255, 0)   # Cyan
}

def draw_bboxes(img_path: Path, label_path: Path, output_path: Path) -> None:
    """Draw YOLO format bboxes on an image."""
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning(f"Could not read image {img_path}")
        return
        
    h, w = img.shape[:2]
    
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    conf = float(parts[5]) if len(parts) == 6 else None
                    
                    x = int((cx - bw / 2) * w)
                    y = int((cy - bh / 2) * h)
                    box_w = int(bw * w)
                    box_h = int(bh * h)
                    
                    color = COLORS.get(class_id, (255, 255, 255))
                    label_name = CLASSES.get(class_id, f"cls_{class_id}")
                    if conf is not None:
                        label_name += f" {conf:.2f}"
                        
                    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), color, 2)
                    cv2.putText(img, label_name, (x, max(10, y - 5)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
    cv2.imwrite(str(output_path), img)

def visualize_batch(images_dir: Path, labels_dir: Path, output_dir: Path) -> None:
    """Batch visualize predictions/labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images = list(images_dir.glob("*.jpg"))
    
    logger.info(f"Visualizing {len(images)} images...")
    for img_path in tqdm(images, desc="Visualizing"):
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        out_path = output_dir / img_path.name
        draw_bboxes(img_path, lbl_path, out_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLO format predictions/labels.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing images")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Directory containing YOLO txt files")
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations"), help="Output directory")
    args = parser.parse_args()
    
    visualize_batch(args.images_dir, args.labels_dir, args.output_dir)

if __name__ == "__main__":
    main()
