import os
import shutil
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def process_wood(data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    defect_dir = data_dir / 'Defect'
    clean_dir = data_dir / 'No_Defect'
    
    if not defect_dir.exists():
        print(f"Error: {defect_dir} does not exist.")
        return
        
    images_out = output_dir / 'images'
    labels_out = output_dir / 'labels'
    
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    defect_images = list(defect_dir.glob('*.*'))
    defect_images = [p for p in defect_images if p.suffix.lower() in ['.jpg', '.png', '.bmp', '.jpeg']]
    
    print(f"Found {len(defect_images)} defect images.")
    
    dataset = []
    print("Generating labels...")
    for img_path in tqdm(defect_images):
        try:
            # OpenCV contour detection approach
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            h, w = img.shape
            
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            yolo_labels = []
            if contours:
                # Find largest contour as the defect
                c = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(c)
                
                # Convert to YOLO
                cx = (x + bw/2) / w
                cy = (y + bh/2) / h
                nw = bw / w
                nh = bh / h
                yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            else:
                # Fallback to whole image
                yolo_labels.append("0 0.5 0.5 1.0 1.0")
                
            dataset.append((img_path, yolo_labels))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    clean_images = list(clean_dir.glob('*.*'))
    clean_images = [p for p in clean_images if p.suffix.lower() in ['.jpg', '.png', '.bmp', '.jpeg']]
    print(f"Found {len(clean_images)} clean images.")
    
    for img_path in clean_images:
        dataset.append((img_path, [])) # Empty labels for negative samples
        
    if len(dataset) == 0:
        print("No valid data found to split.")
        return
        
    train_val, test = train_test_split(dataset, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.15/0.85, random_state=42)
    
    splits = {'train': train, 'val': val, 'test': test}
    
    print("Copying files and saving labels...")
    for split_name, split_data in splits.items():
        split_img_dir = images_out / split_name
        split_lbl_dir = labels_out / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path, labels in tqdm(split_data, desc=split_name):
            dst_img = split_img_dir / img_path.name
            shutil.copy(img_path, dst_img)
            
            if labels:
                dst_lbl = split_lbl_dir / (img_path.stem + '.txt')
                with open(dst_lbl, 'w') as f:
                    f.write('\n'.join(labels))
                
    print(f"Done. Summary:")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Wood data to YOLO format')
    parser.add_argument('--data_dir', type=str, default=r'c:\Users\mmddf\Desktop\RVCE\data\Wood\dataset_wood', help='Original data directory')
    parser.add_argument('--output_dir', type=str, default=r'c:\Users\mmddf\Desktop\RVCE\data\processed_wood', help='Output directory')
    args = parser.parse_args()
    
    process_wood(args.data_dir, args.output_dir)
