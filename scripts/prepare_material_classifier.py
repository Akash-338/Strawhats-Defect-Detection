import os
import shutil
import random
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def prepare_classifier(neu_dir, al_dir, wood_dir, output_dir):
    neu_dir = Path(neu_dir)
    al_dir = Path(al_dir)
    wood_dir = Path(wood_dir)
    output_dir = Path(output_dir)
    
    if not (neu_dir.exists() and al_dir.exists() and wood_dir.exists()):
        print("Error: One or more source directories do not exist.")
        return
        
    dataset = []
    
    # 1. Steel clean samples (random 1,000 from NEU-DET)
    neu_images = [p for p in neu_dir.rglob('*') if p.suffix.lower() in ['.jpg', '.png', '.bmp', '.jpeg']]
    if len(neu_images) > 1000:
        neu_images = random.sample(neu_images, 1000)
    for img in neu_images:
        dataset.append((img, 'steel'))
        
    # 2. Aluminum clean samples (random 1,000)
    al_images = [p for p in al_dir.rglob('*') if p.suffix.lower() in ['.jpg', '.png', '.bmp', '.jpeg']]
    if len(al_images) > 1000:
        al_images = random.sample(al_images, 1000)
    for img in al_images:
        dataset.append((img, 'aluminum'))
        
    # 3. Wood clean samples (random 1,000)
    wood_images = [p for p in wood_dir.rglob('*') if p.suffix.lower() in ['.jpg', '.png', '.bmp', '.jpeg']]
    if len(wood_images) > 1000:
        wood_images = random.sample(wood_images, 1000)
    for img in wood_images:
        dataset.append((img, 'wood'))
        
    print(f"Total images collected: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No valid data found.")
        return
        
    labels = [x[1] for x in dataset]
    train_val, test, _, _ = train_test_split(dataset, labels, test_size=0.15, random_state=42, stratify=labels)
    train_val_labels = [x[1] for x in train_val]
    train, val, _, _ = train_test_split(train_val, train_val_labels, test_size=0.15/0.85, random_state=42, stratify=train_val_labels)
    
    splits = {'train': train, 'val': val, 'test': test}
    
    print("Resizing and saving images...")
    for split_name, split_data in splits.items():
        for cls in ['steel', 'aluminum', 'wood']:
            (output_dir / split_name / cls).mkdir(parents=True, exist_ok=True)
            
        for img_path, cls in tqdm(split_data, desc=split_name):
            try:
                # np.fromfile handles non-ASCII/Unicode paths on Windows
                img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_resized = cv2.resize(img, (224, 224))
                dst_path = output_dir / split_name / cls / img_path.name
                
                dst_path = dst_path.with_suffix('.jpg')
                cv2.imwrite(str(dst_path), img_resized)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    print(f"Done. Summary:")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")

if __name__ == '__main__':
    # Auto-detect project root from script location
    _project_root = str(Path(__file__).resolve().parent.parent)
    
    wood_path = os.path.join(_project_root, 'data', 'processed', 'wood_10class')
    if not os.path.exists(wood_path):
        wood_path = os.path.join(_project_root, 'data', 'Wood')
        
    parser = argparse.ArgumentParser(description='Prepare material classifier dataset')
    parser.add_argument('--neu_dir', type=str, default=os.path.join(_project_root, 'data', 'NEU-DET'), help='NEU-DET data directory (steel images)')
    parser.add_argument('--al_dir', type=str, default=os.path.join(_project_root, 'data', 'Aluminum'), help='Aluminum clean samples directory')
    parser.add_argument('--wood_dir', type=str, default=wood_path, help='Wood clean samples directory')
    parser.add_argument('--output_dir', type=str, default=os.path.join(_project_root, 'data', 'material_classifier'), help='Output directory')
    args = parser.parse_args()
    
    prepare_classifier(args.neu_dir, args.al_dir, args.wood_dir, args.output_dir)
