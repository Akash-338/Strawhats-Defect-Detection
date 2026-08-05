import os
import glob
import shutil
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

CLASS_MAP = {
    1: 'crease',        # 压褶
    2: 'crescent_gap',  # 月牙缺口  
    3: 'water_spot',    # 水斑
    4: 'welding_line',  # 焊接线
    5: 'silk_spot',     # 丝印
    6: 'inclusion',     # 夹杂
    7: 'oil_spot',      # 油污
    8: 'punching',      # 冲孔
    9: 'rolling_pit',   # 轧坑
    10: 'waist_fold'    # 腰折
}

def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def process_aluminum(data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    label_dir = data_dir / 'lable'
    if not label_dir.exists():
        print(f"Error: {label_dir} does not exist.")
        return
        
    images_out = output_dir / 'images'
    labels_out = output_dir / 'labels'
    
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    xml_files = list(label_dir.glob('*.xml'))
    if not xml_files:
        print(f"No XML files found in {label_dir}")
        return
        
    dataset = []
    
    print("Processing annotations...")
    for xml_file in tqdm(xml_files):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            
            # Find the corresponding image
            image_name = xml_file.stem + '.jpg'
            image_path = None
            
            # Match XML filenames to images in folders 1/ through 10/
            for i in range(1, 11):
                potential_img = data_dir / str(i) / image_name
                if potential_img.exists():
                    image_path = potential_img
                    break
                    
            if not image_path:
                # Also try .bmp just in case
                image_name_bmp = xml_file.stem + '.bmp'
                for i in range(1, 11):
                    potential_img = data_dir / str(i) / image_name_bmp
                    if potential_img.exists():
                        image_path = potential_img
                        break
            
            if not image_path:
                continue
                
            yolo_labels = []
            
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text if obj.find('difficult') is not None else '0'
                if int(difficult) == 1:
                    continue
                
                # class_id is the folder number - 1 (0-indexed for YOLO)
                class_id = int(image_path.parent.name) - 1 
                
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_box((w, h), b)
                
                yolo_labels.append(f"{class_id} {' '.join([str(a) for a in bb])}")
                
            if yolo_labels:
                dataset.append((image_path, yolo_labels))
                
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            
    print(f"Found {len(dataset)} annotated images.")
    
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
            dst_lbl = split_lbl_dir / (img_path.stem + '.txt')
            
            shutil.copy(img_path, dst_img)
            with open(dst_lbl, 'w') as f:
                f.write('\n'.join(labels))
                
    print(f"Done. Summary:")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Aluminum VOC to YOLO format')
    parser.add_argument('--data_dir', type=str, default=r'c:\Users\mmddf\Desktop\RVCE\data\Aluminum', help='Original data directory')
    parser.add_argument('--output_dir', type=str, default=r'c:\Users\mmddf\Desktop\RVCE\data\processed_aluminum', help='Output directory')
    args = parser.parse_args()
    
    process_aluminum(args.data_dir, args.output_dir)
