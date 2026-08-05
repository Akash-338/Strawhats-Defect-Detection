"""
Download and convert GC10-DET steel surface defect dataset to YOLO format.
GC10-DET: 3,570 images, 10 defect classes, lab-controlled (morphology-friendly).

Usage:
    python scripts/download_gc10det.py
"""

import os
import sys
import json
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

# ── Class mapping ─────────────────────────────────────────────────────────────
# GC10-DET 10 classes (start from ID 6 to not clash with NEU-DET 0-5)
GC10_CLASSES = {
    'punching':      6,
    'weld_line':     7,
    'crescent_gap':  8,
    'water_spot':    9,
    'oil_spot':      10,
    'silk_spot':     11,
    'inclusion':     1,   # Same as NEU-DET inclusion → reuse ID 1
    'rolled_pit':    12,
    'crease':        13,
    'waist_folding': 14,
}

# Also handle variant spellings found in the dataset
# INCLUDING Chinese pinyin names used in the Kaggle version!
GC10_ALIASES = {
    # English variants
    'weldline':     'weld_line',
    'crescentgap':  'crescent_gap',
    'waterspot':    'water_spot',
    'oilspot':      'oil_spot',
    'silkspot':     'silk_spot',
    'rolledpit':    'rolled_pit',
    'waistfolding': 'waist_folding',
    # Chinese pinyin numbered names (Kaggle version)
    '1_chongkong':  'punching',      # 冲孔 = punching holes
    '2_hanfeng':    'weld_line',     # 焊缝 = weld seam
    '3_yueyawan':   'crescent_gap',  # 月牙弯 = crescent gap
    '4_shuiban':    'water_spot',    # 水斑 = water spot
    '5_youban':     'oil_spot',      # 油斑 = oil spot
    '6_siban':      'silk_spot',     # 丝斑 = silk spot
    '7_yiwu':       'inclusion',     # 异物 = foreign object/inclusion
    '8_yahen':      'rolled_pit',    # 压痕 = press mark
    '9_zhehen':     'crease',        # 褶痕 = crease/fold mark
    '10_yaozhe':    'waist_folding', # 腰折 = waist folding
    '10_yaozhed':   'waist_folding', # variant spelling
}

ALL_CLASSES = [
    'crazing',       # 0  - NEU-DET
    'inclusion',     # 1  - NEU-DET + GC10
    'patches',       # 2  - NEU-DET
    'pitted_surface',# 3  - NEU-DET
    'rolled_in_scale',# 4 - NEU-DET
    'scratches',     # 5  - NEU-DET
    'punching',      # 6  - GC10
    'weld_line',     # 7  - GC10
    'crescent_gap',  # 8  - GC10
    'water_spot',    # 9  - GC10
    'oil_spot',      # 10 - GC10
    'silk_spot',     # 11 - GC10
    'rolled_pit',    # 12 - GC10
    'crease',        # 13 - GC10
    'waist_folding', # 14 - GC10
]


def download_gc10det(output_dir: Path):
    """Download GC10-DET from Kaggle."""
    raw_dir = output_dir / 'gc10det'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded/extracted
    existing_xmls = list(raw_dir.rglob('*.xml'))
    if len(existing_xmls) > 0:
        print(f"✅ Found {len(existing_xmls)} existing XML annotations in {raw_dir}")
        return raw_dir
        
    print("📦 Downloading GC10-DET from Kaggle...")
    print("   Dataset: alex000kim/gc10det")
    
    ret = os.system(f'python -m kaggle datasets download -d alex000kim/gc10det --path "{raw_dir}" --unzip')
    if ret != 0:
        ret = os.system(f'kaggle datasets download -d alex000kim/gc10det --path "{raw_dir}" --unzip')
        
    if ret != 0 and len(list(raw_dir.rglob('*.xml'))) == 0:
        print("\n❌ Kaggle download failed. Try manually:")
        print("   1. Go to https://www.kaggle.com/datasets/alex000kim/gc10det")
        print("   2. Download and extract to:", raw_dir)
        sys.exit(1)
    
    print(f"✅ Downloaded to {raw_dir}")
    return raw_dir


def find_annotation_files(raw_dir: Path):
    """Find all XML annotation files in the dataset."""
    xml_files = list(raw_dir.rglob('*.xml'))
    print(f"   Found {len(xml_files)} XML annotation files")
    return xml_files


def parse_voc_xml(xml_path: Path):
    """Parse a VOC-format XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Find corresponding image
    filename = root.find('filename').text
    
    annotations = []
    for obj in root.findall('object'):
        name = obj.find('name').text.lower().strip()
        
        # Normalize class name
        name = GC10_ALIASES.get(name, name)
        
        if name not in GC10_CLASSES:
            # Try partial match
            matched = next((k for k in GC10_CLASSES if k in name or name in k), None)
            if matched:
                name = matched
            else:
                print(f"   ⚠️ Unknown class '{name}' in {xml_path.name}, skipping")
                continue
        
        class_id = GC10_CLASSES[name]
        
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # Convert to YOLO format (normalized center x, y, w, h)
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        
        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w = max(0.001, min(1.0, w))
        h = max(0.001, min(1.0, h))
        
        annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    
    return filename, annotations


def convert_gc10det(raw_dir: Path, out_dir: Path):
    """Convert GC10-DET VOC XML annotations to YOLO format."""
    print("\n🔄 Converting GC10-DET to YOLO format...")
    
    images_out = out_dir / 'images'
    labels_out = out_dir / 'labels'
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    xml_files = find_annotation_files(raw_dir)
    
    converted = 0
    skipped = 0
    
    for xml_path in tqdm(xml_files, desc="Converting"):
        try:
            filename, annotations = parse_voc_xml(xml_path)
            
            if not annotations:
                skipped += 1
                continue
            
            # Find image file
            img_candidates = list(raw_dir.rglob(filename))
            if not img_candidates:
                # Try without extension
                stem = Path(filename).stem
                img_candidates = list(raw_dir.rglob(f"{stem}.*"))
                img_candidates = [f for f in img_candidates if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
            
            if not img_candidates:
                skipped += 1
                continue
            
            img_path = img_candidates[0]
            
            # Copy image
            dest_img = images_out / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # Write label
            label_name = img_path.stem + '.txt'
            dest_label = labels_out / label_name
            with open(dest_label, 'w') as f:
                f.write('\n'.join(annotations))
            
            converted += 1
            
        except Exception as e:
            skipped += 1
            continue
    
    print(f"✅ Converted: {converted} images | Skipped: {skipped}")
    return converted


def save_class_map(out_dir: Path):
    """Save the unified class mapping JSON."""
    class_map = {str(v): k for k, v in {**{'crazing': 0, 'inclusion': 1, 'patches': 2,
                                             'pitted_surface': 3, 'rolled_in_scale': 4,
                                             'scratches': 5}, **GC10_CLASSES}.items()}
    
    with open(out_dir / 'gc10det_classes.json', 'w') as f:
        json.dump({'classes': ALL_CLASSES, 'num_classes': len(ALL_CLASSES)}, f, indent=2)
    
    print(f"\n📋 Unified class map ({len(ALL_CLASSES)} classes):")
    for i, cls in enumerate(ALL_CLASSES):
        source = "NEU-DET" if i <= 5 else "GC10-DET"
        print(f"   {i:2d}: {cls:20s} ({source})")


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / 'data' / 'raw'
    gc10_out = project_root / 'data' / 'gc10det_converted'
    
    print("=" * 60)
    print("  GC10-DET Download & Conversion Script")
    print("  Steel surface defect dataset — 3,570 images, 10 classes")
    print("=" * 60)
    
    # Step 1: Download
    raw_gc10 = download_gc10det(raw_dir)
    
    # Step 2: Convert
    converted = convert_gc10det(raw_gc10, gc10_out)
    
    # Step 3: Save class map
    save_class_map(gc10_out)
    
    print(f"\n✅ Done! GC10-DET converted to YOLO format:")
    print(f"   Images: {gc10_out / 'images'}")
    print(f"   Labels: {gc10_out / 'labels'}")
    print(f"\n📌 Next step: Run merge_steel_datasets.py to combine with NEU-DET")


if __name__ == '__main__':
    main()
