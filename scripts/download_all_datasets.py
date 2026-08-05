import os
import argparse
import subprocess
import zipfile
from pathlib import Path
from typing import List, Dict

# MANUAL DOWNLOAD INSTRUCTIONS
# If the Kaggle API is not set up, you can manually download the datasets from the following URLs:
# 
# Steel:
# - NEU-DET: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
# - Severstal: https://www.kaggle.com/competitions/severstal-steel-defect-detection
#
# Aluminum:
# - Aluminum Profile Defects: https://www.kaggle.com/datasets/kaixu7/defects-in-aluminium-profiles
# - GC10-DET: https://www.kaggle.com/datasets/alex000kim/gc10det
#
# Wood:
# - Wood Surface Defects: https://www.kaggle.com/datasets/rishikeshkonapure/large-scale-image-dataset-of-wood-surface-defects
#
# Extract the contents into data/steel, data/aluminum, and data/wood respectively.

DATASETS: Dict[str, List[Dict[str, str]]] = {
    'steel': [
        {'type': 'dataset', 'id': 'kaustubhdikshit/neu-surface-defect-database'},
        {'type': 'competition', 'id': 'severstal-steel-defect-detection'}
    ],
    'aluminum': [
        {'type': 'dataset', 'id': 'kaixu7/defects-in-aluminium-profiles'},
        {'type': 'dataset', 'id': 'alex000kim/gc10det'}
    ],
    'wood': [
        {'type': 'dataset', 'id': 'rishikeshkonapure/large-scale-image-dataset-of-wood-surface-defects'}
    ]
}

def ensure_kaggle_installed():
    """Ensures kaggle module is installed."""
    try:
        import kaggle
    except ImportError:
        print("Kaggle module not found. Installing via pip...")
        subprocess.check_call([os.sys.executable, "-m", "pip", "install", "kaggle"])
        print("Kaggle installed successfully.")

def download_and_extract(item: Dict[str, str], output_dir: Path):
    """Downloads a Kaggle dataset or competition and extracts it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    item_id = item['id']
    item_type = item['type']
    
    slug_name = item_id.split('/')[-1]
    zip_path = output_dir / f"{slug_name}.zip"
    
    print(f"Downloading {item_type} '{item_id}' to {output_dir}...")
    try:
        if item_type == 'dataset':
            subprocess.check_call(['kaggle', 'datasets', 'download', '-d', item_id, '-p', str(output_dir)])
        elif item_type == 'competition':
            subprocess.check_call(['kaggle', 'competitions', 'download', '-c', item_id, '-p', str(output_dir)])
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {item_id}: {e}")
        print("Please ensure your Kaggle API credentials are set up correctly (~/.kaggle/kaggle.json).")
        return

    if zip_path.exists():
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        zip_path.unlink()
        
        # Summary
        img_count = len(list(output_dir.rglob('*.jpg'))) + len(list(output_dir.rglob('*.png')))
        xml_count = len(list(output_dir.rglob('*.xml')))
        txt_count = len(list(output_dir.rglob('*.txt')))
        print(f"Extraction complete for {item_id}.")
        print(f"Summary: {img_count} images found, {xml_count} XML annotations, {txt_count} TXT annotations.\n")
    else:
        print(f"Warning: Expected zip file {zip_path} not found after download.")

def main():
    parser = argparse.ArgumentParser(description="Download material defect datasets from Kaggle.")
    parser.add_argument('--materials', type=str, default='all', 
                        help="Comma-separated list of materials (steel,aluminum,wood) or 'all'.")
    parser.add_argument('--data-dir', type=str, default='data', help="Base output directory for datasets.")
    
    args = parser.parse_args()
    
    ensure_kaggle_installed()
    
    if args.materials.lower() == 'all':
        materials = list(DATASETS.keys())
    else:
        materials = [m.strip().lower() for m in args.materials.split(',')]
        
    base_dir = Path(args.data_dir)
    
    for mat in materials:
        if mat not in DATASETS:
            print(f"Warning: Material '{mat}' not recognized. Skipping.")
            continue
        
        mat_dir = base_dir / mat
        print(f"--- Processing material: {mat.upper()} ---")
        for item in DATASETS[mat]:
            download_and_extract(item, mat_dir)

if __name__ == '__main__':
    main()
