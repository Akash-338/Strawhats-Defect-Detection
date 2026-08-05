import argparse
import logging
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_dataset(output_dir: Path) -> None:
    """Download NEU-DET dataset using Kaggle API."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "kaustubhbhole/neu-surface-defect-database"
    zip_path = output_dir / "neu-surface-defect-database.zip"
    
    if not zip_path.exists():
        logger.info(f"Downloading {dataset_name} from Kaggle...")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(output_dir)],
                check=True
            )
        except FileNotFoundError:
            logger.error("Kaggle CLI not found. Please 'pip install kaggle' and set up ~/.kaggle/kaggle.json")
            return
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download dataset: {e}")
            return
            
    logger.info("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc="Extracting"):
            zip_ref.extract(member, output_dir)
            
    validate_dataset(output_dir)

def validate_dataset(data_dir: Path) -> None:
    """Validate that we have 1800 images and 1800 XML annotations."""
    images_dir = data_dir / "NEU-DET" / "IMAGES"
    annotations_dir = data_dir / "NEU-DET" / "ANNOTATIONS"
    
    if not images_dir.exists() or not annotations_dir.exists():
        logger.warning(f"Expected directories not found in {data_dir}. Check extraction.")
        return
        
    num_images = len(list(images_dir.glob("*.jpg")))
    num_xml = len(list(annotations_dir.glob("*.xml")))
    
    logger.info(f"Found {num_images} images and {num_xml} XML annotations.")
    if num_images == 1800 and num_xml == 1800:
        logger.info("Validation successful: 1800 images and 1800 XMLs present.")
    else:
        logger.warning("Validation failed: Expected exactly 1800 images and 1800 XMLs.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate NEU-DET dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/NEU-DET"), help="Directory to save the dataset")
    args = parser.parse_args()
    
    download_dataset(args.output_dir)

if __name__ == "__main__":
    main()
