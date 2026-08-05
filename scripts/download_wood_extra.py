"""
download_wood_extra.py
=======================
Downloads the Surface Crack Detection dataset from Kaggle (~4,000 images)
and converts it into YOLO bounding box format for the wood detector.

Dataset: arunrk7/surface-crack-detection
  - Positive/ (2,227 crack images)
  - Negative/ (2,227 no-crack images — skipped)

Since this is a classification dataset (no bounding boxes), we treat the
ENTIRE image as the defect bounding box for Positive images:
  → class 0 (defect) cx=0.5 cy=0.5 w=1.0 h=1.0

This gives the YOLO model examples that teach:
  "this whole image-type = cracked wood surface"

Usage:
    # Step 1: Install kaggle API (if not already)
    pip install kaggle

    # Step 2: Place kaggle.json in ~/.kaggle/kaggle.json
    # (Download from: https://www.kaggle.com/settings → API → Create Token)

    # Step 3: Run
    python scripts/download_wood_extra.py           # dry run
    python scripts/download_wood_extra.py --apply   # download + convert + merge
"""

import argparse
import logging
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).parent.parent
DOWNLOAD_DIR  = PROJECT_ROOT / "data" / "Wood" / "surface_crack_kaggle"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "processed_wood"

KAGGLE_DATASET = "arunrk7/surface-crack-detection"
RANDOM_SEED    = 42
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15

# YOLO label for a whole-image bounding box
WHOLE_IMAGE_LABEL = "0 0.500000 0.500000 1.000000 1.000000"


def check_kaggle_api() -> bool:
    """Verify kaggle CLI is available and authenticated."""
    try:
        result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            return False
    except FileNotFoundError:
        logger.error("kaggle CLI not found. Run: pip install kaggle")
        return False

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.error(
            f"kaggle.json not found at {kaggle_json}\n"
            "  1. Go to https://www.kaggle.com/settings\n"
            "  2. Click 'API' → 'Create New Token'\n"
            "  3. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json"
        )
        return False

    logger.info("Kaggle API: OK")
    return True


def download_dataset() -> Path | None:
    """Download and extract the Kaggle dataset. Returns extracted directory."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOAD_DIR / "surface-crack-detection.zip"

    if zip_path.exists():
        logger.info(f"Zip already exists at {zip_path} — skipping download")
    else:
        logger.info(f"Downloading {KAGGLE_DATASET}...")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(DOWNLOAD_DIR)],
            capture_output=False
        )
        if result.returncode != 0:
            logger.error("Download failed. Check your Kaggle API credentials.")
            return None

    # Extract
    extract_dir = DOWNLOAD_DIR / "extracted"
    if extract_dir.exists():
        logger.info("Already extracted — skipping")
    else:
        logger.info("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)

    return extract_dir


def find_positive_images(extract_dir: Path) -> list[Path]:
    """Find all crack (Positive) images in the extracted dataset."""
    positive_dir = None

    # Common structures: Positive/, positive/, crack/, Surface Crack Detection/Positive/
    for candidate in extract_dir.rglob("*"):
        if candidate.is_dir() and candidate.name.lower() in {"positive", "crack", "cracked"}:
            positive_dir = candidate
            break

    if positive_dir is None:
        # Fallback: list all subdirs and pick the larger one
        subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 2:
            # Binary dataset: take the one that's not 'negative'
            positive_dir = next(
                (d for d in subdirs if "neg" not in d.name.lower()), subdirs[0]
            )
        else:
            logger.error(f"Could not find Positive folder in {extract_dir}")
            return []

    images = list(positive_dir.glob("*.jpg")) + \
             list(positive_dir.glob("*.jpeg")) + \
             list(positive_dir.glob("*.png"))

    logger.info(f"Found {len(images)} Positive (crack) images in {positive_dir}")
    return images


def merge_into_wood_split(images: list[Path], dry_run: bool = True) -> None:
    """
    Split crack images 70/15/15 and add to processed_wood with whole-image YOLO labels.
    """
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split_name, imgs in splits.items():
        out_img = OUTPUT_DIR / "images" / split_name
        out_lbl = OUTPUT_DIR / "labels" / split_name
        logger.info(f"  {split_name}: +{len(imgs)} images {'(dry run)' if dry_run else ''}")

        if not dry_run:
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

            for img_path in imgs:
                dest_img = out_img / img_path.name
                dest_lbl = out_lbl / (img_path.stem + ".txt")

                counter = 1
                while dest_img.exists():
                    stem = f"{img_path.stem}_crack{counter}"
                    dest_img = out_img / (stem + img_path.suffix)
                    dest_lbl = out_lbl / (stem + ".txt")
                    counter += 1

                shutil.copy2(img_path, dest_img)
                dest_lbl.write_text(WHOLE_IMAGE_LABEL)


def main():
    parser = argparse.ArgumentParser(
        description="Download Surface Crack Detection dataset and merge into wood split"
    )
    parser.add_argument("--apply", action="store_true",
                        help="Actually download and write files (default: dry-run only)")
    args = parser.parse_args()
    dry_run = not args.apply

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN — checks only. Use --apply to download + merge.")
        logger.info("=" * 60)
        logger.info("\nThis script will:")
        logger.info("  1. Download arunrk7/surface-crack-detection from Kaggle (~300 MB)")
        logger.info("  2. Extract Positive/ folder (~2,227 crack images)")
        logger.info("  3. Create whole-image YOLO labels (class 0 = defect)")
        logger.info("  4. Merge into data/processed_wood/ train/val/test splits")
        logger.info("\nExpected result:")
        logger.info("  Wood train: 1,366 → ~2,924 images (+~1,558)")
        logger.info("\nPrerequisites:")
        logger.info("  pip install kaggle")
        logger.info("  Place kaggle.json at ~/.kaggle/kaggle.json")
        check_kaggle_api()
        return

    # APPLY mode
    logger.info("=" * 60)
    logger.info("APPLYING — downloading and merging wood crack data")
    logger.info("=" * 60)

    if not check_kaggle_api():
        logger.error("Fix Kaggle API issues above, then retry.")
        sys.exit(1)

    logger.info("\n[1/3] Downloading dataset...")
    extract_dir = download_dataset()
    if extract_dir is None:
        sys.exit(1)

    logger.info("\n[2/3] Finding crack images...")
    images = find_positive_images(extract_dir)
    if not images:
        sys.exit(1)

    logger.info(f"\n[3/3] Merging {len(images)} images into processed_wood...")
    merge_into_wood_split(images, dry_run=False)

    existing_train = len(list((OUTPUT_DIR / "images" / "train").glob("*"))) \
                     if (OUTPUT_DIR / "images" / "train").exists() else 1366

    logger.info("\n" + "=" * 60)
    logger.info("DONE!")
    logger.info(f"  Total wood training images now: ~{existing_train}")
    logger.info(f"  Retrain with: python train_all.py --stages 4")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
