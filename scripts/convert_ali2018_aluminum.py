"""
convert_ali2018_aluminum.py  (v2 — fixed)
==========================================
The ali2018 dataset uses Chinese filenames and is organized as
folder-per-class (no separate XML annotations). Each subfolder name
IS the class label.

Strategy: whole-image bounding box (0.5 0.5 1.0 1.0) per image,
class mapped from folder name → nearest GC10-DET class index.

ali2018 folder name                 → GC10-DET class id  (name)
----------------------------          -----------------   ------
Be injured by a collision           → 0                  crease
Coating cracking                    → 1                  crescent_gap
Convex powder                       → 6                  oil_spot
Dirty spot                          → 6                  oil_spot
Drain bottom                        → 3                  welding_line
Orange peel                         → 2                  water_spot
The transverse strip is dented      → 0                  crease
non-conducting                      → 4                  silk_spot
pitting                             → 8                  rolling_pit
scuffing                            → 9                  waist_fold
Clean sample                        → SKIP (no defect)

Usage:
    python scripts/convert_ali2018_aluminum.py          # dry run
    python scripts/convert_ali2018_aluminum.py --apply  # write files
"""

import argparse
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

ALI2018_DIR = PROJECT_ROOT / "data" / "Aluminum" / "ali2018"
OUTPUT_DIR  = PROJECT_ROOT / "data" / "processed_aluminum"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

# Whole-image bounding box template (class_id cx cy w h)
WHOLE_IMG_BOX = "0.500000 0.500000 1.000000 1.000000"

# Folder name → GC10-DET class index mapping (case-insensitive)
FOLDER_CLASS_MAP = {
    "be injured by a collision":       0,  # crease
    "coating cracking":                1,  # crescent_gap
    "convex powder":                   6,  # oil_spot
    "dirty spot":                      6,  # oil_spot
    "drain bottom":                    3,  # welding_line
    "orange peel":                     2,  # water_spot
    "the transverse strip is dented":  0,  # crease
    "non-conducting":                  4,  # silk_spot
    "pitting":                         8,  # rolling_pit
    "scuffing":                        9,  # waist_fold
    # "clean sample" → omitted = skip
}

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_ali2018_pairs() -> list[tuple[Path, str]]:
    """
    Walk each ali2018 class subfolder, collect (image_path, yolo_label_line).
    Returns list of (img_path, label_line) for all valid defect images.
    """
    results = []
    skipped_clean = 0
    skipped_unknown = 0

    for subfolder in sorted(ALI2018_DIR.iterdir()):
        if not subfolder.is_dir():
            continue

        folder_name_lower = subfolder.name.lower().strip()

        # Skip clean/no-defect class
        if "clean" in folder_name_lower:
            imgs = [f for f in subfolder.iterdir() if f.suffix.lower() in IMG_EXTENSIONS]
            skipped_clean += len(imgs)
            logger.info(f"  SKIP '{subfolder.name}': {len(imgs)} clean images")
            continue

        class_id = FOLDER_CLASS_MAP.get(folder_name_lower)
        if class_id is None:
            imgs = [f for f in subfolder.iterdir() if f.suffix.lower() in IMG_EXTENSIONS]
            skipped_unknown += len(imgs)
            logger.warning(f"  UNKNOWN class '{subfolder.name}': {len(imgs)} images skipped")
            continue

        imgs = [f for f in subfolder.iterdir() if f.suffix.lower() in IMG_EXTENSIONS]
        label_line = f"{class_id} {WHOLE_IMG_BOX}"
        for img in imgs:
            results.append((img, label_line))

        logger.info(f"  '{subfolder.name}' → class {class_id}: {len(imgs)} images")

    logger.info(f"\n  Valid defect images:  {len(results)}")
    logger.info(f"  Skipped (clean):      {skipped_clean}")
    logger.info(f"  Skipped (unknown):    {skipped_unknown}")
    return results


def append_to_splits(pairs: list[tuple[Path, str]], dry_run: bool = True) -> None:
    """Stratified-split and append to existing processed_aluminum splits."""
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }

    for split_name, split_pairs in splits.items():
        out_img = OUTPUT_DIR / "images" / split_name
        out_lbl = OUTPUT_DIR / "labels" / split_name
        logger.info(f"  {split_name}: +{len(split_pairs)} images {'(dry run)' if dry_run else ''}")

        if not dry_run:
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

            for img_path, label_line in split_pairs:
                # Use bytes-based copy to handle Chinese/Unicode filenames
                dest_stem = f"ali_{abs(hash(str(img_path))) % 10_000_000:07d}"
                dest_img  = out_img / (dest_stem + img_path.suffix.lower())
                dest_lbl  = out_lbl / (dest_stem + ".txt")

                # Collision guard (extremely unlikely with hash)
                counter = 1
                while dest_img.exists():
                    dest_img = out_img / (dest_stem + f"_{counter}" + img_path.suffix.lower())
                    dest_lbl = out_lbl / (dest_stem + f"_{counter}.txt")
                    counter += 1

                # shutil.copy2 handles Unicode paths fine on Python 3.8+
                shutil.copy2(img_path, dest_img)
                dest_lbl.write_text(label_line, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ali2018 folder-per-class aluminum dataset → YOLO whole-image labels"
    )
    parser.add_argument("--apply", action="store_true",
                        help="Actually write files (default: dry run)")
    args = parser.parse_args()
    dry_run = not args.apply

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN — no files written. Use --apply to execute.")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("APPLYING — writing to data/processed_aluminum/")
        logger.info("=" * 60)

    logger.info("\n[1/2] Collecting ali2018 images by class folder...")
    pairs = collect_ali2018_pairs()

    logger.info(f"\n[2/2] Merging {len(pairs)} images into processed_aluminum splits...")
    append_to_splits(pairs, dry_run=dry_run)

    n_train_new = int(len(pairs) * TRAIN_RATIO)
    existing_train = 1604  # original GC10-DET count

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  ali2018 valid pairs:        {len(pairs)}")
    logger.info(f"  New train images:           ~{n_train_new}")
    logger.info(f"  Existing train images:      ~{existing_train}")
    logger.info(f"  Total after merge:          ~{existing_train + n_train_new}")
    if dry_run:
        logger.info("\n  Run with --apply to write the files!")
    else:
        logger.info(f"\n  Done! Retrain with: python train_all.py --stages 3")


if __name__ == "__main__":
    main()
