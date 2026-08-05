"""
fix_material_classifier_data.py
================================
Populates (or refreshes) the material_classifier dataset for all 3 classes
by sampling from the full expanded processed datasets.

Samples from:
  data/processed/steel/train/images/    → material_classifier/*/steel/
  data/processed_aluminum/images/train/ → material_classifier/*/aluminum/
  data/processed_wood/images/train/     → material_classifier/*/wood/

Old class folders are cleared before resampling so you always get fresh,
diverse samples from the full expanded pool.

Usage:
    python scripts/fix_material_classifier_data.py          # dry run
    python scripts/fix_material_classifier_data.py --apply  # write files
    python scripts/fix_material_classifier_data.py --apply --train-count 500 --val-count 100
"""

import argparse
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

SOURCES = {
    "steel":    PROJECT_ROOT / "data" / "processed" / "steel" / "train" / "images",
    "aluminum": PROJECT_ROOT / "data" / "processed_aluminum" / "images" / "train",
    "wood":     PROJECT_ROOT / "data" / "processed_wood" / "images" / "train",
}

CLASSIFIER_DIR = PROJECT_ROOT / "data" / "material_classifier"

# Default: 300 train / 75 val / 75 test per class = 450 total
# (wood pool is now 15,366 so no reason to keep only 210)
TRAIN_COUNT = 300
VAL_COUNT   = 75
TEST_COUNT  = 75
TOTAL       = TRAIN_COUNT + VAL_COUNT + TEST_COUNT

RANDOM_SEED = 42


def sample_and_copy(material: str, source_dir: Path, dry_run: bool) -> bool:
    """Sample TOTAL images from source_dir and distribute into train/val/test."""
    images = list(source_dir.glob("*.jpg")) + \
             list(source_dir.glob("*.jpeg")) + \
             list(source_dir.glob("*.png")) + \
             list(source_dir.glob("*.bmp"))

    if not images:
        logger.error(f"No images found in {source_dir}")
        return False

    logger.info(f"  {material}: {len(images)} available → sampling {TOTAL}")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(images)

    # If fewer than TOTAL images, take all and warn
    selected = images[:TOTAL]
    if len(selected) < TOTAL:
        logger.warning(f"  Only {len(selected)} images available for {material} (wanted {TOTAL})")

    splits = {
        "train": selected[:TRAIN_COUNT],
        "val":   selected[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT],
        "test":  selected[TRAIN_COUNT + VAL_COUNT:],
    }

    for split_name, imgs in splits.items():
        dest_dir = CLASSIFIER_DIR / split_name / material

        if not dry_run:
            # Clear old samples first — ensures fresh diverse samples from expanded pool
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"    {split_name}/{material}: {len(imgs)} images {'(dry run)' if dry_run else '(writing...)'}")

        if not dry_run:
            for img_path in imgs:
                shutil.copy2(img_path, dest_dir / img_path.name)

    return True


def verify_current_state():
    """Show current image counts per class per split."""
    logger.info("\nCurrent material_classifier state:")
    for split in ["train", "val", "test"]:
        for cls in ["steel", "aluminum", "wood"]:
            d = CLASSIFIER_DIR / split / cls
            count = len(list(d.glob("*"))) if d.exists() else 0
            status = "✅" if count > 0 else "❌ EMPTY"
            logger.info(f"  {split}/{cls}: {count} images {status}")


def main():
    global TRAIN_COUNT, VAL_COUNT, TEST_COUNT, TOTAL

    parser = argparse.ArgumentParser(
        description="Populate/refresh material_classifier dataset for all 3 classes"
    )
    parser.add_argument("--apply", action="store_true",
                        help="Actually copy files (default: dry run)")
    parser.add_argument("--train-count", type=int, default=TRAIN_COUNT,
                        help=f"Images per class for train (default: {TRAIN_COUNT})")
    parser.add_argument("--val-count", type=int, default=VAL_COUNT,
                        help=f"Images per class for val (default: {VAL_COUNT})")
    args = parser.parse_args()

    # ── Apply CLI overrides to module-level globals used by sample_and_copy ──
    TRAIN_COUNT = args.train_count
    VAL_COUNT   = args.val_count
    TEST_COUNT  = args.val_count   # keep test == val for symmetry
    TOTAL       = TRAIN_COUNT + VAL_COUNT + TEST_COUNT

    dry_run = not args.apply

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN — no files written. Use --apply to fix.")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info(f"FIXING material_classifier — {TRAIN_COUNT} train / {VAL_COUNT} val / {TEST_COUNT} test per class")
        logger.info("=" * 60)

    verify_current_state()

    logger.info("\n[Fixing] Sampling all 3 material classes...")
    all_ok = True
    for material, source_dir in SOURCES.items():
        ok = sample_and_copy(material, source_dir, dry_run)
        if not ok:
            all_ok = False

    if not dry_run:
        logger.info("\n[Verifying] After fix:")
        verify_current_state()

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for mat in SOURCES:
        logger.info(f"  {mat.capitalize():10s}: {TRAIN_COUNT} train / {VAL_COUNT} val / {TEST_COUNT} test")
    logger.info(f"  Total per class:   {TOTAL}")
    logger.info(f"  Grand total:       {TOTAL * len(SOURCES)} images across {len(SOURCES)} classes")

    if dry_run:
        logger.info("\n  Run with --apply to actually fix the dataset!")
    else:
        if all_ok:
            logger.info("\n  ✅ Dataset fixed! Now re-run Stage 1:")
            logger.info("     python train_all.py --stages 1")
        else:
            logger.info("\n  ⚠️  Some sources had issues — check logs above.")


if __name__ == "__main__":
    main()

