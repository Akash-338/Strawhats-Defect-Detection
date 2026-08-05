"""
merge_and_split_steel.py
========================
Merges the already-converted Severstal patches (data/processed/severstal/)
with NEU-DET images (data/processed/neu-det/) into a single combined steel
dataset, then performs a stratified 70/15/15 split into:

    data/processed/steel/train/images+labels
    data/processed/steel/val/images+labels
    data/processed/steel/test/images+labels

This replaces the current 1,257-image steel split with ~15,000+ images.

Usage:
    python scripts/merge_and_split_steel.py          # dry run first (shows counts)
    python scripts/merge_and_split_steel.py --apply  # actually writes files
"""

import argparse
import logging
import shutil
import random
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
SEVERSTAL_DIR = PROJECT_ROOT / "data" / "processed" / "severstal"
NEU_DET_DIR   = PROJECT_ROOT / "data" / "processed" / "neu-det"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "processed" / "steel"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = remaining 0.15

RANDOM_SEED = 42


def collect_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    """Collect (image_path, label_path) pairs from a processed directory."""
    img_dir = source_dir / "images"
    lbl_dir = source_dir / "labels"

    if not img_dir.exists():
        logger.warning(f"No images/ folder in {source_dir} — skipping.")
        return []

    pairs = []
    for img in img_dir.glob("*"):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            logger.warning(f"No label for {img.name} — skipping.")

    logger.info(f"  {source_dir.name}: {len(pairs)} image-label pairs found")
    return pairs


def get_primary_class(label_path: Path) -> int:
    """Return first class id found in a YOLO label file, or -1."""
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    return int(parts[0])
    except Exception:
        pass
    return -1


def stratified_split(pairs: list[tuple[Path, Path]]) -> dict:
    """Stratify pairs by class, split 70/15/15."""
    by_class = defaultdict(list)
    for pair in pairs:
        cls = get_primary_class(pair[1])
        by_class[cls].append(pair)

    splits = {"train": [], "val": [], "test": []}
    rng = random.Random(RANDOM_SEED)

    for cls, class_pairs in sorted(by_class.items()):
        rng.shuffle(class_pairs)
        n = len(class_pairs)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits["train"].extend(class_pairs[:n_train])
        splits["val"].extend(class_pairs[n_train:n_train + n_val])
        splits["test"].extend(class_pairs[n_train + n_val:])
        logger.info(f"    class {cls:2d}: {n:5d} total → {n_train} train / {int(n*VAL_RATIO)} val / {n - n_train - n_val} test")

    return splits


def write_splits(splits: dict, dry_run: bool = True) -> None:
    """Copy image+label files into the output train/val/test folders."""
    for split_name, pairs in splits.items():
        out_img = OUTPUT_DIR / split_name / "images"
        out_lbl = OUTPUT_DIR / split_name / "labels"

        if not dry_run:
            # Clear existing split to avoid stale NEU-DET-only files
            if (OUTPUT_DIR / split_name).exists():
                shutil.rmtree(OUTPUT_DIR / split_name)
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

        # Delete any leftover Ultralytics cache files in OUTPUT_DIR and parent
        if not dry_run and OUTPUT_DIR.exists():
            for cache in OUTPUT_DIR.rglob("*.cache"):
                try:
                    cache.unlink()
                except Exception:
                    pass

        logger.info(f"  {split_name}: {len(pairs)} images {'(would write)' if dry_run else '(writing...)'}")

        if not dry_run:
            for img_path, lbl_path in pairs:
                # Prefix filename with source to avoid name collisions
                new_name = img_path.name
                dest_img = out_img / new_name
                dest_lbl = out_lbl / (Path(new_name).stem + ".txt")

                # Handle filename collision (NEU-DET and Severstal may share names)
                counter = 1
                while dest_img.exists():
                    stem = img_path.stem + f"_{counter}"
                    new_name = stem + img_path.suffix
                    dest_img = out_img / new_name
                    dest_lbl = out_lbl / (stem + ".txt")
                    counter += 1

                shutil.copy2(img_path, dest_img)
                shutil.copy2(lbl_path, dest_lbl)


def main():
    parser = argparse.ArgumentParser(description="Merge NEU-DET + Severstal → steel split")
    parser.add_argument("--apply", action="store_true",
                        help="Actually write files (default: dry-run shows counts only)")
    parser.add_argument("--severstal-dir", type=Path, default=SEVERSTAL_DIR,
                        help=f"Path to converted Severstal data (default: {SEVERSTAL_DIR})")
    parser.add_argument("--neu-det-dir", type=Path, default=NEU_DET_DIR,
                        help=f"Path to converted NEU-DET data (default: {NEU_DET_DIR})")
    args = parser.parse_args()

    dry_run = not args.apply

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE — no files will be written.")
        logger.info("Run with --apply to actually merge and split.")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("APPLYING — files will be written to data/processed/steel/")
        logger.info("=" * 60)

    # Collect from both sources
    logger.info("\n[1/3] Collecting image-label pairs...")
    severstal_pairs = collect_pairs(args.severstal_dir)
    if not severstal_pairs:
        logger.info("  Severstal processed data not found in data/processed/severstal. Processing now...")
        import subprocess
        subprocess.run([sys.executable, "scripts/download_severstal.py"], check=False)
        severstal_pairs = collect_pairs(args.severstal_dir)

    neu_det_pairs   = collect_pairs(args.neu_det_dir)
    all_pairs = severstal_pairs + neu_det_pairs

    logger.info(f"\n  TOTAL combined: {len(all_pairs)} pairs")
    logger.info(f"  Severstal:  {len(severstal_pairs)}")
    logger.info(f"  NEU-DET:    {len(neu_det_pairs)}")

    if len(all_pairs) == 0:
        logger.error("No pairs found! Check your source directories.")
        return

    # Stratified split
    logger.info("\n[2/3] Stratified split (70 / 15 / 15)...")
    splits = stratified_split(all_pairs)

    # Write (or preview)
    logger.info(f"\n[3/3] Output → {OUTPUT_DIR}")
    write_splits(splits, dry_run=dry_run)

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for split_name, pairs in splits.items():
        logger.info(f"  {split_name:5s}: {len(pairs):6,} images")
    logger.info(f"  TOTAL: {sum(len(v) for v in splits.values()):6,} images")

    if dry_run:
        logger.info("\nRun with --apply to write these files!")
    else:
        logger.info(f"\nDone! Now retrain with:")
        logger.info(f"  python train_all.py --stages 2 3 4")


if __name__ == "__main__":
    main()
