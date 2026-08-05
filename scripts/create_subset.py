"""
create_subset.py
Creates a stratified subset of a YOLO dataset by fraction.
Used to speed up training of Aluminum and Wood stages.

Usage:
    python scripts/create_subset.py --src data/processed_aluminum \
           --dst data/processed_aluminum_50pct --fraction 0.5
"""

import argparse
import random
import shutil
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_class_from_label(label_path: Path) -> int:
    """Return the first class id found in a YOLO label file."""
    try:
        # Explicit utf-8 with errors='ignore' — prevents hanging on Windows
        content = label_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            return -1
        first_line = content.splitlines()[0]
        return int(first_line.split()[0])
    except Exception:
        try:
            # Binary fallback for any encoding edge case
            raw = label_path.read_bytes().decode("latin-1", errors="ignore").strip()
            if raw:
                return int(raw.splitlines()[0].split()[0])
        except Exception:
            pass
    return -1


def stratified_subset(image_dir: Path, label_dir: Path, fraction: float):
    """
    Returns a stratified subset of (image, label) pairs by class.
    Fraction is applied per-class so rare classes are preserved.
    """
    # Build class -> list of (img, lbl) pairs
    class_groups: dict[int, list] = defaultdict(list)
    unmatched = []

    for img_path in sorted(image_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        lbl_path = label_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            unmatched.append((img_path, None))
            continue
        cls = get_class_from_label(lbl_path)
        class_groups[cls].append((img_path, lbl_path))

    selected = []
    for cls, pairs in class_groups.items():
        random.shuffle(pairs)
        n = max(1, int(len(pairs) * fraction))
        selected.extend(pairs[:n])
        logger.info(f"  Class {cls:2d}: {len(pairs):4d} → keeping {n:4d}")

    return selected


def copy_subset(pairs, src_root: Path, dst_root: Path, split: str):
    """Copy selected image/label pairs into dst_root/split/images and labels."""
    img_out = dst_root / split / "images"
    lbl_out = dst_root / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, img_out / img_path.name)
        if lbl_path and lbl_path.exists():
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

    logger.info(f"  Copied {len(pairs)} samples → {dst_root / split}")


def copy_yaml(src_root: Path, dst_root: Path):
    """Copy and patch the dataset YAML to point at new path."""
    for yaml_file in src_root.glob("*.yaml"):
        content = yaml_file.read_text()
        # Update path to new location
        content = content.replace(
            f"path: {src_root}",
            f"path: {dst_root}"
        )
        out_yaml = dst_root / yaml_file.name
        out_yaml.write_text(content)
        logger.info(f"  Copied YAML → {out_yaml}")
        break  # only first yaml


def main():
    parser = argparse.ArgumentParser(description="Create a stratified subset of a YOLO dataset")
    parser.add_argument("--src", required=True, help="Source dataset root (has train/val/test)")
    parser.add_argument("--dst", required=True, help="Destination dataset root")
    parser.add_argument("--fraction", type=float, default=0.5, help="Fraction to keep (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--splits", nargs="+", default=["train"], help="Which splits to subset (default: train only)")
    args = parser.parse_args()

    random.seed(args.seed)

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating {args.fraction*100:.0f}% subset: {src} → {dst}")

    for split in args.splits:
        logger.info(f"\n[{split.upper()}]")

        # Detect image/label directory structure
        # Try train/images or images/train
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"

        if not img_dir.exists():
            img_dir = src / "images" / split
            lbl_dir = src / "labels" / split

        if not img_dir.exists():
            logger.warning(f"  Could not find images for split '{split}' in {src}")
            continue

        if split == "train":
            pairs = stratified_subset(img_dir, lbl_dir, args.fraction)
        else:
            # Keep val and test FULLY intact — don't reduce them
            logger.info(f"  Keeping {split} split intact (full data)")
            pairs = []
            for img_path in sorted(img_dir.glob("*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                pairs.append((img_path, lbl_path if lbl_path.exists() else None))

        copy_subset(pairs, src, dst, split)

    # Copy val and test fully if not already done
    for split in ["val", "test"]:
        if split not in args.splits:
            for variant in [src / split, src / "images" / split]:
                if variant.exists():
                    img_dir = src / split / "images" if (src / split / "images").exists() else src / "images" / split
                    lbl_dir = src / split / "labels" if (src / split / "labels").exists() else src / "labels" / split
                    pairs = []
                    for img_path in sorted(img_dir.glob("*")):
                        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                            continue
                        lbl_path = lbl_dir / (img_path.stem + ".txt")
                        pairs.append((img_path, lbl_path if lbl_path.exists() else None))
                    copy_subset(pairs, src, dst, split)
                    break

    copy_yaml(src, dst)

    # Print summary
    total = sum(
        len(list((dst / s / "images").glob("*")))
        for s in ["train", "val", "test"]
        if (dst / s / "images").exists()
    )
    logger.info(f"\n✅ Subset created: {dst}")
    logger.info(f"   Total images: {total}")


if __name__ == "__main__":
    main()
