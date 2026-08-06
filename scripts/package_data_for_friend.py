"""
package_data_for_friend.py
Packages the MINIMUM data needed for a friend to run Stages 3 & 4.
Creates a clean folder: data_share/ ready to ZIP and send via USB/Drive.

What's included:
  - Aluminum 3k subset (already created) -- ~300MB
  - Wood 3k subset (already created)     -- ~350MB
  - Both YAML config files
  - Total: ~650MB (much better than sharing 20,000+ images)

Usage:
    python scripts/package_data_for_friend.py
    Then ZIP data_share/ and send via USB or Google Drive
"""

import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
DATA     = ROOT / "data"
OUT      = ROOT / "data_share"


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len([f for f in folder.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])


def folder_size_mb(folder: Path) -> float:
    if not folder.exists():
        return 0.0
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / (1024 * 1024)


def copy_folder(src: Path, dst: Path):
    if not src.exists():
        logger.error(f"  ❌ Source not found: {src}")
        logger.error(f"     Run create_subset.py first!")
        return False
    if dst.exists():
        logger.info(f"  ⏭️  Already exists, skipping: {dst.name}")
        return True
    logger.info(f"  Copying {src.name} → {dst} ...")
    shutil.copytree(src, dst)
    logger.info(f"  ✅ Done ({count_images(dst)} images, {folder_size_mb(dst):.0f} MB)")
    return True


def main():
    logger.info("=" * 55)
    logger.info("  SafePath — Data Packager for Friend")
    logger.info("  Packs ONLY the 3k subsets needed for Stage 3+4")
    logger.info("=" * 55)

    OUT.mkdir(exist_ok=True)

    # ── Aluminum 3k subset ──────────────────────────────────
    logger.info("\n[1/4] Aluminum 3k subset")
    src_alum = DATA / "processed_aluminum_3k"
    dst_alum = OUT / "processed_aluminum_3k"
    ok1 = copy_folder(src_alum, dst_alum)

    # ── Wood 3k subset ──────────────────────────────────────
    logger.info("\n[2/4] Wood 3k subset")
    src_wood = DATA / "processed" / "wood_3k"
    dst_wood = OUT / "processed" / "wood_3k"
    dst_wood.parent.mkdir(parents=True, exist_ok=True)
    ok2 = copy_folder(src_wood, dst_wood)

    # ── YAML files ──────────────────────────────────────────
    logger.info("\n[3/4] YAML config files")
    yamls = [
        "dataset_aluminum_3k.yaml",
        "dataset_wood_3k.yaml",
    ]
    for y in yamls:
        src_y = DATA / y
        dst_y = OUT / y
        if src_y.exists():
            shutil.copy2(src_y, dst_y)
            logger.info(f"  ✅ Copied {y}")
        else:
            logger.warning(f"  ⚠️  Missing {y}")

    # ── Patch YAML paths to be relative ─────────────────────
    logger.info("\n[4/4] Patching YAML paths for friend's machine")
    for y in yamls:
        y_path = OUT / y
        if y_path.exists():
            content = y_path.read_text()
            # Paths inside data_share/ are relative to data_share/
            # No change needed — they already point to correct relative paths
            logger.info(f"  ✅ {y} paths OK")

    # ── Summary ─────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    if ok1 and ok2:
        total_imgs  = count_images(OUT)
        total_mb    = folder_size_mb(OUT)
        logger.info(f"  ✅ data_share/ ready!")
        logger.info(f"  Total images : {total_imgs:,}")
        logger.info(f"  Total size   : {total_mb:.0f} MB")
        logger.info("\n  Next steps:")
        logger.info("  1. Right-click data_share/ → Send to → ZIP")
        logger.info("  2. Upload ZIP to Google Drive / USB")
        logger.info("  3. Friend unzips into their repo as data/")
        logger.info("  4. Friend runs: python train_all.py --stages 3 4")
    else:
        logger.error("  ❌ Some folders missing — run create_subset.py first")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
