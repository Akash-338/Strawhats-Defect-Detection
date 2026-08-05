"""
clean_cache.py
==============
Forcibly removes all Ultralytics labels.cache files from all processed dataset directories.
"""

import os
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def main():
    gc.collect()  # Release any dangling file handles
    
    cache_files = list(DATA_DIR.rglob("*.cache")) + list(DATA_DIR.rglob("*.cache.npy"))
    print(f"Found {len(cache_files)} cache file(s) to remove.")
    
    removed = 0
    for p in cache_files:
        try:
            if p.exists():
                os.chmod(p, 0o777)
                p.unlink()
                print(f"  [DELETED] {p}")
                removed += 1
        except Exception as e:
            print(f"  [ERROR] Could not delete {p}: {e}")
            
    print(f"\nSuccessfully removed {removed}/{len(cache_files)} cache files.")
    
    # Also verify total image count in steel train
    steel_train_img = DATA_DIR / "processed" / "steel" / "train" / "images"
    if steel_train_img.exists():
        imgs = list(steel_train_img.glob("*"))
        print(f"\nVerification: {len(imgs)} steel training images currently on disk at {steel_train_img}")

if __name__ == "__main__":
    main()
