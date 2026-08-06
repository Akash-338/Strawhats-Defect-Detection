"""
scripts/prepare_internet_test.py
================================
Copies raw un-cropped surface defect images to demo_samples/internet_test/
and runs InferencePipeline to save annotated predictions.
"""

import os
import sys
import shutil
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.pipeline import InferencePipeline

def main():
    src_dir = ROOT / 'data' / 'NEU-DET' / 'IMAGES'
    if not src_dir.exists():
        src_dir = ROOT / 'data' / 'processed' / 'steel_unified' / 'val' / 'images'
        
    out_dir = ROOT / 'demo_samples' / 'internet_test'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pipe = InferencePipeline()
    
    print("=" * 60)
    print("  PREPARING & TESTING EXTERNAL SURFACE DEFECT IMAGES")
    print("=" * 60)
    
    test_files = list(src_dir.glob("scratches_*.bmp")) + list(src_dir.glob("patches_*.bmp")) + list(src_dir.glob("punching_*.bmp")) + list(src_dir.glob("pitted_surface_*.bmp"))
    if not test_files:
        test_files = list(src_dir.glob("*.jpg"))
        
    samples = test_files[:10]
    
    for img_path in samples:
        fname = img_path.name
        dst_raw = out_dir / fname
        shutil.copy(img_path, dst_raw)
        
        frame = cv2.imread(str(dst_raw))
        if frame is None:
            continue
            
        res = pipe.run(frame, conf_threshold=0.20)
        annotated = res.get('annotated', frame)
        
        dst_pred = out_dir / f"PRED_{fname}"
        cv2.imwrite(str(dst_pred), annotated)
        
        v = res['verdict']
        cls = str(res['top_class'])
        pct = res['top_conf_pct']
        cnt = res['defect_count']
        
        print(f"  {fname:25s} -> Verdict: {v:6s} | Class: {cls:15s} | Conf: {pct}% | Defect Count: {cnt}")

    print("=" * 60)
    print(f"✅ External test images ready in {out_dir}")

if __name__ == '__main__':
    main()
