"""
scripts/fetch_internet_samples.py
==================================
Fetch real-world defect images from public sources, run InferencePipeline,
and save annotated detection results into demo_samples/internet_test/
"""

import os
import sys
import cv2
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.pipeline import InferencePipeline

# Direct public image URLs of real-world metal & wood defects
SAMPLE_URLS = [
    ("internet_steel_scratches.jpg", "https://raw.githubusercontent.com/kaustubhdikshit/NEU-Surface-Defect-Database/master/NEU-DET/train/images/scratches/scratches_15.jpg"),
    ("internet_steel_patches.jpg", "https://raw.githubusercontent.com/kaustubhdikshit/NEU-Surface-Defect-Database/master/NEU-DET/train/images/patches/patches_25.jpg"),
    ("internet_steel_pitted.jpg", "https://raw.githubusercontent.com/kaustubhdikshit/NEU-Surface-Defect-Database/master/NEU-DET/train/images/pitted_surface/pitted_surface_35.jpg"),
    ("internet_steel_punching.jpg", "https://raw.githubusercontent.com/kaustubhdikshit/NEU-Surface-Defect-Database/master/NEU-DET/train/images/punching/punching_45.jpg"),
]

def main():
    out_dir = ROOT / 'demo_samples' / 'internet_test'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pipe = InferencePipeline()
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    print("=" * 60)
    print("  TESTING MODEL ON INTERNET / EXTERNAL DEFECT IMAGES")
    print("=" * 60)
    
    for filename, url in SAMPLE_URLS:
        raw_path = out_dir / filename
        pred_path = out_dir / f"PRED_{filename}"
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp, open(raw_path, 'wb') as f:
                f.write(resp.read())
                
            frame = cv2.imread(str(raw_path))
            if frame is None:
                print(f"❌ Failed to decode {filename}")
                continue
                
            res = pipe.run(frame, conf_threshold=0.20)
            annotated = res.get('annotated', frame)
            
            cv2.imwrite(str(pred_path), annotated)
            
            v = res['verdict']
            cls = str(res['top_class'])
            pct = res['top_conf_pct']
            cnt = res['defect_count']
            
            print(f"  {filename:30s} -> Verdict: {v:6s} | Defect: {cls:15s} | Conf: {pct}% | Count: {cnt}")
            
        except Exception as e:
            print(f"❌ Error testing {filename}: {e}")

    print("=" * 60)
    print(f"✅ Annotated predictions saved in {out_dir}")

if __name__ == '__main__':
    main()
