import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix Anaconda + PyTorch OpenMP conflict on Windows

# Fix Windows encoding for console output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

"""
train_all.py
============
One-command overnight training pipeline for the Multi-Material Crack Inspection System.
Runs all training stages sequentially -- just launch and go to sleep!

Usage:
    python train_all.py                    # Full pipeline (GPU)
    python train_all.py --device cpu       # CPU mode (slow, for testing)
    python train_all.py --skip-material    # Skip material classifier
    python train_all.py --stages 1 2       # Run specific stages only
    
Stages:
    1 - Material Classifier (ResNet18, 3 classes, ~15 min)
    2 - Steel YOLO v2 (NEU-DET, 150 epochs, imgsz=800, mixup+copy_paste, ~8 hrs)
    3 - Aluminum YOLO v2 (10 defect types, 150 epochs, imgsz=800, ~8 hrs)
    4 - Wood YOLO v2 (10 defect types, 150 epochs, imgsz=800, ~6 hrs)
    5 - Steel Morphology Fusion (cross-attention, 100 epochs, ~8 hrs)
    6 - Aluminum Morphology Fusion (cross-attention, 100 epochs, ~8 hrs)
    7 - Wood Morphology Fusion (cross-attention, 100 epochs, ~6 hrs)
    
Total estimated time: ~44 hours on RTX 4050 (6GB) — run overnight!
"""

import argparse
import logging
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging with UTF-8 encoding to handle special characters on Windows
log_handlers = [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler('training_log.txt', mode='a', encoding='utf-8')
]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).parent
TRAINING_DIR = PROJECT_ROOT / "training"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"

STAGE_CONFIG = {
    1: {
        "name": "Material Classifier (ResNet18)",
        "script": "training/train_material_classifier.py",
        "args": [],
        "estimated_minutes": 15,
        "description": "2-class classifier: Steel vs Wood",
    },
    2: {
        "name": "Steel YOLO v2 (High-Accuracy)",
        "script": "training/train_yolo_baseline.py",
        "args": ["--data", "configs/dataset_steel.yaml", "--epochs", "150", "--batch", "4", "--imgsz", "800", "--name", "steel"],
        "estimated_minutes": 480,
        "description": "YOLOv10m on Unified Steel (15 classes, 150ep, imgsz=800, mixup+copy_paste) - target >85% mAP50",
    },
    3: {
        "name": "Wood YOLO v2 (3k-subset, 100ep)",
        "script": "training/train_yolo_baseline.py",
        "args": ["--data", "configs/dataset_wood_3k.yaml", "--epochs", "100", "--batch", "4", "--imgsz", "800", "--name", "wood", "--lr", "0.001"],
        "estimated_minutes": 300,
        "description": "YOLOv10m on Wood Defects (10 classes, 3k subset, 100ep, imgsz=800) - target >68% mAP50",
    },
    4: {
        "name": "Morphology Fusion — Steel",
        "script": "training/train_morphology_fusion.py",
        "args": ["--material", "steel", "--data", "configs/dataset_steel.yaml", "--batch", "8", "--epochs", "20"],
        "estimated_minutes": 180,
        "description": "Cross-attention fusion: Steel YOLO + morphology descriptors (20 epochs)",
    },
    5: {
        "name": "Morphology Fusion — Wood",
        "script": "training/train_morphology_fusion.py",
        "args": ["--material", "wood", "--data", "configs/dataset_wood_3k.yaml", "--batch", "8", "--epochs", "20"],
        "estimated_minutes": 120,
        "description": "Cross-attention fusion: Wood YOLO + morphology descriptors (20 epochs)",
    },
}


def print_banner():
    """Print a fancy startup banner."""
    banner = """
+==============================================================+
|                                                              |
|   MULTI-MATERIAL CRACK INSPECTION SYSTEM                     |
|   ------------------------------------------                 |
|   Overnight Training Pipeline v1.0                           |
|                                                              |
|   Materials: Steel | Wood                           |
|   Architecture: Material Router -> Material-Specific YOLO    |
|   Dataset: ~16,000 images across 2 materials                 |
|                                                              |
|   RVCE Hackathon 2026 -- Team SafePath                       |
|                                                              |
+==============================================================+
    """
    print(banner)


def check_gpu():
    """Check if CUDA GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"[OK] GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
            return True
        else:
            logger.warning("[WARN] No CUDA GPU detected. Training will be SLOW on CPU.")
            logger.warning("       Fix: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            return False
    except ImportError:
        logger.warning("[WARN] PyTorch not installed. Run: pip install -r requirements.txt")
        return False


def check_datasets():
    """Verify all required datasets exist."""
    datasets = {
        "Steel Unified (15-Class)": DATA_DIR / "processed" / "steel_unified",
        "Aluminum (10-Class)": DATA_DIR / "processed_aluminum",
        "Wood (10-Class)": DATA_DIR / "processed" / "wood_10class",
        "Material Classifier (3-Class)": DATA_DIR / "material_classifier",
    }
    
    all_ok = True
    for name, path in datasets.items():
        if path.exists():
            count = len([f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp')])
            logger.info(f"  [OK] {name}: {count} images found at {path.name}")
            if count == 0:
                logger.warning(f"       WARNING: {name} directory exists but has no images!")
                all_ok = False
        else:
            logger.error(f"  [MISSING] {name}: NOT FOUND at {path}")
            all_ok = False
    
    return all_ok


def run_stage(stage_num: int, device: str) -> bool:
    """Run a single training stage."""
    config = STAGE_CONFIG[stage_num]
    
    total_stages = max(STAGE_CONFIG.keys())
    logger.info("=" * 60)
    logger.info(f"STAGE {stage_num}/{total_stages}: {config['name']}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Estimated time: ~{config['estimated_minutes']} minutes")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Purge any stale Ultralytics .cache files before YOLO stages
    if stage_num in (2, 3, 4):
        if sys.platform == 'win32':
            os.system('del /f /q /s data\\*.cache >nul 2>&1')
        for c in PROJECT_ROOT.glob("data/**/*.cache"):
            try:
                c.unlink()
            except Exception:
                pass
    
    # Build command
    cmd = [sys.executable, config["script"]] + config["args"]
    
    # Add device arg if the script supports it
    if stage_num >= 2 and stage_num <= 4:
        cmd.extend(["--device", device])
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,  # Show output in real-time
            text=True,
        )
        
        elapsed = (time.time() - start_time) / 60
        
        if result.returncode == 0:
            logger.info(f"[OK] Stage {stage_num} COMPLETE in {elapsed:.1f} minutes")
            return True
        else:
            logger.error(f"[FAIL] Stage {stage_num} FAILED after {elapsed:.1f} minutes (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        elapsed = (time.time() - start_time) / 60
        logger.error(f"[FAIL] Stage {stage_num} ERROR after {elapsed:.1f} minutes: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="One-command overnight training pipeline for Multi-Material Crack Inspection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_all.py                    # Full pipeline on GPU
  python train_all.py --device cpu       # CPU mode
  python train_all.py --stages 1 2       # Only material classifier + steel
  python train_all.py --stages 3 4       # Only aluminum + wood
  python train_all.py --skip-fusion      # Skip morphology fusion (stage 5)
        """
    )
    parser.add_argument("--device", type=str, default="0", help="Device: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--stages", nargs="+", type=int, default=None, help="Specific stages to run (1-5)")
    parser.add_argument("--skip-material", action="store_true", help="Skip material classifier (stage 1)")
    parser.add_argument("--skip-fusion", action="store_true", help="Skip morphology fusion (stage 5)")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue even if a stage fails")
    args = parser.parse_args()
    
    print_banner()
    
    # Determine which stages to run
    if args.stages:
        stages = sorted(args.stages)
    else:
        stages = [1, 2, 3, 4, 5, 6, 7]
        if args.skip_material:
            stages.remove(1)
        if args.skip_fusion:
            stages = [s for s in stages if s not in (5, 6, 7)]
    
    # Pre-flight checks
    logger.info("[PRE-FLIGHT] Checking GPU...")
    has_gpu = check_gpu()
    if args.device == "0" and not has_gpu:
        logger.warning("Falling back to CPU mode.")
        args.device = "cpu"
    
    logger.info("")
    logger.info("[PRE-FLIGHT] Verifying datasets...")
    datasets_ok = check_datasets()
    if not datasets_ok:
        logger.error("[ERROR] Some datasets are missing or empty. Run conversion scripts first.")
        logger.error("        See README.md for instructions.")
        sys.exit(1)
    
    # Calculate total estimated time
    total_est = sum(STAGE_CONFIG[s]["estimated_minutes"] for s in stages)
    est_finish = datetime.now() + timedelta(minutes=total_est)
    
    logger.info("")
    logger.info(f"Training Plan: Stages {stages}")
    logger.info(f"Estimated total time: ~{total_est} minutes ({total_est/60:.1f} hours)")
    logger.info(f"Estimated finish: {est_finish.strftime('%Y-%m-%d %H:%M')}")
    logger.info("")
    logger.info("Starting training pipeline...")
    logger.info("")
    
    # Run stages
    overall_start = time.time()
    results = {}
    
    for stage_num in stages:
        success = run_stage(stage_num, args.device)
        results[stage_num] = success
        
        if not success and not args.continue_on_error:
            logger.error(f"Pipeline stopped at stage {stage_num}. Use --continue-on-error to skip failures.")
            break
        
        logger.info("")
    
    # Summary
    total_elapsed = (time.time() - overall_start) / 60
    
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    
    for stage_num, success in results.items():
        status = "PASS" if success else "FAIL"
        logger.info(f"  Stage {stage_num} ({STAGE_CONFIG[stage_num]['name']}): {status}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    logger.info("")
    logger.info(f"Results: {passed}/{total} stages passed")
    logger.info(f"Total time: {total_elapsed:.1f} minutes ({total_elapsed/60:.1f} hours)")
    logger.info("")
    
    if passed == total:
        logger.info("ALL STAGES COMPLETE! Models saved in runs/ directory.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Check runs/ for trained model weights")
        logger.info("  2. Run evaluation: python evaluation/evaluate_model.py")
        logger.info("  3. Export to TensorRT: python inference/tensorrt_export.py")
        logger.info("  4. Run demo: python inference/realtime_inference.py")
    else:
        logger.error("Some stages failed. Check training_log.txt for details.")
    
    # Save results summary
    summary_path = PROJECT_ROOT / "training_results.txt"
    with open(summary_path, "w") as f:
        f.write(f"Training completed at: {datetime.now().isoformat()}\n")
        f.write(f"Total time: {total_elapsed:.1f} minutes\n")
        f.write(f"Stages: {passed}/{total} passed\n\n")
        for stage_num, success in results.items():
            f.write(f"Stage {stage_num} ({STAGE_CONFIG[stage_num]['name']}): {'PASS' if success else 'FAIL'}\n")
    
    logger.info(f"Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
