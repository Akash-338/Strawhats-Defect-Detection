# 🚀 Friend Machine Setup Guide
# SafePath — RVCE Hackathon 2026
# Run these commands IN ORDER on the friend's machine

## ════════════════════════════════════════
## STEP 0 — Prerequisites (ONE TIME ONLY)
## ════════════════════════════════════════

# 1. Install Anaconda / Miniconda if not already installed
#    https://www.anaconda.com/download

# 2. Create environment
conda create -n RVCE python=3.11 -y
conda activate RVCE

# 3. Clone the repo
git clone https://github.com/Luci-699/Strawhats-Defect-Detection-2026.git
cd Strawhats-Defect-Detection-2026

# 4. Install dependencies
pip install -r requirements.txt

## ════════════════════════════════════════
## STEP 1 — Copy Datasets from USB / Drive
## ════════════════════════════════════════
# The datasets are NOT on GitHub (too large).
# Copy the entire data/ folder from the main machine.
#
# Files needed:
#   data/processed_aluminum/          (4,000 images)
#   data/processed/wood_10class/      (6,884 images)
#
# After copying, your data/ folder should look like:
#   data/
#   ├── processed_aluminum/
#   │   ├── images/train/  (~3200 images)
#   │   ├── images/val/
#   │   └── images/test/
#   └── processed/
#       └── wood_10class/
#           ├── train/images/  (~4819 images)
#           ├── val/images/
#           └── test/images/

## ════════════════════════════════════════
## STEP 2 — Create 3,000-image Subsets
## ════════════════════════════════════════

# Aluminum: 3,000 images (75% of 4,000)
python scripts/create_subset.py ^
    --src data/processed_aluminum ^
    --dst data/processed_aluminum_3k ^
    --fraction 0.75 ^
    --splits train

# Wood: 3,000 images (44% of 6,884)
python scripts/create_subset.py ^
    --src data/processed/wood_10class ^
    --dst data/processed/wood_3k ^
    --fraction 0.44 ^
    --splits train

# Verify counts (should show ~3000 each):
python -c "from pathlib import Path; print('Aluminum train:', len(list(Path('data/processed_aluminum_3k/images/train').glob('*')))); print('Wood train:', len(list(Path('data/processed/wood_3k/train/images').glob('*'))))"

## ════════════════════════════════════════
## STEP 3 — Run Training (Stages 3 + 4)
## ════════════════════════════════════════

python train_all.py --stages 3 4

# This runs:
#   Stage 3: Aluminum YOLOv10m — 3k images, 100 epochs (~5 hours)
#   Stage 4: Wood YOLOv10m     — 3k images, 100 epochs (~5 hours)
# Total: ~10 hours

## ════════════════════════════════════════
## STEP 4 — Run Fusion (After Stage 3+4 done)
## ════════════════════════════════════════

python train_all.py --stages 6 7

# This runs:
#   Stage 6: Aluminum Morphology Fusion (~2 hours)
#   Stage 7: Wood Morphology Fusion     (~2 hours)

## ════════════════════════════════════════
## STEP 5 — Copy Weights Back
## ════════════════════════════════════════
# After training, copy these files to the main machine:
#
#   runs/detect/aluminum/weights/best.pt
#   runs/detect/wood/weights/best.pt
#   runs/fusion/aluminum_fusion/best_fusion.pt  (if fusion done)
#   runs/fusion/wood_fusion/best_fusion.pt      (if fusion done)

## ════════════════════════════════════════
## COMMON ERRORS & FIXES
## ════════════════════════════════════════

# ERROR: "CUDA out of memory"
# FIX:   Reduce batch size
python train_all.py --stages 3 4 --batch 2

# ERROR: "No module named ultralytics"
# FIX:
pip install ultralytics

# ERROR: "FileNotFoundError: data/processed_aluminum_3k"
# FIX:   Run STEP 2 first to create the subsets

# ERROR: "path '../data/processed_aluminum' not found"
# FIX:   The YAML uses relative paths from the data/ folder.
#        Make sure you run all commands from the repo root:
#        cd Strawhats-Defect-Detection-2026
#        (NOT from inside data/ or any subfolder)

# ERROR: "workers=0 warning"
# FIX:   Normal on Windows — ignore it

# ERROR: "torch not compiled with CUDA"
# FIX:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
