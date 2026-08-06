# Strawhats Defect Detection 2026 🔍

**Strawhats Defect Detection 2026 | RVCE Hackathon 2026**
*Problem Statement 5: Automated Inspection System for Cracks & Fractures in Finished Materials*

---

## 🎯 Problem Statement

Quality control in manufacturing is currently performed **manually by operators**, leading to:
- Human fatigue → missed defects → product failures
- High rejection rates → factory losses
- Inconsistent results across shifts

**Our Solution:** A fully automated, real-time multi-material crack and defect inspection system using deep learning + morphological analysis.

---

## 🏗️ System Architecture

```
                        ┌─────────────────────────────────────┐
  Camera / Image  ──►   │   Material Classifier (ResNet18)    │
                        │        Steel  /  Wood               │
                        └────────────┬────────────────────────┘
                                     │ Routes to material-specific model
                      ┌──────────────┴───────────────┐
                      ▼                               ▼
            ┌──────────────────┐           ┌──────────────────┐
            │   YOLOv10m       │           │   YOLOv10m       │
            │  Steel Model     │           │   Wood Model     │
            │   15 classes     │           │   10 classes     │
            └────────┬─────────┘           └────────┬─────────┘
                     │                               │
                     └───────────────┬───────────────┘
                                     ▼
                      ┌──────────────────────────────┐
                      │   Morphology Fusion Layer    │
                      │  ┌─────────────────────────┐ │
                      │  │  11 Morphological       │ │
                      │  │  Descriptors (OpenCV)   │ │
                      │  └──────────┬──────────────┘ │
                      │             ▼                 │
                      │  MorphologyEncoder (MLP)      │
                      │  11D → 64D → 128D             │
                      │             ▼                 │
                      │  CrossAttentionFusion         │
                      │  Q=Visual, K=V=Morphology     │
                      │             ▼                 │
                      │  Classification Head          │
                      └──────────────────────────────┘
                                     ▼
                      ┌──────────────────────────────┐
                      │   Real-Time HUD Overlay      │
                      │   Defect Class + Confidence  │
                      │   Material + FPS Counter     │
                      └──────────────────────────────┘
```

---

## 📦 Dataset

| Material | Source | Images | Classes |
|----------|--------|--------|---------|
| 🔩 Steel | NEU-DET + GC10-DET | ~5,200 | 15 |
| 🪵 Wood | Kodytek Benchmark | ~7,400 | 10 |
| 🤖 Material Classifier | Steel + Wood combined | ~4,659 | 2 |
| **Total** | | **~15,600** | **25 Defect Types** |

### Steel Defect Classes (15)
`crazing` · `inclusion` · `patches` · `pitted_surface` · `rolled_in_scale` · `scratches` · `crease` · `crescent_gap` · `water_spot` · `welding_line` · `silk_spot` · `oil_spot` · `punching` · `rolling_pit` · `waist_fold`

### Wood Defect Classes (10)
`live_knot` · `dead_knot` · `knot_with_crack` · `crack` · `resin` · `marrow` · `quartzite` · `missing_knot` · `blue_stain` · `overgrown`

---

## 🧠 Technical Innovation

### 1. Morphology-Aware Detection (Novel Contribution)
Unlike standard YOLO-only systems, we augment visual detection with **11 mathematical morphological descriptors** computed per defect region:

| # | Descriptor | Captures |
|---|-----------|---------|
| 1 | Area | Defect size |
| 2 | Perimeter | Boundary length |
| 3 | Aspect Ratio | Elongation (cracks vs. spots) |
| 4 | Circularity | Shape regularity |
| 5 | Solidity | Convexity (pitting vs. crazing) |
| 6 | Convex Hull Perimeter | Outer boundary |
| 7 | Compactness | Area/ellipse ratio |
| 8 | Eccentricity | Stretch direction |
| 9 | Edge Density | Surface roughness |
| 10 | Skeleton Orientation | Crack direction angle |
| 11 | Texture Roughness | RMS surface deviation |

### 2. Cross-Attention Fusion
```
Q = W_Q · f_visual   (from YOLOv10m ROI features, dim=576)
K = W_K · f_morph    (from MorphologyEncoder, dim=128)
V = W_V · f_morph

fused = LayerNorm( Concat(Attention(Q,K,V), f_visual) )
```

### 3. DSP Preprocessing Pipeline
Each image undergoes: `CLAHE → Gaussian Blur → Otsu Threshold → Attention Map`
This enhances defect visibility before feeding to the model.

---

## 🚀 Quick Start

### Prerequisites
```bash
conda create -n RVCE python=3.11
conda activate RVCE
pip install -r requirements.txt
```

### Run Live Demo (Camera)
```bash
python inference/realtime_demo.py
```

### Run Live Demo (Image/Video)
```bash
python inference/realtime_demo.py --source path/to/image.jpg
python inference/realtime_demo.py --source path/to/video.mp4
```

### Demo Controls
| Key | Action |
|-----|--------|
| `SPACE` | Freeze frame for judges |
| `S` | Save screenshot |
| `+` / `-` | Adjust confidence threshold |
| `P` | Pause/Resume |
| `Q` | Quit |

### Train All Models (Full Pipeline)
```bash
# Stage 1: Material Classifier (ResNet18)        ~15 min
# Stage 2: Steel YOLO (150 epochs)               ~8 hrs
# Stage 3: Wood YOLO  (100 epochs, 3k subset)    ~7 hrs
# Stage 4: Steel Morphology Fusion (20 epochs)   ~3 hrs
# Stage 5: Wood  Morphology Fusion (20 epochs)   ~2 hrs
python train_all.py --stages 1 2 3 4 5
```

### Train Specific Stages
```bash
python train_all.py --stages 2 3        # YOLO only
python train_all.py --stages 4 5        # Fusion only
python train_all.py --stages 4          # Steel fusion only
```

### Evaluate
```bash
python evaluation/evaluate.py
```

---

## 📁 Project Structure

```
Strawhats-Defect-Detection-2026/
│
├── configs/                       # Dataset YAML files (Git-tracked)
│   ├── dataset_steel.yaml         # Steel: 15 classes, 5232 images
│   └── dataset_wood_3k.yaml       # Wood: 10 classes, 3k subset
│
├── data/                          # Datasets (gitignored — download separately)
│   ├── processed/steel_unified/   # Steel: 15 classes, 5232 images
│   ├── processed/wood_10class/    # Wood: 10 classes, 7393 images
│   ├── processed/wood_3k/         # Wood 3k training subset
│   └── material_classifier/       # 2-class routing dataset, 4659 images
│
├── models/                        # Neural network modules
│   ├── yolo_backbone.py           # YOLOv10m frozen feature extractor
│   ├── cross_attention.py         # CrossAttentionFusion module
│   ├── material_router.py         # ResNet18 material classifier
│   └── classification_head.py     # Morphology classifier head
│
├── morphology/                    # Morphological analysis
│   ├── preprocessing.py           # CLAHE + Gaussian pipeline
│   ├── feature_extractor.py       # 11 morphological descriptors
│   ├── encoder.py                 # MLP: 11D → 128D embedding
│   ├── attention_map.py           # Binary mask → attention map
│   └── dsp_filters.py             # DSP preprocessing filters
│
├── training/                      # Training scripts
│   ├── config.yaml                # All hyperparameters
│   ├── train_yolo_baseline.py     # YOLOv10m training (Stages 2-3)
│   ├── train_morphology_fusion.py # Fusion training (Stages 4-5)
│   ├── train_material_classifier.py # ResNet18 (Stage 1)
│   └── losses.py                  # MorphologyAwareLoss
│
├── inference/
│   ├── realtime_demo.py           # Live demo with HUD overlay
│   ├── detect.py                  # Single image detection
│   ├── batch_detect.py            # Batch processing
│   └── api_server.py              # FastAPI WebSocket server
│
├── evaluation/
│   ├── evaluate.py                # mAP50, accuracy, per-class metrics
│   ├── ablation.py                # Ablation study runner
│   └── explainability.py          # Grad-CAM + SHAP visualizations
│
├── hardware/
│   ├── esp32_reject.ino           # ESP32 firmware (Red LED + Buzzer + Servo)
│   ├── serial_bridge.py           # Python → ESP32 serial bridge
│   └── wiring_diagram.md          # GPIO pinout diagram
│
├── scripts/                       # Data preparation utilities
│   ├── create_subset.py           # Stratified subset creation
│   ├── download_neu_det.py
│   ├── download_severstal.py
│   ├── convert_annotations.py
│   └── split_dataset.py
│
├── demo/
│   └── print_cards/               # Printed sample cards for demo day
│       ├── print_sheet.html       # A4 print-ready HTML
│       ├── Steel/                 # 6 steel defect cards
│       └── Wood/                  # 9 wood defect cards
│
├── train_all.py                   # Master training pipeline (5 stages)
└── requirements.txt
```

---

## 📊 Training Configuration

| Parameter | Steel YOLO | Wood YOLO | Fusion (Both) |
|-----------|-----------|-----------|---------------|
| Model | YOLOv10m | YOLOv10m | CrossAttn Head |
| Image Size | 800×800 | 800×800 | 640×640 |
| Batch Size | 4 | 4 | 8 |
| Epochs | **150** | **100** | **20** |
| Dataset | Full (5,232) | 3k subset | Same as YOLO |
| Optimizer | AdamW | AdamW | AdamW |
| LR | 0.005 | 0.001 | 1e-4 |
| AMP | FP16 ✅ | FP16 ✅ | FP16 ✅ |

---

## 📈 Results

| Model | Metric | Target | Status |
|-------|--------|--------|--------|
| Material Classifier (ResNet18) | Accuracy | > 95% | ✅ Done (~97%) |
| Steel YOLOv10m (150ep, 5232 imgs) | mAP50 | > 85% | ✅ Training Done |
| Wood YOLOv10m (100ep, 3k subset) | mAP50 | > 68% | 🔄 Training (Friend 1) |
| Steel Morphology Fusion (20ep) | mAP50 | > 88% | 🔄 Training (Friend 2) |
| Wood Morphology Fusion (20ep) | mAP50 | > 72% | ⏳ Queued |

> *Full evaluation results available after training completes. See `evaluation/evaluate.py`.*

---

## 👥 Team

**Team Strawhat-Pirates | RVCE Hackathon 2026**

| Member | Role |
|--------|------|
| Faizan | Model architecture + training pipeline |
| Pulkit | Data preprocessing + augmentation |
| Chethan | Morphology module + fusion |
| Rahul | Hardware (ESP32) + demo |
| Akash | Evaluation + presentation |

---

## 🏆 Judging Criteria Coverage

| Criterion | Marks | Our Implementation |
|-----------|-------|-------------------|
| **Problem Understanding** | 10 | Factory QC automation for Steel & Wood, directly mirrors PS-5 |
| **Technical Complexity** | 10 | YOLOv10m + Morphology Fusion + Cross-Attention — 5-stage pipeline |
| **Feasibility** | 10 | Runs real-time at 15-30 FPS on RTX 4050 |
| **Functionality** | 10 | Detects 25 defect classes across Steel & Wood |
| **Code Quality** | 20 | Modular architecture, docstrings, type hints, config-driven |
| **Teamwork** | 10 | Git history with tracked contributions across 5 members |
| **Model Training** | 10 | 150-epoch steel + 20-epoch fusion with augmentation + early stopping |
| **Output Response** | 10 | Live HUD: class label, confidence, FPS, defect count |
| **Demo** | 10 | Webcam demo + physical sample cards + ESP32 reject arm |
| **TOTAL** | **100** | **Target: 85-96/100** |
