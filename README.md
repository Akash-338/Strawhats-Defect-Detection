# Strawhats Defect Detection 2026 🔍

**Strawhats Defect Detection 2026 | RVCE Hackathon 2026**  
*Problem Statement 5: Automated Inspection System for Cracks & Fractures in Finished Materials*

---

## 🎯 Problem Statement

Quality control in industrial manufacturing is currently performed **manually by operators**, leading to:
- Human fatigue → missed defects → product failures & safety hazards
- High rejection rates → costly manufacturing scrap & factory losses
- Inconsistent inspection quality across shifts

**Our Solution:** A fully automated, real-time multi-material defect inspection system combining **YOLOv10 Deep Learning**, **Mathematical Morphological Analysis**, **Morphological Line Fallbacks**, and an **ESP32 Industrial Reject Actuator** controlled via a modern web UI dashboard.

---

## 🏗️ System Architecture

```
                                  ┌─────────────────────────────────────┐
  Camera Stream / Image Upload ──►│   Material Classifier (ResNet18)    │
                                  │      Steel  /  Aluminum  /  Wood    │
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
                                │ Morphological Line Detector  │
                                │ (Canny + HoughLinesP Scratch)│
                                └──────────────┬───────────────┘
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
                                └──────────────┬───────────────┘
                                               ▼
                         ┌───────────────────────────────────────────┐
                         │   FastAPI Web App + WebSocket Server      │
                         │   Single-Scan Mode / Feed Image Interface │
                         └──────────────┬────────────────────┬───────┘
                                        │                    │
                                        ▼                    ▼
                         ┌────────────────────┐    ┌────────────────────┐
                         │  Web UI Dashboard  │    │  ESP32 Serial      │
                         │ (index.html HUD)   │    │  Reject Hardware   │
                         └────────────────────┘    └────────────────────┘
```

---

## 📦 Datasets & Supported Materials

| Material | Source | Images | Classes |
|----------|--------|--------|---------|
| 🔩 **Steel** | NEU-DET + Severstal + GC10-DET | ~5,200 | 15 |
| 🪵 **Wood** | Kodytek Wood Surface Benchmark | ~7,400 | 10 |
| 🤖 **Material Router** | Combined Steel, Aluminum, Wood | ~4,659 | 3 |
| **Total** | | **~15,600** | **25 Defect Classes** |

### Steel Defect Classes (15)
`crazing` · `inclusion` · `patches` · `pitted_surface` · `rolled_in_scale` · `scratches` · `crease` · `crescent_gap` · `water_spot` · `welding_line` · `silk_spot` · `oil_spot` · `punching` · `rolling_pit` · `waist_fold`

### Wood Defect Classes (10)
`live_knot` · `dead_knot` · `knot_with_crack` · `crack` · `resin` · `marrow` · `quartzite` · `missing_knot` · `blue_stain` · `overgrown`

---

## 🧠 Core Features & Technical Innovations

### 1. Single-Scan Architecture (Hardware-Safe Inspection)
To prevent constant buzzer/servo chatter during continuous video feeds, the system provides a **Single-Scan Mode**:
- **Continuous MJPEG Preview**: Low-latency camera preview for sample alignment.
- **On-Demand Server-Side Capture (`GET /scan`)**: User triggers "Scan Now" → backend captures raw BGR frame directly from memory buffer → executes full inference pipeline → fires ESP32 hardware rejection **once**.

### 2. Morphological Line Scratch Detector (Fallback)
Standard bounding-box detectors often miss ultra-faint or elongated scratches. We integrate an automated **morphological line transform fallback**:
- Applies **CLAHE contrast normalization** & **Gaussian smoothing**.
- Runs **Canny Edge Detection** + **Probabilistic Hough Line Transform (`HoughLinesP`)**.
- Groups nearby line segments into scratch clusters and appends annotated bounding boxes with confidence scores.

### 3. Smart Material Router & Monochrome Auto-Detect
- **ResNet18 Neural Classifier**: Classifies surface material textures.
- **Organic Warmth Check (HSV)**: Detects natural timber hues ($Hue: 0-65$, $Sat > 10$, $R+G > 1.3 \times 2B$).
- **Monochrome Industrial Check**: Automatically routes pure grayscale images ($R=G=B$) from NEU-DET / Severstal benchmark datasets to Steel YOLO.

### 4. 11 Morphological Feature Descriptors (XAI Panel)
For each detected defect region, 11 mathematical morphological descriptors are computed:
- *Area, Perimeter, Aspect Ratio, Circularity, Solidity, Convex Hull Perimeter, Compactness, Eccentricity, Edge Density, Skeleton Orientation, Texture Roughness.*

### 5. Physical ESP32 Hardware Reject System
Communicates via high-speed USB Serial (115200 baud):
- **OLED Status Display**: `STATUS:<MATERIAL>,<DEFECT_COUNT>`
- **Red LED + Buzzer**: Instant audible and visual alert on `REJECT`
- **SG90 Servo Sweeper**: 90° physical sorting arm for defective parts
- **Green LED**: Solid indicator for `PASS` parts

---

## 💻 Interactive Web Dashboard (`index.html`)

The system features a **dark-mode cyberpunk web UI** served directly by FastAPI:
- **Quadrant 1 (Input Feed)**: Displays live camera feed or uploaded test image.
- **Quadrant 2 (Detection Result)**: Shows annotated output with bounding boxes, confidence badges, and material tag.
- **Quadrant 3 (Inspection Controls)**: Toggle between **Feed Image** mode and **Live Demo (Single-Scan)**.
- **Quadrant 4 (Results & XAI Panel)**: Real-time verdict (`PASS`/`REJECT`), defect counter, material badge, detected class breakdown, and extracted 11D morphological metrics.

---

## 🚀 Quick Start & Installation

### Prerequisites
```bash
# 1. Create and activate Conda environment
conda create -n rvce python=3.11 -y
conda activate rvce

# 2. Install dependencies
pip install -r requirements.txt
```

### Launch Web Server & Dashboard
```bash
# Run FastAPI server with uvicorn (Port 8000)
python -m uvicorn inference.api_server:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` or open `index.html` directly in your browser.

### Run Standalone OpenCV Live Demo (GUI)
```bash
python inference/realtime_demo.py
```

### Run Command Line Detection on Single Image
```bash
python inference/detect.py --image path/to/sample.jpg
```

---

## 📊 Training Pipeline

```bash
# Master pipeline (Stage 1 to 5)
python train_all.py --stages 1 2 3 4 5

# Individual Stages:
python training/train_material_classifier.py    # Stage 1: ResNet18 Router
python training/train_yolo_baseline.py          # Stages 2-3: Steel & Wood YOLO
python training/train_morphology_fusion.py      # Stages 4-5: Cross-Attention Fusion
```

---

## 📁 Project Structure

```
Strawhats-Defect-Detection-2026/
│
├── index.html                     # Web UI Dashboard (Single-Scan & Feed Image)
├── requirements.txt               # Project dependencies
├── train_all.py                   # Master 5-stage training pipeline
│
├── configs/                       # Dataset YAML configurations
│   ├── dataset_steel.yaml         # Steel: 15 classes, 5,232 images
│   └── dataset_wood_3k.yaml       # Wood: 10 classes, 3k subset
│
├── models/                        # Neural network architectures
│   ├── material_router.py         # ResNet18 + HSV/Monochrome Material Router
│   ├── yolo_backbone.py           # YOLOv10 feature extractor
│   ├── cross_attention.py         # Cross-Attention Fusion module
│   └── classification_head.py     # Morphology-informed classification head
│
├── morphology/                    # Morphological Processing & Fallbacks
│   ├── preprocessing.py           # CLAHE + Gaussian smoothing
│   ├── feature_extractor.py       # 11 Morphological descriptors
│   ├── encoder.py                 # MLP Morphology Encoder (11D → 128D)
│   ├── attention_map.py           # Binary mask → EDT attention map
│   └── dsp_filters.py             # Spatial IIR/FIR filters
│
├── inference/                     # Inference & Server Modules
│   ├── api_server.py              # FastAPI REST & WebSocket server (`/scan`, `/detect`)
│   ├── pipeline.py                # Integrated InferencePipeline with Scratch Fallback
│   ├── realtime_demo.py           # OpenCV desktop HUD demo
│   ├── detect.py                  # Single-image inference CLI
│   └── batch_detect.py            # Folder batch inference
│
├── hardware/                      # Microcontroller Reject Subsystem
│   ├── esp32_reject.ino           # ESP32 C++ Sketch (OLED + Servo + Buzzer + LEDs)
│   ├── serial_bridge.py           # Python PySerial Bridge
│   └── wiring_diagram.md          # Circuit schematic & GPIO pinout guide
│
├── evaluation/                    # Validation & XAI Utilities
│   ├── evaluate.py                # mAP50, Precision, Recall, F1 metrics
│   ├── ablation.py                # Ablation study runner
│   └── explainability.py          # Grad-CAM & SHAP feature importance
│
└── runs/                          # Trained model weights & evaluation curves
    ├── classifier/                # ResNet18 weights
    ├── detect/                    # Steel & Wood YOLO best.pt weights
    └── evaluation/                # Confusion matrices & P-R curves
```

---

## 📈 Training & Evaluation Results

| Model / Subsystem | Metric | Target | Status |
|-------------------|--------|--------|--------|
| **Material Classifier (ResNet18)** | Accuracy | > 95% | ✅ Done (~97.8%) |
| **Steel YOLOv10m (150ep, 5232 imgs)** | mAP50 | > 85% | ✅ Done (87.2%) |
| **Wood YOLOv10m (100ep, 3k subset)** | mAP50 | > 68% | ✅ Done (71.5%) |
| **Steel Morphology Fusion (20ep)** | mAP50 | > 88% | ✅ Done (89.1%) |
| **Wood Morphology Fusion (20ep)** | mAP50 | > 72% | ✅ Done (73.8%) |
| **Morphology Line Fallback** | Recall Increase | +10% | ✅ Integrated (+14.2% on thin scratches) |
| **System Processing Latency** | Latency | < 100ms | ✅ Real-time (~45 ms / frame) |

> *Full evaluation results, confusion matrices, and Precision-Recall curves are available in `runs/evaluation/` and via `python evaluation/evaluate.py`.*

---

## 👥 Team Strawhat-Pirates

**RVCE Hackathon 2026 | Problem Statement 5**

| Member | Focus Area |
|--------|------------|
| **Faizan** | Deep Learning Architecture & Training Pipeline |
| **Pulkit** | Dataset Processing, Augmentation & Pipeline Testing |
| **Chethan** | Morphology Feature Descriptors & Cross-Attention |
| **Rahul** | ESP32 Hardware Integration & Firmware |
| **Akash** | Model Evaluation, Metrics & Web Dashboard |

---

## 📜 License & Acknowledgments
Built for RVCE Hackathon 2026. Special thanks to NEU-DET, Severstal, and Kodytek benchmark dataset creators.
