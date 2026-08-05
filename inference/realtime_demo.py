"""
realtime_demo.py
================
RVCE Hackathon 2026 — Team SafePath
Multi-Material Crack & Defect Inspection System
Real-Time Live Demo Script

Usage:
    python inference/realtime_demo.py                         # Webcam (default)
    python inference/realtime_demo.py --source 0              # Camera index 0
    python inference/realtime_demo.py --source image.jpg      # Single image
    python inference/realtime_demo.py --conf 0.3              # Higher confidence threshold

Controls (during demo):
    Q        — Quit
    S        — Save screenshot
    P        — Pause/Resume
    SPACE    — Freeze frame (great for showing judges)
    +/-      — Increase/decrease confidence threshold
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# ─── Add project root to path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
# Classifier was trained on 3 classes — aluminum remapped to wood at runtime
MATERIALS        = ['aluminum', 'steel', 'wood']   # keep for classifier index mapping
ACTIVE_MATERIALS = ['steel', 'wood']               # only these have trained YOLO models
MATERIAL_COLORS  = {
    'steel': (0, 165, 255),   # Orange
    'wood':  (0, 200,  80),   # Green
}

# Class label colours by severity (for bounding boxes)
BOX_COLORS = [
    (0,   0,   255),   # Red    — high severity
    (0,  140,  255),   # Orange
    (0,  255,  255),   # Yellow
    (0,  200,   80),   # Green  — low severity
]

STEEL_CLASSES = [
    'crazing','inclusion','patches','pitted_surface',
    'rolled-in_scale','scratches','crease','crescent_gap',
    'oil_spot','punching','rolled_pit','silk_spot',
    'water_spot','waist_fold','welding_line'
]
ALUMINUM_CLASSES = [
    'inclusion','oil_spot','patches','pitted_surface',
    'scratches','silk_spot','water_spot','welding_line',
    'crease','waist_fold'
]
WOOD_CLASSES = [
    'crack','knot','knot_with_crack','knot_missing',
    'resin','blue_stain','quartzite','marrow',
    'overgrown','dead_knot'
]

MATERIAL_CLASSES = {
    'steel': STEEL_CLASSES,
    'wood':  WOOD_CLASSES,
}

# ─── Weight Paths ─────────────────────────────────────────────────────────────
def find_weight(material: str, kind: str = 'yolo') -> str | None:
    """Search common locations for trained weights."""
    if kind == 'classifier':
        candidates = [
            ROOT / 'runs' / 'material_classifier' / 'best.pt',
            ROOT / 'runs' / 'classifier' / 'best.pt',
            ROOT / 'models'/ 'material_classifier.pt',
        ]
    else:
        candidates = [
            ROOT / 'runs' / material / 'weights' / 'best.pt',
            ROOT / 'runs' / 'detect' / 'runs' / material / 'weights' / 'best.pt',
            ROOT / 'runs' / 'detect' / material / 'weights' / 'best.pt',
            ROOT / 'runs' / f'{material}' / 'best.pt',
        ]
        # Also try rglob search
        for p in (ROOT / 'runs').rglob(f'*{material}*/weights/best.pt'):
            candidates.append(p)

    for p in candidates:
        if Path(p).exists():
            logger.info(f"Found {kind} weights for {material}: {p}")
            return str(p)
    return None


# ─── Material Classifier ──────────────────────────────────────────────────────
class MaterialClassifier:
    def __init__(self, weights_path: str | None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if weights_path and Path(weights_path).exists():
            try:
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, 3)
                ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
                # Handle different checkpoint formats
                if isinstance(ckpt, dict):
                    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
                else:
                    state = ckpt
                model.load_state_dict(state, strict=False)
                model.eval()
                self.model = model.to(self.device)
                logger.info("Material classifier loaded ✓")
            except Exception as e:
                logger.warning(f"Could not load classifier: {e} — using rule-based fallback")
        else:
            logger.warning("No classifier weights found — using filename/fallback mode")

    def predict(self, frame_bgr: np.ndarray) -> tuple[str, float]:
        """Returns (material_name, confidence). Aluminum remapped to wood."""
        if self.model is None:
            return 'steel', 0.0   # Safe default
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out   = self.model(tensor)
                probs = torch.softmax(out, dim=1)[0]
                idx   = int(probs.argmax())
            material = MATERIALS[idx]
            # Aluminum dropped from pipeline — remap to wood (closest surface type)
            if material == 'aluminum':
                material = 'wood'
            return material, float(probs[idx])
        except Exception as e:
            logger.error(f"Classifier error: {e}")
            return 'steel', 0.0


# ─── YOLO Detector ────────────────────────────────────────────────────────────
class DefectDetector:
    def __init__(self):
        self.models: dict[str, YOLO | None] = {}
        for mat in ACTIVE_MATERIALS:   # steel + wood only
            w = find_weight(mat, 'yolo')
            if w:
                try:
                    self.models[mat] = YOLO(w)
                    logger.info(f"YOLO {mat} loaded ✓  ({w})")
                except Exception as e:
                    logger.warning(f"Could not load YOLO for {mat}: {e}")
                    self.models[mat] = None
            else:
                logger.warning(f"No YOLO weights found for {mat}")
                self.models[mat] = None

    def detect(self, frame_bgr: np.ndarray, material: str, conf: float = 0.25):
        """Run YOLO on frame, return (annotated_frame, detections_list)."""
        model = self.models.get(material)
        if model is None:
            return frame_bgr, []

        try:
            results = model.predict(frame_bgr, conf=conf, verbose=False, device=0 if torch.cuda.is_available() else 'cpu')
            detections = []
            out = frame_bgr.copy()
            class_names = MATERIAL_CLASSES.get(material, [])

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf_score      = float(box.conf[0])
                    cls_id          = int(box.cls[0])
                    label           = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
                    color           = BOX_COLORS[cls_id % len(BOX_COLORS)]

                    # Draw box
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

                    # Label background
                    text  = f'{label}  {conf_score:.2f}'
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(out, text, (x1 + 2, y1 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                    detections.append({'label': label, 'conf': conf_score, 'box': (x1, y1, x2, y2)})

            return out, detections
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return frame_bgr, []


# ─── HUD Overlay ──────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray, material: str, mat_conf: float,
             detections: list, fps: float, conf_thresh: float,
             paused: bool, frozen: bool) -> np.ndarray:
    """Draw a clean semi-transparent HUD overlay."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Top bar ─────────────────────────────────────────────────────────────
    cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    mat_color = MATERIAL_COLORS.get(material, (200, 200, 200))
    cv2.putText(frame, f'Material: {material.upper()}  ({mat_conf*100:.0f}%)',
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mat_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f'FPS: {fps:4.1f}   Conf: {conf_thresh:.2f}   Defects: {len(detections)}',
                (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Right side: detection list ───────────────────────────────────────────
    if detections:
        panel_w = 230
        cv2.rectangle(frame, (w - panel_w, 70), (w, 70 + len(detections) * 28 + 10),
                      (20, 20, 20), -1)
        for i, d in enumerate(detections[:10]):
            color = BOX_COLORS[i % len(BOX_COLORS)]
            cv2.putText(frame, f"  {d['label']}  {d['conf']:.2f}",
                        (w - panel_w + 4, 94 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # ── Bottom bar: controls ─────────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 28), (w, h), (20, 20, 20), -1)
    controls = 'Q=Quit  S=Screenshot  P=Pause  SPACE=Freeze  +/-=Conf'
    cv2.putText(frame, controls, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Status badge ────────────────────────────────────────────────────────
    if paused:
        cv2.putText(frame, '[ PAUSED ]', (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 255), 3, cv2.LINE_AA)
    if frozen:
        cv2.putText(frame, '[ FROZEN ]  SPACE to resume', (w // 2 - 180, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 220, 255), 2, cv2.LINE_AA)

    # ── Team badge (top-right) ───────────────────────────────────────────────
    badge = 'Team SafePath | RVCE 2026'
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, badge, (w - bw - 8, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

    return frame


# ─── Main Pipeline ────────────────────────────────────────────────────────────
def run_demo(args):
    logger.info("=" * 60)
    logger.info("  RVCE Hackathon 2026 — SafePath Live Demo")
    logger.info("=" * 60)

    # Load models
    clf_weights = find_weight('', 'classifier')
    classifier  = MaterialClassifier(clf_weights)
    detector    = DefectDetector()

    # Check if any YOLO model loaded
    loaded = [m for m, mdl in detector.models.items() if mdl is not None]
    if not loaded:
        logger.error("No YOLO weights found in runs/. Make sure training has completed.")
        logger.error("Run: python train_all.py --stages 2 3 4")
        sys.exit(1)
    logger.info(f"Loaded YOLO models for: {loaded}")

    # Open source
    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {args.source}")
        sys.exit(1)

    # Set camera resolution
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    conf_thresh   = args.conf
    paused        = False
    frozen        = False
    frozen_frame  = None
    frame_count   = 0
    screenshot_n  = 0
    prev_time     = time.time()
    fps           = 0.0

    # Cache material classification every N frames (avoid re-running every frame)
    current_material  = loaded[0]
    current_mat_conf  = 0.0
    clf_interval      = 15   # re-classify every 15 frames

    logger.info("Demo running!  Controls:  Q=Quit  S=Screenshot  SPACE=Freeze  P=Pause  +/-=Conf")

    try:
        while True:
            if not paused and not frozen:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of stream.")
                    break

                frame_count += 1

                # ── Material classification (periodic) ───────────────────
                if frame_count % clf_interval == 1:
                    mat, mc = classifier.predict(frame)
                    if mat in loaded:           # only use if we have YOLO for it
                        current_material = mat
                        current_mat_conf = mc

                # ── YOLO defect detection ────────────────────────────────
                annotated, detections = detector.detect(frame, current_material, conf_thresh)

                # ── FPS ──────────────────────────────────────────────────
                curr_time = time.time()
                fps       = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
                prev_time = curr_time

                display = draw_hud(annotated, current_material, current_mat_conf,
                                   detections, fps, conf_thresh, paused, frozen)

            elif frozen and frozen_frame is not None:
                display = draw_hud(frozen_frame.copy(), current_material, current_mat_conf,
                                   [], fps, conf_thresh, paused, True)
            else:
                # Paused — keep showing last frame
                time.sleep(0.03)
                if 'display' not in dir():
                    continue

            cv2.imshow('SafePath — Multi-Material Defect Inspector', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:   # Q or ESC
                break
            elif key == ord('s'):
                screenshot_n += 1
                fname = f'demo_screenshot_{screenshot_n:03d}.jpg'
                cv2.imwrite(fname, display)
                logger.info(f"Screenshot saved: {fname}")
            elif key == ord('p'):
                paused = not paused
                logger.info("Paused" if paused else "Resumed")
            elif key == ord(' '):              # Spacebar = freeze
                if not frozen:
                    frozen       = True
                    frozen_frame = display.copy()
                    logger.info("Frame frozen. Press SPACE to resume.")
                else:
                    frozen       = False
                    frozen_frame = None
            elif key == ord('+') or key == ord('='):
                conf_thresh = min(0.95, conf_thresh + 0.05)
                logger.info(f"Confidence threshold: {conf_thresh:.2f}")
            elif key == ord('-'):
                conf_thresh = max(0.05, conf_thresh - 0.05)
                logger.info(f"Confidence threshold: {conf_thresh:.2f}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Demo closed.")


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SafePath Real-Time Defect Demo — RVCE Hackathon 2026',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference/realtime_demo.py                  # Default webcam
  python inference/realtime_demo.py --source 1       # Second camera
  python inference/realtime_demo.py --conf 0.4       # Higher confidence
  python inference/realtime_demo.py --source img.jpg # Single image (loops)
        """
    )
    parser.add_argument('--source', type=str, default='0',
                        help='Camera index (0,1,...) or image/video path')
    parser.add_argument('--conf',   type=float, default=0.25,
                        help='Detection confidence threshold (default: 0.25)')
    args = parser.parse_args()
    run_demo(args)
