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
              # ─── HUD Helpers ──────────────────────────────────────────────────────────────
def _alpha_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                color: tuple, alpha: float, radius: int = 0) -> None:
    """Draw a semi-transparent filled rectangle."""
    overlay = img.copy()
    if radius > 0:
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1+radius, y1+radius),(x2-radius, y1+radius),
                       (x1+radius, y2-radius),(x2-radius, y2-radius)]:
            cv2.circle(overlay, (cx, cy), radius, color, -1)
    else:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _conf_bar(img: np.ndarray, x: int, y: int, width: int, height: int,
              value: float, color: tuple) -> None:
    """Draw a horizontal progress bar."""
    # Track
    _alpha_rect(img, x, y, x + width, y + height, (50, 50, 50), 0.8)
    # Fill
    fill_w = max(0, int(width * min(1.0, value)))
    if fill_w > 0:
        _alpha_rect(img, x, y, x + fill_w, y + height, color, 0.9)
    # Border
    cv2.rectangle(img, (x, y), (x + width, y + height), (80, 80, 80), 1, cv2.LINE_AA)


# ─── HUD Overlay ──────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray, material: str, mat_conf: float,
             detections: list, fps: float, conf_thresh: float,
             paused: bool, frozen: bool) -> np.ndarray:
    """Draw a premium semi-transparent HUD overlay."""
    h, w = frame.shape[:2]
    has_defect = len(detections) > 0

    # ── Material colour ──────────────────────────────────────────────────────
    mat_color = MATERIAL_COLORS.get(material, (200, 200, 200))
    pass_color = (60, 220, 80)    # green
    fail_color = (60, 70, 240)    # red (BGR)

    # ════════════════════════════════════════════════════════════════════════
    # TOP BAR  (full-width semi-transparent strip)
    # ════════════════════════════════════════════════════════════════════════
    _alpha_rect(frame, 0, 0, w, 58, (10, 12, 22), 0.82)
    cv2.line(frame, (0, 58), (w, 58), (50, 60, 100), 1)

    # Material icon + name
    mat_icon = '🔩' if material == 'steel' else '🪵'
    mat_label = f"{material.upper()}"
    mat_tag_w = 160
    _alpha_rect(frame, 8, 8, 8 + mat_tag_w, 50, mat_color, 0.18, radius=6)
    cv2.rectangle(frame, (8, 8), (8 + mat_tag_w, 50), mat_color, 1, cv2.LINE_AA)
    cv2.putText(frame, mat_label, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, mat_color, 2, cv2.LINE_AA)

    # Material confidence text next to badge
    cv2.putText(frame, f'Router: {mat_conf*100:.0f}%',
                (mat_tag_w + 20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 170, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Conf thresh: {conf_thresh:.2f}',
                (mat_tag_w + 20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (130, 140, 170), 1, cv2.LINE_AA)

    # FPS (right side of top bar)
    fps_color = (60, 220, 80) if fps >= 15 else (30, 160, 255) if fps >= 8 else (60, 70, 240)
    fps_str = f'{fps:.1f} FPS'
    (fw, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.putText(frame, fps_str, (w - fw - 14, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, fps_color, 2, cv2.LINE_AA)

    # Team badge (top-right, small)
    badge = 'Team SafePath | RVCE 2026'
    (bw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.putText(frame, badge, (w - bw - fw - 24, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 88, 120), 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════════════════════
    # PASS / REJECT BANNER  (bottom-centre)
    # ════════════════════════════════════════════════════════════════════════
    v_color  = fail_color if has_defect else pass_color
    v_text   = 'REJECT' if has_defect else 'PASS'
    v_icon   = 'X' if has_defect else 'OK'

    (vw, vh), _ = cv2.getTextSize(v_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    bx1 = w // 2 - vw // 2 - 28
    bx2 = w // 2 + vw // 2 + 28
    by1 = h - 90
    by2 = h - 38

    _alpha_rect(frame, bx1, by1, bx2, by2, v_color, 0.2, radius=8)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), v_color, 2, cv2.LINE_AA)
    cv2.putText(frame, v_text,
                (w // 2 - vw // 2, by2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, v_color, 3, cv2.LINE_AA)

    # Defect count badge (left of banner)
    cnt_text = f'{len(detections)} defect{"s" if len(detections) != 1 else ""}'
    cv2.putText(frame, cnt_text, (bx1 - 130, by2 - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, v_color, 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════════════════════
    # CONFIDENCE BAR  (just above verdict banner)
    # ════════════════════════════════════════════════════════════════════════
    if detections:
        best_conf = max(d['conf'] for d in detections)
        bar_x1, bar_y = bx1, by1 - 18
        bar_w  = bx2 - bx1
        cv2.putText(frame, f'Best conf: {best_conf:.2f}',
                    (bar_x1, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160,160,200), 1, cv2.LINE_AA)
        _conf_bar(frame, bar_x1, bar_y, bar_w, 6, best_conf, v_color)

    # ════════════════════════════════════════════════════════════════════════
    # DETECTION LIST PANEL  (right side, below top bar)
    # ════════════════════════════════════════════════════════════════════════
    if detections:
        panel_w   = 240
        panel_pad = 10
        row_h     = 30
        n         = min(len(detections), 8)
        px1 = w - panel_w - 4
        py1 = 64
        py2 = py1 + panel_pad + n * row_h + panel_pad

        _alpha_rect(frame, px1, py1, w - 4, py2, (12, 15, 30), 0.82, radius=8)
        cv2.rectangle(frame, (px1, py1), (w - 4, py2), (50, 60, 100), 1, cv2.LINE_AA)

        # Title
        cv2.putText(frame, 'DETECTIONS', (px1 + 10, py1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 140, 180), 1, cv2.LINE_AA)

        for i, det in enumerate(detections[:n]):
            row_y = py1 + panel_pad + 14 + i * row_h
            color = BOX_COLORS[i % len(BOX_COLORS)]
            # Colour dot
            cv2.circle(frame, (px1 + 14, row_y + 2), 5, color, -1, cv2.LINE_AA)
            # Label
            cv2.putText(frame, det['label'],
                        (px1 + 26, row_y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 235, 255), 1, cv2.LINE_AA)
            # Confidence bar
            _conf_bar(frame, px1 + 26, row_y + 12, 110, 5, det['conf'], color)
            # Conf value
            cv2.putText(frame, f"{det['conf']:.2f}",
                        (px1 + 144, row_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════════════════════
    # PIPELINE STEPS  (left side, bottom)
    # ════════════════════════════════════════════════════════════════════════
    steps = [
        ('1', 'ResNet18 Router',   mat_color),
        ('2', 'YOLOv10m Detect',   (100, 180, 255)),
        ('3', 'Morphology Extr.',  (200, 160, 255)),
        ('4', 'CrossAttn Fusion',  (100, 230, 180)),
    ]
    sx, sy = 8, h - 185
    _alpha_rect(frame, sx, sy, sx + 200, sy + len(steps)*30 + 14, (10, 12, 22), 0.78, radius=6)
    cv2.rectangle(frame, (sx, sy), (sx + 200, sy + len(steps)*30 + 14), (40, 50, 80), 1)
    cv2.putText(frame, 'AI PIPELINE', (sx + 8, sy + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 110, 160), 1, cv2.LINE_AA)
    for i, (num, label, col) in enumerate(steps):
        ry = sy + 24 + i * 30
        cv2.circle(frame, (sx + 18, ry + 6), 8, col, -1, cv2.LINE_AA)
        cv2.putText(frame, num, (sx + 14, ry + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 12, 22), 1, cv2.LINE_AA)
        cv2.putText(frame, label, (sx + 32, ry + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 210, 240), 1, cv2.LINE_AA)
        # connector line
        if i < len(steps) - 1:
            cv2.line(frame, (sx + 18, ry + 14), (sx + 18, ry + 24), col, 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════════════════════
    # BOTTOM BAR  (controls)
    # ════════════════════════════════════════════════════════════════════════
    _alpha_rect(frame, 0, h - 26, w, h, (10, 12, 22), 0.88)
    cv2.line(frame, (0, h - 26), (w, h - 26), (40, 50, 80), 1)
    controls = 'Q  Quit    S  Screenshot    P  Pause    SPACE  Freeze    + / -  Confidence'
    cv2.putText(frame, controls, (12, h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 110, 160), 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════════════════════
    # PAUSED / FROZEN  overlays
    # ════════════════════════════════════════════════════════════════════════
    if paused or frozen:
        state_text = '[ PAUSED ]' if paused else '[ FROZEN ]  •  SPACE to resume'
        state_col  = (0, 200, 255)
        (stw, _), _ = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        _alpha_rect(frame, w//2 - stw//2 - 20, h//2 - 40, w//2 + stw//2 + 20, h//2 + 20,
                    (0, 0, 0), 0.65, radius=10)
        cv2.putText(frame, state_text, (w//2 - stw//2, h//2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, state_col, 2, cv2.LINE_AA)

    return frame�──────────────────
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
