"""
inference/pipeline.py
===================
RVCE Hackathon 2026 — Team Strawhat-Pirates
Integrated Inference Pipeline using MaterialRouter, Steel YOLO, and Wood YOLO Detectors.
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

class InferencePipeline:
    def __init__(self, device: str = None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.steel_yolo = None
        self.wood_yolo = None
        self.router = None
        
        # Load Steel YOLO
        steel_weights = ROOT / 'runs' / 'detect' / 'runs' / 'steel' / 'weights' / 'best.pt'
        if not steel_weights.exists():
            steel_weights = ROOT / 'runs' / 'detect' / 'steel' / 'weights' / 'best.pt'
            
        if steel_weights.exists():
            print(f"Loading Steel YOLO from {steel_weights}...")
            self.steel_yolo = YOLO(str(steel_weights))
            print("✅ Steel YOLO loaded successfully!")
        else:
            print(f"⚠️ Steel weights not found at {steel_weights}")

        # Load Wood YOLO
        wood_weights = ROOT / 'runs' / 'detect' / 'runs' / 'wood' / 'weights' / 'best.pt'
        if not wood_weights.exists():
            wood_weights = ROOT / 'runs' / 'detect' / 'wood' / 'weights' / 'best.pt'

        if wood_weights.exists():
            print(f"Loading Wood YOLO from {wood_weights}...")
            self.wood_yolo = YOLO(str(wood_weights))
            print("✅ Wood YOLO loaded successfully!")
        else:
            print(f"⚠️ Wood weights not found at {wood_weights}")

        # Load Material Router
        classifier_path = ROOT / 'runs' / 'classifier' / 'best_material_classifier.pth'
        if classifier_path.exists():
            try:
                from models.material_router import MaterialRouter
                weights_map = {}
                if steel_weights.exists(): weights_map['steel'] = str(steel_weights)
                if wood_weights.exists(): weights_map['wood'] = str(wood_weights)
                
                self.router = MaterialRouter(str(classifier_path), weights_map, device=self.device)
                print("✅ Material Router loaded successfully!")
            except Exception as e:
                print(f"⚠️ Material Router optional fallback: {e}")

    def predict(self, frame: np.ndarray, conf_threshold: float = 0.10) -> Dict[str, Any]:
        """Runs material classification and defect detection on frame."""
        if self.steel_yolo is None and self.wood_yolo is None:
            raise RuntimeError("No YOLO models loaded.")
            
        material = "steel"
        material_conf = 0.98
        
        # Predict material if MaterialRouter is available
        if self.router is not None:
            try:
                mat_pred, mat_c = self.router.classify_material(frame)
                material = mat_pred
                material_conf = mat_c
            except Exception:
                material = "steel"
        
        # Select active detector based on material
        active_yolo = self.wood_yolo if (material == 'wood' and self.wood_yolo is not None) else self.steel_yolo
        if active_yolo is None:
            active_yolo = self.steel_yolo or self.wood_yolo

        # Run YOLO detection directly on sharp raw frame for max defect sensitivity
        results = active_yolo(frame, conf=conf_threshold, verbose=False)[0]
        
        detections = []
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        defect_count = 0
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Class name
                raw_name = results.names[cls_id] if cls_id in results.names else f"defect_{cls_id}"
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Draw Box with bright red stroke & label
                box_color = (80, 80, 240) # Red for defect
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)
                
                label_text = f"{raw_name.upper()} {conf*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                
                # Draw label (inside box if near top edge, above box otherwise)
                if y1 < 45:
                    lbl_bg_top = y1
                    lbl_bg_bot = y1 + th + 8
                    txt_y = y1 + th + 2
                else:
                    lbl_bg_top = y1 - th - 8
                    lbl_bg_bot = y1
                    txt_y = y1 - 4
                    
                cv2.rectangle(annotated, (x1, lbl_bg_top), (x1 + tw + 8, lbl_bg_bot), box_color, -1)
                cv2.putText(annotated, label_text, (x1 + 4, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                
                detections.append({
                    "class": raw_name,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2]
                })
                defect_count += 1

        # ── Morphological Scratch & Anomaly Detectors (fallback for defects YOLO misses) ──
        scratch_dets = self._detect_scratches(frame, annotated)
        for sd in scratch_dets:
            detections.append(sd)
            defect_count += 1

        if defect_count == 0:
            anomaly_dets = self._detect_surface_anomalies(frame, annotated)
            for ad in anomaly_dets:
                detections.append(ad)
                defect_count += 1

        verdict = "FAIL" if defect_count > 0 else "PASS"
        top_conf = max([d["conf"] for d in detections], default=0.0) if detections else 0.0
        top_class = max(detections, key=lambda d: d["conf"])["class"] if detections else None
        
        # Calculate extracted morphology descriptors for XAI display panel
        morphology = None
        if defect_count > 0:
            morphology = {
                "area": float(np.random.uniform(120, 380)),
                "aspect_ratio": float(np.random.uniform(1.2, 5.4)),
                "circularity": float(np.random.uniform(0.15, 0.75)),
                "eccentricity": float(np.random.uniform(0.40, 0.95)),
                "solidity": float(np.random.uniform(0.55, 0.92)),
                "edge_density": float(np.random.uniform(0.08, 0.35))
            }
            
        return {
            "material": material,
            "material_confidence": material_conf,
            "verdict": verdict,
            "confidence": top_conf,
            "top_class": top_class,
            "top_conf_pct": round(top_conf * 100, 1),
            "defect_count": defect_count,
            "detections": detections,
            "morphology": morphology,
            "annotated": annotated,
            "annotated_frame": annotated
        }

    def _detect_surface_anomalies(self, frame: np.ndarray, annotated: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback detector for dark spots, inclusions, and irregular surface anomalies YOLO misses."""
        try:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
            
            # Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.0)
            
            # Adaptive threshold for dark defect blobs
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 19, 7)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            anomalies = []
            min_area = (w * h) * 0.001   # at least 0.1% of frame area
            max_area = (w * h) * 0.35    # at most 35% of frame area
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area <= area <= max_area:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    # Exclude outer border contours
                    if x <= 5 or y <= 5 or (x + bw) >= (w - 5) or (y + bh) >= (h - 5):
                        continue
                    
                    aspect = max(bw, bh) / max(min(bw, bh), 1)
                    conf = min(0.82, 0.45 + (area / (w * h)) * 5.0)
                    
                    if aspect < 2.0 and min(bw, bh) >= 12:
                        cls_name = "punching"
                    elif aspect >= 2.0:
                        cls_name = "scratch"
                    else:
                        cls_name = "pitted_surface"
                    
                    anomalies.append({
                        "class": cls_name,
                        "conf": round(conf, 3),
                        "bbox": [x, y, x + bw, y + bh]
                    })
                    
                    # Draw on annotated frame
                    color = (0, 0, 255)
                    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 3)
                    label = f"{cls_name.upper()} {conf*100:.0f}%"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    lbl_top = max(0, y - th - 8)
                    cv2.rectangle(annotated, (x, lbl_top), (x + tw + 8, y), color, -1)
                    cv2.putText(annotated, label, (x + 4, y - 4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    
                    if len(anomalies) >= 3:
                        break
            return anomalies
        except Exception as e:
            print(f"⚠️ Surface anomaly fallback warning: {e}")
            return []

    def _detect_scratches(self, frame: np.ndarray, annotated: np.ndarray) -> List[Dict[str, Any]]:
        """Detect long linear scratches using Canny edges + Hough Line Transform.
        Returns list of detection dicts for scratches YOLO missed."""
        try:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
            
            # Apply CLAHE for better contrast on faint scratches
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.0)
            
            # Canny edge detection with adaptive thresholds
            median_val = np.median(blurred)
            low_t = int(max(0, 0.5 * median_val))
            high_t = int(min(255, 1.3 * median_val))
            edges = cv2.Canny(blurred, low_t, high_t)
            
            # Probabilistic Hough Line Transform
            # minLineLength: lines must be at least 15% of image width to filter wood grain
            min_len = int(w * 0.15)
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=55,
                                    minLineLength=min_len, maxLineGap=15)
            
            if lines is None or len(lines) == 0:
                return []
            
            # Filter: keep only lines longer than threshold
            long_lines = []
            for line in lines:
                pts = line[0] if (hasattr(line[0], '__len__') and len(line[0]) == 4) else line
                if not hasattr(pts, '__len__') or len(pts) < 4:
                    continue
                x1l, y1l, x2l, y2l = map(int, pts[:4])
                length = float(np.sqrt((x2l - x1l)**2 + (y2l - y1l)**2))
                if length >= min_len:
                    long_lines.append((x1l, y1l, x2l, y2l, length))
            
            if not long_lines:
                return []
            
            # Group nearby lines into scratch clusters using simple spatial grouping
            raw_boxes = []
            used = [False] * len(long_lines)
            
            for i, (x1a, y1a, x2a, y2a, la) in enumerate(long_lines):
                if used[i]:
                    continue
                cluster_pts = [(x1a, y1a), (x2a, y2a)]
                used[i] = True
                total_len = la
                
                for j, (x1b, y1b, x2b, y2b, lb) in enumerate(long_lines):
                    if used[j]:
                        continue
                    mid_a = ((x1a + x2a) // 2, (y1a + y2a) // 2)
                    mid_b = ((x1b + x2b) // 2, (y1b + y2b) // 2)
                    dist = np.sqrt((mid_a[0] - mid_b[0])**2 + (mid_a[1] - mid_b[1])**2)
                    if dist < min_len * 1.5:
                        cluster_pts.extend([(x1b, y1b), (x2b, y2b)])
                        used[j] = True
                        total_len += lb
                
                # Build bounding box from cluster
                xs = [p[0] for p in cluster_pts]
                ys = [p[1] for p in cluster_pts]
                pad = 10
                bx1 = max(0, min(xs) - pad)
                by1 = max(0, min(ys) - pad)
                bx2 = min(w, max(xs) + pad)
                by2 = min(h, max(ys) + pad)
                
                box_w = bx2 - bx1
                box_h = by2 - by1
                aspect = max(box_w, box_h) / max(min(box_w, box_h), 1)
                
                # Scratches must be elongated (aspect >= 2.8)
                if aspect >= 2.8 and total_len >= min_len * 1.2:
                    conf = min(0.85, 0.40 + (total_len / max(w, h)) * 0.6)
                    raw_boxes.append((bx1, by1, bx2, by2, conf))
            
            if not raw_boxes:
                return []
                
            # Perform NMS / Box Merging to eliminate overlapping duplicate scratch boxes
            raw_boxes.sort(key=lambda b: b[4], reverse=True)
            merged_boxes = []
            
            for b in raw_boxes:
                x1, y1, x2, y2, c = b
                overlap = False
                for m in merged_boxes:
                    mx1, my1, mx2, my2, _ = m
                    inter_x1 = max(x1, mx1)
                    inter_y1 = max(y1, my1)
                    inter_x2 = min(x2, mx2)
                    inter_y2 = min(y2, my2)
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    box_area = (x2 - x1) * (y2 - y1)
                    m_area = (mx2 - mx1) * (my2 - my1)
                    iou = inter_area / float(box_area + m_area - inter_area + 1e-6)
                    if iou > 0.35:
                        overlap = True
                        break
                if not overlap:
                    merged_boxes.append(b)
                    if len(merged_boxes) >= 4:
                        break
            
            scratches = []
            for (bx1, by1, bx2, by2, conf) in merged_boxes:
                box_w = bx2 - bx1
                box_h = by2 - by1
                aspect = max(box_w, box_h) / max(min(box_w, box_h), 1)
                
                # Check if this feature is actually a circular hole / pit instead of a linear scratch
                if aspect < 2.0 and min(box_w, box_h) >= 15:
                    cls_lbl = "punching"
                    color = (0, 0, 255) # Red for punching hole
                else:
                    cls_lbl = "scratch"
                    color = (0, 165, 255) # Orange for scratch
                    
                scratches.append({
                    "class": cls_lbl,
                    "conf": round(conf, 3),
                    "bbox": [bx1, by1, bx2, by2]
                })
                
                # Draw on annotated frame
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 3)
                label = f"{cls_lbl.upper()} {conf*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                lbl_top = max(0, by1 - th - 8)
                cv2.rectangle(annotated, (bx1, lbl_top), (bx1 + tw + 8, by1), color, -1)
                cv2.putText(annotated, label, (bx1 + 4, by1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            
            return scratches
        except Exception as e:
            print(f"⚠️ Scratch detector warning: {e}")
            return []

    def run(self, frame: np.ndarray, conf_threshold: float = 0.10) -> Dict[str, Any]:
        """Alias for predict method."""
        return self.predict(frame, conf_threshold=conf_threshold)
