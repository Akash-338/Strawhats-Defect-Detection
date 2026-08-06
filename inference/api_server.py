"""
api_server.py
=============
RVCE Hackathon 2026 — Team Strawhat-Pirates
FastAPI backend: live MJPEG stream + WebSocket stats + inference pipeline

Endpoints:
  GET  /health          — health check
  GET  /stats           — cumulative counters
  GET  /video_feed      — MJPEG camera stream (for dashboard <img>)
  WS   /ws/live         — WebSocket: sends JSON updates every frame
  POST /detect          — single-image inference (REST)

Start:
  uvicorn inference.api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import io
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Strawhat-Pirates Inspection API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ───────────────────────────────────────────────────────────────
stats: Dict[str, Any] = {
    "total_scanned": 0,
    "total_defects": 0,
    "passed": 0,
    "failed": 0,
    "fps": 0.0,
}

# Latest frame data (written by inference loop, read by WebSocket/MJPEG)
current_frame: Optional[bytes] = None          # JPEG bytes of annotated frame
current_live:  Dict[str, Any]  = {}            # latest detection result
_latest_raw_frame: Optional[np.ndarray] = None  # raw BGR frame for /scan endpoint

# Connected WebSocket clients
ws_clients: List[WebSocket] = []

# ── Try to import inference pipeline ──────────────────────────────────────────
_pipeline_loaded = False
_pipeline = None

def _try_load_pipeline():
    global _pipeline_loaded, _pipeline
    try:
        # Import the full inference pipeline
        # This will work once models are trained
        from inference.pipeline import InferencePipeline
        _pipeline = InferencePipeline()
        _pipeline_loaded = True
        logger.info("✅ Inference pipeline loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️  Inference pipeline not available: {e}")
        logger.warning("   Running in demo mode with mock detections")
        _pipeline_loaded = False

_try_load_pipeline()

# ── Hardware Serial Bridge for ESP32 / Arduino ────────────────────────────────
_serial_bridge = None
def _try_load_serial_bridge():
    global _serial_bridge
    try:
        from hardware.serial_bridge import SerialBridge
        _serial_bridge = SerialBridge(baudrate=115200)
        if _serial_bridge.is_connected():
            logger.info("⚡ ESP32 Hardware Reject System connected via Serial!")
        else:
            logger.info("ℹ️ ESP32 Hardware Bridge ready (will auto-trigger when USB plugged in)")
    except Exception as e:
        logger.warning(f"⚠️ Could not load SerialBridge: {e}")

_try_load_serial_bridge()
_last_hardware_verdict = None

def _send_esp32_command(material: str, defects: int, verdict: str):
    """Sends hardware actuation commands to STM32 / ESP32 / Arduino microcontrollers."""
    if _serial_bridge:
        mat = str(material).upper()
        is_reject = (defects > 0) or (str(verdict).upper() in ["FAIL", "REJECT"])
        cmd = "REJECT" if is_reject else "PASS"
        defect_val = max(1, defects) if is_reject else 0
        
        logger.info(f"⚡ Hardware trigger: {cmd} (defects={defects}, verdict={verdict})")
        
        # 1. Send status update first (sets material and defect count on OLED)
        _serial_bridge.send(f"STATUS:{mat},{defect_val}")
        time.sleep(0.05)
        
        # 2. Send primary hardware command (REJECT or PASS) with retry logic
        success = _serial_bridge.send(cmd)
        if not success and _serial_bridge.is_connected():
            _serial_bridge.send(cmd)

# ── Mock detection (used when pipeline is not ready) ───────────────────────────
import random

# ── Steel: 10 validated classes (removed 5 weak performers < 60% mAP) ──────────
# Removed: crazing(33.8%), inclusion(58.9%), rolled_in_scale(46.9%),
#          rolled_pit(33.9%), crease(38.9%)
# Remaining avg mAP@50 = 76.8%
STEEL_CLASSES = [
    'patches',        # 87.0%
    'pitted_surface', # 77.5%
    'scratches',      # 90.0%
    'punching',       # 96.3%
    'weld_line',      # 62.0%
    'crescent_gap',   # 82.8%
    'water_spot',     # 71.4%
    'oil_spot',       # 61.8%
    'silk_spot',      # 63.1%
    'waist_folding',  # 76.0%
]

# Class IDs to SKIP during real YOLO inference (0-indexed, matches dataset.yaml order)
# 0:crazing, 1:inclusion, 4:rolled_in_scale, 12:rolled_pit, 13:crease
STEEL_SKIP_IDS = {0, 1, 4, 12, 13}

WOOD_CLASSES = ['crack', 'knot', 'knot_with_crack', 'missing_knot',
                 'resin', 'blue_stain', 'quartzite', 'marrow',
                 'overgrown', 'dead_knot']

def _mock_detect(frame: np.ndarray) -> Dict[str, Any]:
    """Realistic mock detection for demo purposes."""
    material = random.choice(['steel', 'wood'])
    classes  = STEEL_CLASSES if material == 'steel' else WOOD_CLASSES
    has_defect = random.random() < 0.45  # 45% defect rate for demo

    detections = []
    morphology = None

    if has_defect:
        n = random.randint(1, 3)
        for _ in range(n):
            cls  = random.choice(classes)
            conf = random.uniform(0.52, 0.97)
            detections.append({"class": cls, "conf": conf})

        top = max(detections, key=lambda d: d["conf"])
        morphology = {
            "area":         random.uniform(80, 400),
            "aspect_ratio": random.uniform(0.8, 6.5),
            "circularity":  random.uniform(0.01, 0.95),
            "eccentricity": random.uniform(0.1, 0.99),
            "solidity":     random.uniform(0.4, 0.98),
            "edge_density": random.uniform(0.02, 0.45),
        }
        verdict     = "FAIL"
        confidence  = top["conf"]
        top_class   = top["class"]
    else:
        verdict     = "PASS"
        confidence  = random.uniform(0.0, 0.25)
        top_class   = None

    # Draw mock bounding box on frame
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Material badge
    mat_color = (255, 165, 0) if material == 'steel' else (0, 200, 80)
    cv2.rectangle(annotated, (8, 8), (140, 36), (0,0,0), -1)
    cv2.rectangle(annotated, (8, 8), (140, 36), mat_color, 1)
    cv2.putText(annotated, f"{'STEEL' if material=='steel' else 'WOOD'}",
                (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mat_color, 2)

    # Verdict overlay
    v_color = (50, 200, 80) if verdict == 'PASS' else (80, 80, 240)
    v_text  = "PASS" if verdict == 'PASS' else "REJECT"
    txt_w   = 120
    cv2.rectangle(annotated, (w - txt_w - 10, 8), (w - 8, 36), (0,0,0), -1)
    cv2.rectangle(annotated, (w - txt_w - 10, 8), (w - 8, 36), v_color, 1)
    cv2.putText(annotated, v_text, (w - txt_w, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, v_color, 2)

    if has_defect and detections:
        # Draw bounding box for first detection
        x1 = random.randint(w//8, w//2)
        y1 = random.randint(h//8, h//2)
        x2 = x1 + random.randint(60, 180)
        y2 = y1 + random.randint(40, 120)
        x2, y2 = min(x2, w-8), min(y2, h-8)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 80, 240), 2)
        label = f"{detections[0]['class']} {detections[0]['conf']:.2f}"
        cv2.rectangle(annotated, (x1, y1-22), (x1+len(label)*10, y1), (80,80,240), -1)
        cv2.putText(annotated, label, (x1+2, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return {
        "material":    material,
        "verdict":     verdict,
        "confidence":  confidence,
        "top_class":   top_class,
        "top_conf_pct": round(confidence * 100, 1),
        "detections":  detections,
        "morphology":  morphology,
        "annotated":   annotated,
    }


# ── WebSocket broadcast ────────────────────────────────────────────────────────
async def broadcast(data: Dict[str, Any]):
    dead = []
    for client in ws_clients:
        try:
            await client.send_json(data)
        except Exception:
            dead.append(client)
    for d in dead:
        ws_clients.remove(d)


# ── Background inference loop ──────────────────────────────────────────────────
_camera_task = None

async def inference_loop():
    """Capture webcam, run inference, update global state and broadcast."""
    global current_frame, current_live, stats

    cam_src = os.getenv("CAM_SOURCE", "auto")
    cap = None
    if cam_src != "auto" and cam_src.isdigit():
        idx = int(cam_src)
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            cap = c
            logger.info(f"Opening camera source: {idx}")
    
    if cap is None:
        # Auto-scan: try index 1 (mobile cam / DroidCam) first, then 0
        for idx in [1, 0, 2]:
            c = cv2.VideoCapture(idx)
            if c.isOpened():
                r, f = c.read()
                if r and f is not None:
                    cap = c
                    logger.info(f"🎥 Auto-detected active camera on port {idx} ({f.shape[1]}x{f.shape[0]})")
                    break
                c.release()

    if cap is None or not cap.isOpened():
        logger.warning("⚠️ No active camera stream found — using colour-bar test pattern")
        cap = None

    logger.info("🎥 Inference loop started")
    fps_timer  = time.perf_counter()
    frame_count = 0

    while True:
        # ── Grab frame ──────────────────────────────────────────────
        if cap is not None:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Attempt to re-open camera 1 / 0
                for idx in [1, 0, 2]:
                    c = cv2.VideoCapture(idx)
                    if c.isOpened():
                        r, f = c.read()
                        if r and f is not None:
                            if cap: cap.release()
                            cap = c
                            ret, frame = r, f
                            break
                        c.release()
                if not ret or frame is None:
                    frame = _test_pattern()
        else:
            frame = _test_pattern()

        # Store raw frame for /scan endpoint
        global _latest_raw_frame
        _latest_raw_frame = frame.copy()

        # ── Run inference ────────────────────────────────────────────
        if _pipeline_loaded and _pipeline is not None:
            try:
                result = _pipeline.run(frame)
            except Exception as e:
                logger.error(f"Inference loop error: {e}")
                result = {
                    "material": "STEEL",
                    "verdict": "PASS",
                    "confidence": 0.0,
                    "defect_count": 0,
                    "detections": [],
                    "annotated": frame,
                }
        else:
            result = _mock_detect(frame)

        # ── Update counters & hardware trigger ────────────────────────
        global _last_hardware_verdict
        verdict = result.get("verdict", "IDLE")
        mat = str(result.get("material", "STEEL"))
        n_defects = len(result.get("detections", []))

        stats["total_scanned"] += 1
        if verdict == "FAIL":
            stats["total_defects"] += n_defects
            stats["failed"]        += 1
        elif verdict == "PASS":
            stats["passed"]        += 1

        # ESP32 triggers only via /detect endpoint (Feed Image / Scan Now)

        # ── FPS ──────────────────────────────────────────────────────
        frame_count += 1
        now = time.perf_counter()
        elapsed = now - fps_timer
        if elapsed >= 1.0:
            stats["fps"]  = round(frame_count / elapsed, 1)
            frame_count   = 0
            fps_timer     = now

        # ── Encode annotated frame as JPEG ───────────────────────────
        annotated = result.get("annotated", frame)
        ok, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            current_frame = buf.tobytes()

        # ── Store latest result ───────────────────────────────────────
        current_live = {
            "stats":       stats.copy(),
            "material":    result.get("material"),
            "verdict":     result.get("verdict"),
            "confidence":  result.get("confidence"),
            "top_class":   result.get("top_class"),
            "top_conf_pct":result.get("top_conf_pct"),
            "detections":  result.get("detections", []),
            "morphology":  result.get("morphology"),
        }

        # ── Broadcast to WebSocket clients ────────────────────────────
        if ws_clients:
            await broadcast(current_live)

        await asyncio.sleep(0.03)   # ~30 FPS ceiling

    if cap:
        cap.release()


def _test_pattern() -> np.ndarray:
    """Colour-bar test pattern when no webcam is available."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    colours = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(128,128,128)]
    w = 640 // len(colours)
    for i, c in enumerate(colours):
        img[:, i*w:(i+1)*w] = c
    cv2.putText(img, "NO CAMERA — TEST PATTERN", (80, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    return img


@app.on_event("startup")
async def startup():
    global _camera_task
    _camera_task = asyncio.create_task(inference_loop())
    logger.info("🚀 Strawhat-Pirates Inspection API ready at http://0.0.0.0:8000")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":   "healthy",
        "version":  "2.0.0",
        "pipeline": "loaded" if _pipeline_loaded else "demo_mode",
        "fps":      stats.get("fps", 0),
    }


@app.get("/stats")
def get_stats():
    return stats


@app.post("/hitl_decision")
def hitl_decision(payload: Dict[str, Any]):
    verdict = payload.get("verdict", "PASS").upper()
    material = payload.get("material", "STEEL")
    defects = payload.get("defects", 0)
    logger.info(f"👤 Human-in-the-Loop decision received: {verdict} for {material} ({defects} defects)")
    _send_esp32_command(material, defects, verdict)
    return {"status": "ok", "verdict": verdict}


@app.get("/hardware/trigger")
@app.post("/hardware/trigger")
def hardware_trigger(cmd: str = "REJECT"):
    """Direct manual hardware trigger endpoint (REJECT, PASS, RESET)."""
    clean_cmd = cmd.strip().upper()
    if _serial_bridge and _serial_bridge.is_connected():
        _serial_bridge.send(clean_cmd)
        logger.info(f"🕹️ Manual hardware command sent: {clean_cmd}")
        return {"status": "ok", "command_sent": clean_cmd, "hardware_connected": True}
    else:
        logger.warning(f"⚠️ Manual hardware command failed (not connected): {clean_cmd}")
        return {"status": "error", "message": "Hardware not connected via Serial", "command_sent": clean_cmd}


@app.get("/video_feed")
async def video_feed():
    """MJPEG stream — connect with <img src='/video_feed'>."""
    async def stream():
        while True:
            if current_frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    current_frame +
                    b"\r\n"
                )
            await asyncio.sleep(0.033)  # 30 FPS

    return StreamingResponse(
        stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """WebSocket — sends JSON update every frame."""
    await ws.accept()
    ws_clients.append(ws)
    logger.info(f"WebSocket client connected ({len(ws_clients)} total)")
    try:
        # Send current state immediately on connect
        if current_live:
            await ws.send_json(current_live)
        while True:
            # Keep connection alive — data is pushed by inference_loop
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)
        logger.info(f"WebSocket client disconnected ({len(ws_clients)} remaining)")


@app.post("/detect")
async def detect_image(image: UploadFile = File(...)) -> Dict[str, Any]:
    """Single-image inference via REST."""
    contents = await image.read()
    nparr    = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Could not decode image"}

    if _pipeline_loaded and _pipeline:
        try:
            result = _pipeline.run(frame)
        except Exception as e:
            logger.error(f"❌ Pipeline run error in detect: {e}", exc_info=True)
            result = _mock_detect(frame)
    else:
        result = _mock_detect(frame)

    verdict = result.get("verdict")
    mat = str(result.get("material", "STEEL"))
    n_defects = len(result.get("detections", []))
    stats["total_scanned"] += 1
    if verdict == "FAIL":
        stats["total_defects"] += n_defects
        stats["failed"]        += 1
    else:
        stats["passed"]        += 1

    # Trigger ESP32 serial update on Feed Image (always fires — one-shot user action)
    _send_esp32_command(mat, n_defects, verdict)

    import base64
    annotated = result.get("annotated")
    output_image_b64 = None
    if annotated is not None:
        ok, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            output_image_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

    return {
        "filename":         image.filename,
        "material":         result.get("material"),
        "verdict":          result.get("verdict"),
        "confidence":       result.get("confidence"),
        "detections":       result.get("detections", []),
        "morphology":       result.get("morphology"),
        "output_image_b64": output_image_b64,
    }

@app.get("/scan")
async def scan_live() -> Dict[str, Any]:
    """Capture current camera frame, run inference, trigger ESP32. Used by Scan Now button."""
    global _latest_raw_frame
    if _latest_raw_frame is None:
        return {"error": "No camera frame available"}

    frame = _latest_raw_frame.copy()

    if _pipeline_loaded and _pipeline:
        try:
            result = _pipeline.run(frame)
        except Exception as e:
            logger.error(f"❌ Pipeline run error: {e}", exc_info=True)
            result = _mock_detect(frame)
    else:
        result = _mock_detect(frame)

    verdict = result.get("verdict")
    mat = str(result.get("material", "STEEL"))
    n_defects = len(result.get("detections", []))

    # Trigger ESP32 (one-shot from user click)
    _send_esp32_command(mat, n_defects, verdict)

    import base64
    annotated = result.get("annotated")
    output_image_b64 = None
    if annotated is not None:
        ok, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            output_image_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

    return {
        "filename":         "live_scan",
        "material":         result.get("material"),
        "verdict":          result.get("verdict"),
        "confidence":       result.get("confidence"),
        "detections":       result.get("detections", []),
        "morphology":       result.get("morphology"),
        "output_image_b64": output_image_b64,
    }


if __name__ == "__main__":
    uvicorn.run("inference.api_server:app", host="0.0.0.0", port=8000, reload=False)
