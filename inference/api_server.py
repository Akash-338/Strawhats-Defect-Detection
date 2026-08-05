import asyncio
from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import uvicorn
import io
import cv2
import numpy as np

app = FastAPI(title="Morphology-Aware Inspection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global stats
stats = {
    "total_scanned": 0,
    "total_defects": 0,
    "passed": 0,
    "failed": 0
}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/stats")
def get_stats():
    return stats

@app.post("/detect")
async def detect(image: UploadFile = File(...)) -> Dict[str, Any]:
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process image (Mock)
    stats["total_scanned"] += 1
    
    # Dummy logic
    has_defect = True if stats["total_scanned"] % 3 == 0 else False
    
    if has_defect:
        stats["total_defects"] += 1
        stats["failed"] += 1
        pass_fail = "FAIL"
    else:
        stats["passed"] += 1
        pass_fail = "PASS"
        
    return {
        "filename": image.filename,
        "detections": 1 if has_defect else 0,
        "counts": {"crack": 1 if has_defect else 0},
        "pass_fail": pass_fail,
        "morphology_features": {
            "crack_1": {"Area": 120, "Aspect Ratio": 0.4} if has_defect else {}
        }
    }

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Mock streaming results every second
            await asyncio.sleep(1)
            await websocket.send_json({
                "type": "live_update",
                "stats": stats
            })
    except Exception as e:
        print(f"WebSocket disconnected: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
