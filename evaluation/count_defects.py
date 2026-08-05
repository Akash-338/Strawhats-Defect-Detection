import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def nms(boxes: List, confidences: List, threshold: float = 0.5) -> List:
    """Mock NMS implementation."""
    # In a real scenario, use torchvision.ops.nms
    return boxes

def process_defects(image_id: str, boxes: List, classes: List, confidences: List, 
                    reject_classes: List[str] = ["crack"]) -> Dict[str, Any]:
    """
    Counts defects and determines pass/fail.
    """
    filtered_boxes = nms(boxes, confidences)
    
    # Mock data for demonstration
    class_names = ["crack", "scratch", "dent", "good"]
    # Suppose we detected these classes based on index
    detected_classes = [class_names[c] for c in classes[:len(filtered_boxes)]]
    
    counts = {c: 0 for c in class_names}
    for cls in detected_classes:
        counts[cls] += 1
        
    total_defects = sum(counts[c] for c in counts if c != "good")
    
    pass_fail = "PASS"
    for r_cls in reject_classes:
        if counts.get(r_cls, 0) > 0:
            pass_fail = "FAIL"
            break
            
    result = {
        "image_id": image_id,
        "total_defects": total_defects,
        "per_class_counts": counts,
        "pass_fail": pass_fail
    }
    return result

def log_to_sqlite(db_path: str, result: Dict[str, Any]):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS defect_logs (
            image_id TEXT PRIMARY KEY,
            total_defects INTEGER,
            crack_count INTEGER,
            scratch_count INTEGER,
            dent_count INTEGER,
            pass_fail TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        INSERT OR REPLACE INTO defect_logs 
        (image_id, total_defects, crack_count, scratch_count, dent_count, pass_fail)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        result['image_id'],
        result['total_defects'],
        result['per_class_counts'].get('crack', 0),
        result['per_class_counts'].get('scratch', 0),
        result['per_class_counts'].get('dent', 0),
        result['pass_fail']
    ))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-id", type=str, default="img_001")
    parser.add_argument("--db-path", type=str, default="defects.db")
    args = parser.parse_args()
    
    # Mock detections
    res = process_defects(args.image_id, [[0,0,10,10], [20,20,30,30]], [0, 1], [0.9, 0.85], reject_classes=["crack"])
    print(json.dumps(res, indent=4))
    log_to_sqlite(args.db_path, res)
    logging.info(f"Logged {args.image_id} to {args.db_path}")
