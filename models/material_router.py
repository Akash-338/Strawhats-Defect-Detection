import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Assuming Ultralytics YOLOv8 is used
try:
    from ultralytics import YOLO
except ImportError:
    # Dummy YOLO class for type hints / testing if not installed
    class YOLO:
        def __init__(self, path):
            pass
        def __call__(self, img):
            return []

class MaterialRouter:
    """Classifies material type and routes to appropriate YOLO model."""
    
    def __init__(self, classifier_weights: str, yolo_weights_map: dict, device: str = 'cuda'):
        """
        Args:
            classifier_weights: path to ResNet18 material classifier
            yolo_weights_map: dict mapping material name -> YOLO weights path
                e.g. {'steel': 'runs/steel/best.pt', 'aluminum': 'runs/aluminum/best.pt'}
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.yolo_weights_map = yolo_weights_map
        
        # Class mapping based on training (assuming alphabetical order of folders: aluminum, steel, wood)
        # Verify this matches your training output
        self.class_names = ['aluminum', 'steel', 'wood']
        
        self.classifier = self._load_classifier(classifier_weights)
        self.yolo_models = {}
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_classifier(self, weights_path: str) -> nn.Module:
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def _get_yolo_model(self, material: str) -> YOLO:
        """Lazy load YOLO models."""
        if material not in self.yolo_models:
            weights_path = self.yolo_weights_map.get(material)
            if not weights_path:
                raise ValueError(f"No YOLO weights provided for material: {material}")
            print(f"Lazy-loading YOLO model for {material} from {weights_path}")
            self.yolo_models[material] = YOLO(weights_path)
            # Send to device if YOLO API supports it, usually handled automatically in ultralytics
        return self.yolo_models[material]

    def classify_material(self, image: Image.Image) -> tuple[str, float]:
        """Returns (material_name, confidence)"""
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            
        material = self.class_names[pred_idx.item()]
        return material, conf.item()

    def detect_defects(self, image: Image.Image) -> dict:
        """Full pipeline: classify material -> route to YOLO -> detect defects"""
        material, conf = self.classify_material(image)
        
        if conf < 0.5:
            print(f"Low material confidence ({conf:.2f}), falling back to steel.")
            material = 'steel'
            
        yolo_model = self._get_yolo_model(material)
        
        # YOLO typically takes numpy array or PIL image
        results = yolo_model(image)
        
        # Parse detections (assume results is list of ultralytics Result objects)
        detections = []
        for result in results:
            if hasattr(result, 'boxes'):
                for box in result.boxes:
                    detections.append({
                        'class_id': int(box.cls[0].item()),
                        'confidence': float(box.conf[0].item()),
                        'bbox': box.xyxy[0].cpu().numpy().tolist()
                    })
            else:
                # Mock result for dummy YOLO
                pass
                
        return {
            'material': material,
            'material_confidence': conf,
            'detections': detections,
            'defect_count': len(detections),
            'pass_fail': 'PASS' if len(detections) == 0 else 'REJECT'
        }
