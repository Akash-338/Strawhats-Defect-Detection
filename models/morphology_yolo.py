import torch
import torch.nn as nn
import numpy as np
import cv2
import asyncio
from typing import List, Dict, Any, Tuple
from scipy.ndimage import distance_transform_edt

from .yolo_backbone import YOLOv10Backbone
from .cross_attention import CrossAttentionFusion
from .classification_head import MorphologyClassificationHead
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from morphology.dsp_filters import DSPPreprocessor
from morphology.encoder import MorphologyEncoder
from morphology.feature_extractor import MorphologicalFeatureExtractor
from morphology.preprocessing import extract_binary_mask

class MorphologyYOLO(nn.Module):
    
    def __init__(self, num_classes: int = 6, device: str = 'cuda'):
        
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.backbone = YOLOv10Backbone('yolov10n.pt').to(self.device)
        
        self.visual_dim = 256  
        self.morph_dim = 128
        self.d_model = 128
        
        self.cross_attention = CrossAttentionFusion(
            visual_dim=self.visual_dim,
            morph_dim=self.morph_dim,
            d_model=self.d_model,
            num_heads=4
        ).to(self.device)
        
        self.classifier = MorphologyClassificationHead(
            in_features=self.visual_dim + self.d_model,
            num_classes=num_classes,
            temperature=1.0
        ).to(self.device)
        
        self.morph_encoder = MorphologyEncoder().to(self.device)
        self.morph_extractor = MorphologicalFeatureExtractor()
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
       
        dsp_img = DSPPreprocessor.full_dsp_pipeline(image)
        
        dsp_img_uint8 = cv2.normalize(dsp_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(dsp_img_uint8)
        blur_img = cv2.GaussianBlur(clahe_img, (5, 5), 0)
        
        _, binary_mask = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edt_map = distance_transform_edt(binary_mask)
        edt_map = cv2.normalize(edt_map, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        img_tensor = torch.from_numpy(blur_img).float() / 255.0
        attn_tensor = torch.from_numpy(edt_map).float()
        
        return img_tensor, attn_tensor
        
    def _extract_morphology_features(self, edt_map: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
       
        num_boxes = boxes.shape[0]
        if num_boxes == 0:
            return torch.empty((0, self.morph_dim), device=self.device)
            
        H, W = edt_map.shape
        edt_np = edt_map.cpu().numpy()
        
        edt_uint8 = (edt_np * 255).astype(np.uint8)
        
        raw_features_list = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            abs_x1 = int(max(0, x1 * W))
            abs_y1 = int(max(0, y1 * H))
            abs_x2 = int(min(W, x2 * W))
            abs_y2 = int(min(H, y2 * H))
            
            if abs_x2 > abs_x1 and abs_y2 > abs_y1:
                crop = edt_uint8[abs_y1:abs_y2, abs_x1:abs_x2]
                mask = extract_binary_mask(crop, method='otsu')
                features_dict = self.morph_extractor.extract_all(mask, crop)
                feat_tensor = self.morph_extractor.to_tensor(features_dict)
            else:
                features_dict = self.morph_extractor._empty_features()
                feat_tensor = self.morph_extractor.to_tensor(features_dict)
                
            raw_features_list.append(feat_tensor)
            
        raw_features = torch.stack(raw_features_list).to(self.device) # Shape: (N, 11)
        
        encoded_morph = self.morph_encoder(raw_features)
        
        return encoded_morph

    def forward(self, image: np.ndarray) -> List[Dict[str, Any]]:
        
        img_t, attn_t = self.preprocess_image(image)
        
        input_tensor = torch.stack([img_t, attn_t], dim=0).unsqueeze(0).to(self.device)
        
        yolo_results = self.backbone(input_tensor)
        
        results_obj = yolo_results[0]
        detected_boxes = results_obj.boxes.xyxyn  # [N, 4] normalized coords
        
        if detected_boxes.shape[0] == 0:
            return []
            
        f_visual = self.backbone.get_roi_features(detected_boxes)
        
        if f_visual.shape[-1] != self.visual_dim and f_visual.numel() > 0:
            if f_visual.shape[-1] < self.visual_dim:
                pad = torch.zeros(f_visual.shape[0], self.visual_dim - f_visual.shape[-1], device=self.device)
                f_visual = torch.cat([f_visual, pad], dim=-1)
            else:
                f_visual = f_visual[:, :self.visual_dim]
                
        f_morph = self._extract_morphology_features(attn_t, detected_boxes)
        
        fused_features = self.cross_attention(f_visual, f_morph)
        
        class_probs = self.classifier(fused_features)
        
        results = []
        for i in range(detected_boxes.shape[0]):
            box = detected_boxes[i].cpu().numpy()
            probs = class_probs[i]
            conf, cls_idx = torch.max(probs, dim=-1)
            
            results.append({
                'bbox': box.tolist(),
                'class': int(cls_idx.item()),
                'confidence': float(conf.item()),
                'morphology_features': f_morph[i].detach().cpu().numpy().tolist()
            })
            
        return results

    async def async_forward(self, image: np.ndarray) -> List[Dict[str, Any]]:
        
        # Run preprocessing in a thread if it's CPU bound
        loop = asyncio.get_event_loop()
        img_t, attn_t = await loop.run_in_executor(None, self.preprocess_image, image)
        
        input_tensor = torch.stack([img_t, attn_t], dim=0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            yolo_results = self.backbone(input_tensor)
            results_obj = yolo_results[0]
            detected_boxes = results_obj.boxes.xyxyn  # [N, 4] normalized coords
            
            if detected_boxes.shape[0] == 0:
                return []
                
            f_visual = self.backbone.get_roi_features(detected_boxes)
            
            if f_visual.shape[-1] != self.visual_dim and f_visual.numel() > 0:
                if f_visual.shape[-1] < self.visual_dim:
                    pad = torch.zeros(f_visual.shape[0], self.visual_dim - f_visual.shape[-1], device=self.device)
                    f_visual = torch.cat([f_visual, pad], dim=-1)
                else:
                    f_visual = f_visual[:, :self.visual_dim]
                
            f_morph = self._extract_morphology_features(attn_t, detected_boxes)
            fused_features = self.cross_attention(f_visual, f_morph)
            class_probs = self.classifier(fused_features)
            
        # 
        results = []
        for i in range(detected_boxes.shape[0]):
            conf, cls_idx = torch.max(class_probs[i], dim=-1)
            results.append({
                'bbox': detected_boxes[i].cpu().numpy().tolist(),
                'class': int(cls_idx.item()),
                'confidence': float(conf.item()),
                'morphology_features': f_morph[i].detach().cpu().numpy().tolist()
            })
            
        return results
