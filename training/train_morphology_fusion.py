import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob

# Add parent directory to path to allow importing from models/ and morphology/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.losses import MorphologyAwareLoss
from models.yolo_backbone import YOLOv10Backbone
from models.cross_attention import CrossAttentionFusion
from models.classification_head import MorphologyClassificationHead
from morphology.encoder import MorphologyEncoder
from morphology.feature_extractor import MorphologicalFeatureExtractor
from morphology.preprocessing import preprocess_image, extract_binary_mask
from morphology.attention_map import generate_attention_map, create_two_channel_input
from morphology.dsp_filters import DSPPreprocessor

def compute_iou(box1, box2):
    """Compute IoU between two boxes (x1, y1, x2, y2) normalized."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def match_boxes(pred_boxes, gt_boxes, gt_classes, iou_threshold=0.5):
    """Match predicted boxes to ground truth boxes using IoU."""
    matched_targets = []
    valid_mask = []
    
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return torch.tensor([]), torch.tensor([], dtype=torch.bool)
        
    for p_box in pred_boxes:
        best_iou = 0
        best_cls = -1
        for g_box, g_cls in zip(gt_boxes, gt_classes):
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_cls = g_cls
                
        if best_iou >= iou_threshold:
            matched_targets.append(best_cls)
            valid_mask.append(True)
        else:
            matched_targets.append(-1)
            valid_mask.append(False)
            
    return torch.tensor(matched_targets), torch.tensor(valid_mask, dtype=torch.bool)

class MorphologyFusionDataset(Dataset):
    def __init__(self, yaml_path, split='train', img_size=640):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        base_path = data.get('path', os.path.dirname(yaml_path))
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(yaml_path), base_path)
            
        img_dir = os.path.join(base_path, data[split])
        self.img_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.img_paths.extend(list(Path(img_dir).rglob(ext)))
            
        self.img_size = img_size
        self.nc = data['nc']

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = str(self.img_paths[idx])
        try:
            # np.fromfile handles non-ASCII/Unicode paths on Windows
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        except Exception:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
        if img is None:
            # Fallback for missing/corrupt image: return blank image
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
        # Ensure 2D grayscale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Resize image
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        
        label_path = img_path.replace('images', 'labels')
        label_path = os.path.splitext(label_path)[0] + '.txt'
        
        boxes = []
        classes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        classes.append(int(parts[0]))
                        # xywh to x1y1x2y2 normalized
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = max(0.0, cx - bw/2)
                        y1 = max(0.0, cy - bh/2)
                        x2 = min(1.0, cx + bw/2)
                        y2 = min(1.0, cy + bh/2)
                        boxes.append([x1, y1, x2, y2])
                        
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            classes = np.array(classes, dtype=np.int64)
            
        return {
            'image': img_resized,
            'boxes': torch.tensor(boxes),
            'classes': torch.tensor(classes)
        }

def collate_fn(batch):
    images = [item['image'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    classes = [item['classes'] for item in batch]
    return images, boxes, classes

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle bare integer device string (e.g. '0' → 'cuda:0')
    dev_str = f'cuda:{args.device}' if args.device.isdigit() else args.device
    device = torch.device(dev_str if torch.cuda.is_available() else 'cpu')
    epochs = args.epochs if args.epochs else config['training']['epochs_fusion']
    # Fusion always uses 640 — backbone is frozen so imgsz doesn't change accuracy
    # but 800 would make the CPU morphology extraction very slow
    img_size = 640
    batch_size = args.batch if args.batch else config['training'].get('batch_size', 4)
    
    if args.data:
        dataset_yaml = args.data
    else:
        default_data = {
            'steel': 'data/processed/steel_unified/dataset.yaml',
            'aluminum': 'data/dataset_aluminum.yaml',
            'wood': 'data/dataset_wood_10class.yaml'
        }
        dataset_yaml = default_data.get(args.material, config['paths'].get('dataset_yaml', 'data/processed/steel_unified/dataset.yaml'))
    
    # Determine weights with automatic fallback search
    if args.material == 'steel':
        weights_path = config['paths']['steel_weights']
    elif args.material == 'aluminum':
        weights_path = config['paths']['aluminum_weights']
    elif args.material == 'wood':
        weights_path = config['paths']['wood_weights']
    else:
        raise ValueError(f"Unknown material: {args.material}")
        
    if not os.path.exists(weights_path):
        # Fallback candidates
        candidates = [
            f"runs/{args.material}/weights/best.pt",
            f"runs/{args.material}/train/weights/best.pt",
            f"runs/detect/runs/{args.material}/weights/best.pt",
            f"runs/detect/runs/{args.material}/train/weights/best.pt",
            f"runs/detect/{args.material}/weights/best.pt",
            f"runs/detect/{args.material}/train/weights/best.pt",
        ]
        found = False
        for cand in candidates:
            if os.path.exists(cand):
                weights_path = cand
                found = True
                break
        if not found:
            # Search in runs/ for YOLO best.pt — exclude morphology_fusion checkpoints
            matches = [
                p for p in Path("runs").rglob(f"*{args.material}*/best.pt")
                if 'morphology_fusion' not in str(p)
            ]
            if matches:
                weights_path = str(matches[0])
                found = True
        if not found:
            print(f"WARNING: Best weights for {args.material} not found at {weights_path}, using default backbone yolov10m.pt")
            weights_path = "yolov10m.pt"
    
    print(f"Starting Morphology Fusion training for {epochs} epochs on {device}...")
    print(f"Using material: {args.material}, weights: {weights_path}, dataset: {dataset_yaml}")

    # ── Resolve absolute dataset root (mirrors train_yolo_baseline.py logic) ──
    yaml_abs = Path(dataset_yaml).resolve()
    yaml_name_lower = str(yaml_abs).lower()
    if (yaml_abs.parent / "train").exists():
        root_dir = yaml_abs.parent
    elif "wood" in yaml_name_lower and "3k" in yaml_name_lower:
        root_dir = (Path("data") / "processed" / "wood_3k").resolve()
    elif "wood" in yaml_name_lower:
        root_dir = (Path("data") / "processed" / "wood_10class").resolve()
    elif "steel" in yaml_name_lower:
        root_dir = (Path("data") / "processed" / "steel_unified").resolve()
    else:
        root_dir = yaml_abs.parent

    with open(dataset_yaml, 'r') as f:
        ds_yaml = yaml.safe_load(f)
    ds_yaml['path'] = str(root_dir).replace('\\', '/')
    with open(dataset_yaml, 'w') as f:
        yaml.dump(ds_yaml, f, sort_keys=False)
    print(f"[Dataset] Resolved root: {root_dir}")
    # ──────────────────────────────────────────────────────────────────────────

    train_dataset = MorphologyFusionDataset(dataset_yaml, split='train', img_size=img_size)
    val_dataset   = MorphologyFusionDataset(dataset_yaml, split='val',   img_size=img_size)
    print(f"[Dataset] train={len(train_dataset)} images, val={len(val_dataset)} images")
    if len(train_dataset) == 0:
        raise RuntimeError(f"Train dataset is empty! Check path: {root_dir}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Initialize components
    backbone = YOLOv10Backbone(weights_path).to(device)
    
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
        
    num_classes = train_dataset.nc
    
    # Dynamically determine visual_dim from backbone
    with torch.no_grad():
        dummy_img = torch.zeros((1, 2, img_size, img_size), device=device)
        _ = backbone(dummy_img)
        dummy_box = torch.tensor([[0.1, 0.1, 0.5, 0.5]], device=device)
        dummy_feat = backbone.get_roi_features(dummy_box)
        visual_dim = dummy_feat.shape[-1]
    print(f"Extracted backbone visual feature dimension: {visual_dim}")
    
    morph_dim = 128
    
    morph_encoder = MorphologyEncoder().to(device)
    cross_attention = CrossAttentionFusion(visual_dim=visual_dim, morph_dim=morph_dim, d_model=morph_dim, num_heads=4).to(device)
    classifier = MorphologyClassificationHead(in_features=visual_dim + morph_dim, num_classes=num_classes).to(device)
    
    feature_extractor = MorphologicalFeatureExtractor()
    
    # Optimizers for fusion components only
    trainable_params = list(morph_encoder.parameters()) + \
                       list(cross_attention.parameters()) + \
                       list(classifier.parameters())
                       
    optimizer = torch.optim.AdamW(trainable_params, 
                                  lr=config['training'].get('fusion_lr', 0.0001),
                                  weight_decay=config['training']['weight_decay'])
                                  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion = MorphologyAwareLoss(alpha=config['training']['alpha'], lambda_morph=config['training']['lambda_morph']).to(device)
    
    writer = SummaryWriter(log_dir=os.path.join('runs', 'morphology_fusion', args.material))
    
    best_val_loss = float('inf')
    early_stop_patience = config['training'].get('early_stopping_patience', 15)
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        morph_encoder.train()
        cross_attention.train()
        classifier.train()
        
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, boxes_list, classes_list in pbar:
            batch_loss = 0
            
            optimizer.zero_grad()
            
            for i in range(len(images)):
                img_np = images[i]
                gt_boxes = boxes_list[i].to(device)
                gt_classes = classes_list[i].to(device)
                
                if len(gt_boxes) == 0:
                    continue
                    
                # Preprocess
                # Using morphology pipeline
                dsp_img = DSPPreprocessor.full_dsp_pipeline(img_np)
                dsp_img_uint8 = cv2.normalize(dsp_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                clahe_img = clahe.apply(dsp_img_uint8)
                blur_img = cv2.GaussianBlur(clahe_img, (5, 5), 0)
                
                _, binary_mask = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                attn_map = generate_attention_map(binary_mask, tau=config['model'].get('morphology_tau', 10.0))
                
                img_tensor = torch.from_numpy(blur_img).float() / 255.0
                attn_tensor = torch.from_numpy(attn_map).float()
                
                input_tensor = torch.stack([img_tensor, attn_tensor], dim=0).unsqueeze(0).to(device)
                
                # Forward backbone
                with torch.no_grad():
                    _ = backbone(input_tensor)
                    
                    # We will use ground-truth boxes to train the classification head
                    # Extract visual features
                    f_visual = backbone.get_roi_features(gt_boxes, batch_idx=0)
                    
                    # Pad/slice visual features if dimension mismatch
                    if f_visual.shape[-1] != visual_dim and f_visual.numel() > 0:
                        if f_visual.shape[-1] < visual_dim:
                            pad = torch.zeros(f_visual.shape[0], visual_dim - f_visual.shape[-1], device=device)
                            f_visual = torch.cat([f_visual, pad], dim=-1)
                        else:
                            f_visual = f_visual[:, :visual_dim]
                
                # Extract morph features from cropped regions
                raw_morph_features = []
                for box in gt_boxes:
                    x1, y1, x2, y2 = box.cpu().numpy()
                    H, W = binary_mask.shape
                    # crop
                    cx1, cy1 = int(x1*W), int(y1*H)
                    cx2, cy2 = int(x2*W), int(y2*H)
                    cx1 = max(0, cx1); cy1 = max(0, cy1)
                    cx2 = min(W, cx2); cy2 = min(H, cy2)
                    
                    if cx2 > cx1 and cy2 > cy1:
                        mask_crop = binary_mask[cy1:cy2, cx1:cx2]
                        img_crop = blur_img[cy1:cy2, cx1:cx2]
                        feats_dict = feature_extractor.extract_all(mask_crop, img_crop)
                    else:
                        feats_dict = feature_extractor._empty_features()
                        
                    raw_morph = feature_extractor.to_tensor(feats_dict)
                    raw_morph_features.append(raw_morph)
                    
                if not raw_morph_features:
                    continue
                    
                raw_morph_tensor = torch.stack(raw_morph_features).to(device)
                
                # Encode morph features
                f_morph = morph_encoder(raw_morph_tensor)
                
                # Fuse features
                fused = cross_attention(f_visual, f_morph)
                
                # Classify
                logits = classifier(fused)
                
                # L_morph: reconstruction consistency loss
                # morph_preds = decode(f_morph) → reconstructed 11-dim descriptors
                # morph_targets = raw_morph_tensor → original 11-dim descriptors
                # MSE(reconstructed, original) gives a meaningful consistency signal
                morph_preds = morph_encoder.decode(f_morph)   # (N, 11)
                morph_targets = raw_morph_tensor              # (N, 11)
                
                loss, loss_dict = criterion(morph_preds, morph_targets, logits, gt_classes, yolo_loss=0.0)
                loss.backward()
                batch_loss += loss.item()
                
            optimizer.step()
            epoch_loss += batch_loss
            pbar.set_postfix({'loss': batch_loss})
            
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        # Validation placeholder (can compute accuracy instead of mAP for simplicity, since it's a classification head)
        morph_encoder.eval()
        cross_attention.eval()
        classifier.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, boxes_list, classes_list in val_loader:
                for i in range(len(images)):
                    img_np = images[i]
                    gt_boxes = boxes_list[i].to(device)
                    gt_classes = classes_list[i].to(device)
                    
                    if len(gt_boxes) == 0:
                        continue
                        
                    dsp_img = DSPPreprocessor.full_dsp_pipeline(img_np)
                    dsp_img_uint8 = cv2.normalize(dsp_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    clahe_img = clahe.apply(dsp_img_uint8)
                    blur_img = cv2.GaussianBlur(clahe_img, (5, 5), 0)
                    
                    _, binary_mask = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    attn_map = generate_attention_map(binary_mask, tau=config['model'].get('morphology_tau', 10.0))
                    
                    img_tensor = torch.from_numpy(blur_img).float() / 255.0
                    attn_tensor = torch.from_numpy(attn_map).float()
                    
                    input_tensor = torch.stack([img_tensor, attn_tensor], dim=0).unsqueeze(0).to(device)
                    
                    _ = backbone(input_tensor)
                    f_visual = backbone.get_roi_features(gt_boxes, batch_idx=0)
                    
                    if f_visual.shape[-1] != visual_dim and f_visual.numel() > 0:
                        if f_visual.shape[-1] < visual_dim:
                            pad = torch.zeros(f_visual.shape[0], visual_dim - f_visual.shape[-1], device=device)
                            f_visual = torch.cat([f_visual, pad], dim=-1)
                        else:
                            f_visual = f_visual[:, :visual_dim]
                            
                    raw_morph_features = []
                    for box in gt_boxes:
                        x1, y1, x2, y2 = box.cpu().numpy()
                        H, W = binary_mask.shape
                        cx1, cy1 = int(x1*W), int(y1*H)
                        cx2, cy2 = int(x2*W), int(y2*H)
                        cx1 = max(0, cx1); cy1 = max(0, cy1)
                        cx2 = min(W, cx2); cy2 = min(H, cy2)
                        
                        if cx2 > cx1 and cy2 > cy1:
                            mask_crop = binary_mask[cy1:cy2, cx1:cx2]
                            img_crop = blur_img[cy1:cy2, cx1:cx2]
                            feats_dict = feature_extractor.extract_all(mask_crop, img_crop)
                        else:
                            feats_dict = feature_extractor._empty_features()
                            
                        raw_morph = feature_extractor.to_tensor(feats_dict)
                        raw_morph_features.append(raw_morph)
                        
                    if not raw_morph_features:
                        continue
                        
                    raw_morph_tensor = torch.stack(raw_morph_features).to(device)
                    f_morph = morph_encoder(raw_morph_tensor)
                    fused = cross_attention(f_visual, f_morph)
                    logits = classifier(fused)
                    
                    loss, _ = criterion(morph_encoder.decode(f_morph), raw_morph_tensor, logits, gt_classes, yolo_loss=0.0)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == gt_classes).sum().item()
                    total += gt_classes.size(0)
                    
        val_acc = correct / total if total > 0 else 0
        avg_val_loss = val_loss / len(val_loader)
        
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            save_dir = os.path.join('runs', 'morphology_fusion', args.material, 'weights')
            os.makedirs(save_dir, exist_ok=True)
            
            torch.save({
                'morph_encoder': morph_encoder.state_dict(),
                'cross_attention': cross_attention.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'val_acc': val_acc
            }, os.path.join(save_dir, 'best.pt'))
            print("Saved new best model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Morphology Fusion training completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Morphology Fusion model.")
    parser.add_argument('--config', type=str, default='training/config.yaml', help='Path to config file.')
    parser.add_argument('--material', type=str, required=True, choices=['steel', 'aluminum', 'wood'], help='Material baseline to fine-tune.')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset.yaml.')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs.')
    parser.add_argument('--batch', type=int, default=None, help='Batch size.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu or cuda).')
    args = parser.parse_args()
    main(args)
