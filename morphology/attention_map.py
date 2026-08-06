import numpy as np
import cv2
import torch
import logging

logger = logging.getLogger(__name__)

def compute_edt(binary_mask: np.ndarray) -> np.ndarray:
    mask = binary_mask.astype(np.uint8)
    if mask.max() == 1:
        mask = mask * 255
    inverted_mask = 255 - mask
    edt = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
    return edt

def generate_attention_map(binary_mask: np.ndarray, tau: float = 10.0) -> np.ndarray:
    
    edt = compute_edt(binary_mask)
    attention_map = np.exp(-(edt ** 2) / (2 * tau ** 2))
    return attention_map

def create_two_channel_input(image: np.ndarray, attention_map: np.ndarray) -> torch.Tensor:
    img_tensor = torch.from_numpy(image).float()
    
    if img_tensor.ndim == 2:
        img_tensor = img_tensor.unsqueeze(0)
    elif img_tensor.ndim == 3 and img_tensor.shape[2] == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.mean(dim=0, keepdim=True)
    elif img_tensor.ndim == 3 and img_tensor.shape[0] == 3:
        img_tensor = img_tensor.mean(dim=0, keepdim=True)

    attn_tensor = torch.from_numpy(attention_map).float()
    if attn_tensor.ndim == 2:
        attn_tensor = attn_tensor.unsqueeze(0)
        
    return torch.cat([img_tensor, attn_tensor], dim=0)
