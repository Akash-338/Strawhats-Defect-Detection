import cv2
import numpy as np
import logging
from typing import Union, Optional, Tuple

logger = logging.getLogger(__name__)

def _load_image(image: Union[str, np.ndarray]) -> np.ndarray:
    """
    Loads an image from file path or returns the image if it's already a numpy array.
    Ensures the image is grayscale.
    """
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image from {image}")
        return img
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()
    else:
        raise TypeError("Image must be a file path or a numpy array.")

def apply_clahe(image: Union[str, np.ndarray], clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    """
    img = _load_image(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)

def gaussian_smooth(image: Union[str, np.ndarray], kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Applies Gaussian smoothing to the image.
    """
    img = _load_image(image)
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma)

def preprocess_image(image: Union[str, np.ndarray]) -> np.ndarray:
    """
    Full preprocessing pipeline: Grayscale conversion -> CLAHE -> Gaussian Smoothing.
    """
    img = _load_image(image)
    clahe_img = apply_clahe(img)
    smooth_img = gaussian_smooth(clahe_img)
    return smooth_img

def extract_binary_mask(image: Union[str, np.ndarray], method: str = 'adaptive') -> np.ndarray:
    """
    Extracts a binary mask for defects using the specified thresholding method.
    """
    img = _load_image(image)
    if method == 'otsu':
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    else:
        raise ValueError("Method must be 'otsu' or 'adaptive'")
    return mask
