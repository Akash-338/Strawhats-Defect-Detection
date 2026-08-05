import cv2
import numpy as np
import logging
from typing import Dict, Optional, Union
import torch
try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

logger = logging.getLogger(__name__)

class MorphologicalFeatureExtractor:
    """
    Extracts 11 morphological descriptors from binary defect masks.
    """
    def __init__(self):
        pass

    def extract_all(self, binary_mask: np.ndarray, original_image: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extracts all 11 morphological features from the binary mask.
        """
        if binary_mask.size == 0 or not np.any(binary_mask):
            logger.warning("Empty mask or no defects found.")
            return self._empty_features()

        # Ensure uint8 for OpenCV contour/threshold ops
        if binary_mask.dtype != np.uint8:
            binary_mask = binary_mask.astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._empty_features()

        # Consider the largest contour for global feature extraction
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area == 0:
            return self._empty_features()

        features = {}
        
        # 1. Area
        features['area'] = area
        
        # 2. Perimeter
        features['perimeter'] = cv2.arcLength(c, True)
        
        # 3. Aspect Ratio
        x, y, w, h = cv2.boundingRect(c)
        features['aspect_ratio'] = float(w) / h if h > 0 else 0.0

        # 4. Circularity
        features['circularity'] = (4 * np.pi * area) / (features['perimeter'] ** 2) if features['perimeter'] > 0 else 0.0

        # 5. Solidity
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        features['solidity'] = float(area) / hull_area if hull_area > 0 else 0.0
        
        # 6. Convex Hull Perimeter
        features['convex_hull_perimeter'] = cv2.arcLength(hull, True)
        
        # 7. Compactness
        if len(c) >= 5:
            try:
                ellipse = cv2.fitEllipse(c)
                major_axis = max(ellipse[1][0], ellipse[1][1])
                features['compactness'] = area / (np.pi * (major_axis / 2) ** 2) if major_axis > 0 else 0.0
            except cv2.error:
                features['compactness'] = 0.0
        else:
            features['compactness'] = 0.0

        # 8. Eccentricity — use numpy covariance (cv2.calcCovarMatrix crashes on OpenCV 5.0)
        if len(c) >= 5:
            try:
                c_pts = c.reshape(-1, 2).astype(np.float64)
                cov_mat = np.cov(c_pts, rowvar=False)
                eig_vals = np.linalg.eigvalsh(cov_mat)
                eig_vals = np.sort(eig_vals)[::-1]  # descending
                if eig_vals[0] > 0:
                    features['eccentricity'] = float(np.sqrt(max(0, 1 - (eig_vals[1] / eig_vals[0]))))
                else:
                    features['eccentricity'] = 0.0
            except Exception:
                features['eccentricity'] = 0.0
        else:
            features['eccentricity'] = 0.0

        # 9. Edge Density
        edges = cv2.Canny(binary_mask, 100, 200)
        features['edge_density'] = np.count_nonzero(edges) / binary_mask.size

        # 10. Skeleton Orientation
        if skeletonize is not None:
            skeleton = skeletonize(binary_mask > 0)
            y_skel, x_skel = np.where(skeleton)
            if len(x_skel) > 1:
                cov_mat = np.cov(x_skel, y_skel)
                vals, vecs = np.linalg.eig(cov_mat)
                # Principal axis angle in degrees [-90, 90]
                angle = np.arctan2(vecs[1, np.argmax(np.abs(vals))],
                                   vecs[0, np.argmax(np.abs(vals))]) * 180.0 / np.pi
                features['skeleton_orientation'] = float(np.real(angle))
            else:
                features['skeleton_orientation'] = 0.0
        else:
            features['skeleton_orientation'] = 0.0
            logger.warning("skimage not installed; skipping skeleton orientation.")

        # 11. Texture Roughness
        if original_image is not None:
            # RMS deviation of intensity from local mean
            blurred = cv2.blur(original_image, (3, 3))
            diff = (original_image.astype(np.float32) - blurred.astype(np.float32)) ** 2
            # Only consider mask pixels if desired, but for general region:
            features['texture_roughness'] = np.sqrt(np.mean(diff[binary_mask > 0])) if np.any(binary_mask > 0) else 0.0
        else:
            features['texture_roughness'] = 0.0

        return features

    def _empty_features(self) -> Dict[str, float]:
        """Returns 0 for all features in case of edge cases."""
        return {
            'area': 0.0, 'perimeter': 0.0, 'aspect_ratio': 0.0, 'circularity': 0.0,
            'solidity': 0.0, 'convex_hull_perimeter': 0.0, 'compactness': 0.0,
            'eccentricity': 0.0, 'edge_density': 0.0, 'skeleton_orientation': 0.0,
            'texture_roughness': 0.0
        }

    def normalize_features(self, features: Dict[str, float], stats: Optional[Dict] = None) -> np.ndarray:
        """
        Normalizes the extracted features using min-max or z-score.
        stats dict should contain 'mean' and 'std' for z-score, or 'min' and 'max'.
        """
        arr = np.array(list(features.values()), dtype=np.float32)
        if stats:
            if 'mean' in stats and 'std' in stats:
                arr = (arr - stats['mean']) / (stats['std'] + 1e-7)
            elif 'min' in stats and 'max' in stats:
                arr = (arr - stats['min']) / (stats['max'] - stats['min'] + 1e-7)
        return arr

    def to_tensor(self, features: Union[Dict[str, float], np.ndarray]) -> torch.Tensor:
        """
        Converts the feature dictionary or array to a torch.Tensor.
        NaN/Inf values are replaced with 0.0 to prevent training instability.
        """
        if isinstance(features, dict):
            arr = np.array(list(features.values()), dtype=np.float32)
        else:
            arr = np.asarray(features, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(arr, dtype=torch.float32)

