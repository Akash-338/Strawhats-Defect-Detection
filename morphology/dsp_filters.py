import numpy as np
from scipy import signal
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
from typing import Union, Optional

class DSPPreprocessor:
    
    
    @staticmethod
    def apply_fir_lowpass(image: np.ndarray, cutoff: float = 0.3, order: int = 5) -> np.ndarray:
        
        b = signal.firwin(order, cutoff)
        b2 = np.outer(b, b)
        
        filtered = signal.convolve2d(image, b2, mode='same', boundary='symm')
        return filtered

    @staticmethod
    def apply_iir_bandpass(image: np.ndarray, low_cutoff: float = 0.05, high_cutoff: float = 0.4) -> np.ndarray:
        
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='bandpass')
        
        # Apply sequentially along both axes for a rough 2D IIR approximation
        filtered_x = signal.filtfilt(b, a, image, axis=0)
        filtered = signal.filtfilt(b, a, filtered_x, axis=1)
        return filtered

    @staticmethod
    def matrix_normalization(image: np.ndarray, method: str = 'zscore') -> np.ndarray:
        
        image_float = image.astype(np.float32)
        
        if method == 'zscore':
            mean = np.mean(image_float)
            std = np.std(image_float)
            if std == 0:
                return image_float - mean
            return (image_float - mean) / std
            
        elif method == 'minmax':
            min_val = np.min(image_float)
            max_val = np.max(image_float)
            if max_val - min_val == 0:
                return np.zeros_like(image_float)
            return (image_float - min_val) / (max_val - min_val)
            
        elif method == 'whitening':
            mean = np.mean(image_float)
            centered = image_float - mean
            cov = np.cov(centered, rowvar=False)
            U, S, V = np.linalg.svd(cov)
            epsilon = 1e-5
            zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
            return np.dot(centered, zca_matrix)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @staticmethod
    def apply_wiener_filter(image: np.ndarray, noise_variance: Optional[float] = None) -> np.ndarray:
        
        return signal.wiener(image.astype(np.float64), noise=noise_variance)

    @classmethod
    def full_dsp_pipeline(cls, image: np.ndarray) -> np.ndarray:
        
        filtered = cls.apply_fir_lowpass(image)
        
        normalized = cls.matrix_normalization(filtered, method='zscore')
        
        return normalized
