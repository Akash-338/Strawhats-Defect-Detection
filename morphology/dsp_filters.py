import numpy as np
from scipy import signal
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
from typing import Union, Optional

class DSPPreprocessor:
    """Digital Signal Processing preprocessing for industrial surface images.
    
    Applies spatial filters and normalization to suppress high-frequency noise
    and refine visual data before deep learning inference.
    Matches approach doc: 'strict DSP methodologies — FIR/IIR filters + matrix normalization'
    """
    
    @staticmethod
    def apply_fir_lowpass(image: np.ndarray, cutoff: float = 0.3, order: int = 5) -> np.ndarray:
        """2D FIR low-pass filter using scipy.signal.
        
        Suppresses high-frequency sensor noise while preserving defect edges.
        
        Parameters
        ----------
        image : np.ndarray
            Input 2D grayscale image.
        cutoff : float, optional
            Normalized cutoff frequency (0 to 1), by default 0.3
        order : int, optional
            Filter order, by default 5
            
        Returns
        -------
        np.ndarray
            Filtered image.
        """
        # Create a 2D FIR filter using outer product of 1D FIR filters
        b = signal.firwin(order, cutoff)
        b2 = np.outer(b, b)
        
        # Apply 2D convolution
        filtered = signal.convolve2d(image, b2, mode='same', boundary='symm')
        return filtered

    @staticmethod
    def apply_iir_bandpass(image: np.ndarray, low_cutoff: float = 0.05, high_cutoff: float = 0.4) -> np.ndarray:
        """2D IIR band-pass filter. 
        
        Preserves defect-relevant spatial frequencies, suppresses both low-freq 
        illumination gradients and high-freq noise.
        
        Parameters
        ----------
        image : np.ndarray
            Input 2D grayscale image.
        low_cutoff : float, optional
            Lower normalized cutoff frequency (0 to 1), by default 0.05
        high_cutoff : float, optional
            Higher normalized cutoff frequency (0 to 1), by default 0.4
            
        Returns
        -------
        np.ndarray
            Filtered image.
        """
        # Design IIR butterworth filter in 1D
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='bandpass')
        
        # Apply sequentially along both axes for a rough 2D IIR approximation
        filtered_x = signal.filtfilt(b, a, image, axis=0)
        filtered = signal.filtfilt(b, a, filtered_x, axis=1)
        return filtered

    @staticmethod
    def matrix_normalization(image: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """Matrix normalization. Options: 'zscore', 'minmax', 'whitening'.
        
        Normalizes pixel intensity distribution for consistent model input.
        
        Parameters
        ----------
        image : np.ndarray
            Input image array.
        method : str, optional
            Normalization method ('zscore', 'minmax', 'whitening'), by default 'zscore'
            
        Returns
        -------
        np.ndarray
            Normalized image array.
            
        Raises
        ------
        ValueError
            If an invalid method is provided.
        """
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
            # ZCA Whitening-like (simplified)
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
        """Wiener deconvolution filter for motion blur compensation.
        
        Parameters
        ----------
        image : np.ndarray
            Input grayscale image.
        noise_variance : float, optional
            Variance of noise. If None, estimated from local variance.
            
        Returns
        -------
        np.ndarray
            Restored image.
        """
        # Using scipy's wiener filter which applies a Wiener filter to an N-dimensional array.
        return signal.wiener(image.astype(np.float64), noise=noise_variance)

    @classmethod
    def full_dsp_pipeline(cls, image: np.ndarray) -> np.ndarray:
        """Complete DSP chain: FIR lowpass → matrix normalization → output.
        
        This is applied BEFORE CLAHE and morphology attention map generation.
        
        Parameters
        ----------
        image : np.ndarray
            Raw input image.
            
        Returns
        -------
        np.ndarray
            Processed image, ready for further stages.
        """
        # 1. FIR lowpass to suppress noise
        filtered = cls.apply_fir_lowpass(image)
        
        # 2. Matrix normalization (zscore)
        normalized = cls.matrix_normalization(filtered, method='zscore')
        
        return normalized
