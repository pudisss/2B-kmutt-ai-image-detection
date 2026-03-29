"""Domain transforms for multi-domain image analysis.

This module provides transforms for extracting features from three domains:
1. RGB (spatial) - standard image normalization
2. Frequency - FFT magnitude spectrum
3. Noise - High-pass filtered residuals using SRM filters
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import cv2


class RGBTransform:
    """Standard RGB image transform with normalization."""
    
    def __init__(
        self,
        size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.size = size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to normalized tensor."""
        # Resize
        img = img.resize((self.size, self.size), Image.BILINEAR)
        
        # Convert to tensor [0, 1]
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Handle grayscale
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # HWC -> CHW
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # Normalize
        tensor = (tensor - self.mean) / self.std
        
        return tensor


class FrequencyTransform:
    """Transform image to frequency domain using 2D FFT.
    
    Extracts the log magnitude spectrum of the image, which captures
    periodic patterns and artifacts introduced by generative models.
    """
    
    def __init__(
        self,
        size: int = 256,
        log_scale: bool = True,
        normalize: bool = True,
    ):
        self.size = size
        self.log_scale = log_scale
        self.normalize = normalize
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Extract FFT magnitude spectrum from image."""
        # Resize and convert to grayscale
        img = img.resize((self.size, self.size), Image.BILINEAR)
        img_gray = img.convert('L')
        img_array = np.array(img_gray).astype(np.float32)
        
        # Apply 2D FFT
        f_transform = np.fft.fft2(img_array)
        f_shift = np.fft.fftshift(f_transform)  # Center low frequencies
        
        # Compute magnitude spectrum
        magnitude = np.abs(f_shift)
        
        # Log scale for better visualization and learning
        if self.log_scale:
            magnitude = np.log1p(magnitude)
        
        # Normalize to [0, 1]
        if self.normalize:
            magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        # Convert to tensor [1, H, W]
        tensor = torch.from_numpy(magnitude).unsqueeze(0).float()
        
        return tensor
    
    @staticmethod
    def extract_spectrum_rgb(img: Image.Image, size: int = 256) -> torch.Tensor:
        """Extract FFT spectrum for each RGB channel separately."""
        img = img.resize((size, size), Image.BILINEAR)
        img_array = np.array(img).astype(np.float32)
        
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        spectrums = []
        for c in range(3):
            f_transform = np.fft.fft2(img_array[:, :, c])
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log1p(np.abs(f_shift))
            magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
            spectrums.append(magnitude)
        
        tensor = torch.from_numpy(np.stack(spectrums, axis=0)).float()
        return tensor


class NoiseTransform:
    """Extract noise residuals using SRM (Spatial Rich Model) high-pass filters.
    
    SRM filters are designed to suppress image content and enhance
    noise patterns and subtle artifacts that distinguish real from synthetic images.
    """
    
    # SRM filter kernels (3 types)
    SRM_KERNELS = {
        'srm1': np.array([
            [ 0,  0,  0,  0,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  2, -4,  2,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  0,  0,  0,  0]
        ], dtype=np.float32) / 4.0,
        
        'srm2': np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32) / 12.0,
        
        'srm3': np.array([
            [ 0,  0, -1,  0,  0],
            [ 0,  0,  2,  0,  0],
            [-1,  2, -4,  2, -1],
            [ 0,  0,  2,  0,  0],
            [ 0,  0, -1,  0,  0]
        ], dtype=np.float32) / 4.0,
    }
    
    def __init__(
        self,
        size: int = 256,
        kernels: Optional[list] = None,
        normalize: bool = True,
        truncate: float = 3.0,
    ):
        self.size = size
        self.kernel_names = kernels or ['srm1', 'srm2', 'srm3']
        self.normalize = normalize
        self.truncate = truncate
        
        # Prepare kernels
        self.kernels = [self.SRM_KERNELS[k] for k in self.kernel_names]
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Extract noise residuals using SRM filters."""
        # Resize and convert to grayscale
        img = img.resize((self.size, self.size), Image.BILINEAR)
        img_gray = np.array(img.convert('L')).astype(np.float32)
        
        residuals = []
        for kernel in self.kernels:
            # Apply high-pass filter
            residual = cv2.filter2D(img_gray, -1, kernel)
            
            # Truncate extreme values
            if self.truncate > 0:
                residual = np.clip(residual, -self.truncate, self.truncate)
            
            # Normalize to [-1, 1] or [0, 1]
            if self.normalize:
                residual = residual / (self.truncate if self.truncate > 0 else 1.0)
            
            residuals.append(residual)
        
        # Stack residuals [num_kernels, H, W]
        tensor = torch.from_numpy(np.stack(residuals, axis=0)).float()
        
        return tensor
    
    @staticmethod
    def get_srm_conv_layer(in_channels: int = 3) -> nn.Conv2d:
        """Create a convolutional layer initialized with SRM filters.
        
        This can be used as the first layer of a noise branch network.
        """
        kernels = list(NoiseTransform.SRM_KERNELS.values())
        num_kernels = len(kernels)
        
        # Create conv layer: in_channels -> num_kernels * in_channels
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_kernels * in_channels,
            kernel_size=5,
            padding=2,
            bias=False,
        )
        
        # Initialize with SRM kernels
        with torch.no_grad():
            weight = torch.zeros(num_kernels * in_channels, in_channels, 5, 5)
            for c in range(in_channels):
                for k, kernel in enumerate(kernels):
                    idx = c * num_kernels + k
                    weight[idx, c] = torch.from_numpy(kernel)
            conv.weight.copy_(weight)
        
        # Freeze the SRM layer weights
        conv.weight.requires_grad = False
        
        return conv


class MultiDomainTransform:
    """Combined transform that extracts all three domains simultaneously."""
    
    def __init__(
        self,
        size: int = 256,
        rgb_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        rgb_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.rgb_transform = RGBTransform(size, rgb_mean, rgb_std)
        self.freq_transform = FrequencyTransform(size)
        self.noise_transform = NoiseTransform(size)
    
    def __call__(self, img: Image.Image) -> Dict[str, torch.Tensor]:
        """Extract RGB, frequency, and noise features from image."""
        return {
            'rgb': self.rgb_transform(img),
            'freq': self.freq_transform(img),
            'noise': self.noise_transform(img),
        }


class AugmentationTransform:
    """Data augmentation for training robustness."""
    
    def __init__(
        self,
        size: int = 256,
        p_flip: float = 0.5,
        p_rotate: float = 0.3,
        p_color: float = 0.3,
        p_blur: float = 0.2,
        p_jpeg: float = 0.3,
    ):
        self.size = size
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_color = p_color
        self.p_blur = p_blur
        self.p_jpeg = p_jpeg
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random augmentations to image."""
        # Random horizontal flip
        if np.random.random() < self.p_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random rotation (90, 180, 270 degrees)
        if np.random.random() < self.p_rotate:
            angle = np.random.choice([90, 180, 270])
            img = img.rotate(angle)
        
        # Random color jitter (simplified)
        if np.random.random() < self.p_color:
            img = self._color_jitter(img)
        
        # Random Gaussian blur
        if np.random.random() < self.p_blur:
            img = self._gaussian_blur(img)
        
        # Random JPEG compression (important for robustness)
        if np.random.random() < self.p_jpeg:
            img = self._jpeg_compress(img)
        
        return img
    
    def _color_jitter(self, img: Image.Image, factor: float = 0.1) -> Image.Image:
        """Apply random brightness/contrast adjustment."""
        from PIL import ImageEnhance
        
        # Random brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.0 + np.random.uniform(-factor, factor))
        
        # Random contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.0 + np.random.uniform(-factor, factor))
        
        return img
    
    def _gaussian_blur(self, img: Image.Image, max_sigma: float = 1.5) -> Image.Image:
        """Apply Gaussian blur."""
        from PIL import ImageFilter
        sigma = np.random.uniform(0.5, max_sigma)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))
    
    def _jpeg_compress(self, img: Image.Image, quality_range: Tuple[int, int] = (70, 95)) -> Image.Image:
        """Apply JPEG compression."""
        import io
        quality = np.random.randint(*quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
