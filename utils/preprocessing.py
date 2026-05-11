"""
preprocessing.py
────────────────
Advanced preprocessing for emotion detection model.

Techniques:
  - Histogram equalization (CLAHE)
  - Gaussian blur denoising
  - Morphological operations
  - Contrast stretching
"""

import numpy as np
import cv2


def clahe_equalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Better than standard histogram equalization for facial features.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def denoise(image, h=10, template_window_size=7, search_window_size=21):
    """
    Non-Local Means Denoising.
    Reduces noise while preserving facial features.
    """
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(
            image, h=h, 
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
    else:
        return cv2.fastNlMeansDenoising(
            image, h=h,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )


def morphological_cleanup(image, kernel_size=3):
    """
    Morphological operations to clean up image.
    Removes small artifacts and fills small holes.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Open: remove small noise
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # Close: fill small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed


def contrast_stretching(image, lower_percentile=2, upper_percentile=98):
    """
    Stretch contrast to use full dynamic range.
    Enhances subtle facial features.
    """
    lower = np.percentile(image, lower_percentile)
    upper = np.percentile(image, upper_percentile)
    
    stretched = np.clip((image - lower) / (upper - lower + 1e-5) * 255, 0, 255)
    return stretched.astype(np.uint8)


def advanced_preprocessing(image, mode='full'):
    """
    Apply comprehensive preprocessing pipeline.
    
    Modes:
      - 'full': All techniques (slowest, best quality)
      - 'fast': CLAHE + denoise (balanced)
      - 'basic': CLAHE only (fastest)
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mode == 'full':
        # 1. Denoise first (preserves edges)
        image = denoise(image, h=8)
        
        # 2. CLAHE for contrast enhancement
        image = clahe_equalization(image, clip_limit=3.0)
        
        # 3. Morphological cleanup
        image = morphological_cleanup(image, kernel_size=3)
        
        # 4. Contrast stretching
        image = contrast_stretching(image)
        
        # 5. Gaussian blur for smoothing
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
    elif mode == 'fast':
        # Balanced speed/quality
        image = denoise(image, h=6)
        image = clahe_equalization(image, clip_limit=2.0)
        image = contrast_stretching(image, lower_percentile=1, upper_percentile=99)
        
    else:  # 'basic'
        image = clahe_equalization(image, clip_limit=2.0)
        image = contrast_stretching(image)
    
    return image.astype(np.float32)


def preprocess_batch(images, mode='fast'):
    """
    Preprocess a batch of images efficiently.
    """
    return np.array([advanced_preprocessing(img, mode=mode) for img in images])


# Test the preprocessing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create sample face-like image
    sample = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(sample, cmap='gray')
    axes[0].set_title('Original')
    
    axes[1].imshow(advanced_preprocessing(sample, 'basic'), cmap='gray')
    axes[1].set_title('Basic')
    
    axes[2].imshow(advanced_preprocessing(sample, 'fast'), cmap='gray')
    axes[2].set_title('Fast')
    
    axes[3].imshow(advanced_preprocessing(sample, 'full'), cmap='gray')
    axes[3].set_title('Full')
    
    plt.tight_layout()
    plt.savefig('preprocessing_demo.png')
    print("✅ Preprocessing demo saved")
