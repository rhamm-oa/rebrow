import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

class EyebrowEffects:
    """Class for applying special effects to eyebrows like alpha matting and recoloring"""
    
    def __init__(self):
        pass
    
    def create_trimap(self, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Create a trimap from a binary mask
        
        Args:
            mask: Binary mask
            kernel_size: Size of the kernel for erosion and dilation
            
        Returns:
            trimap: Trimap with values 0 (background), 1 (unknown), 2 (foreground)
        """
        if mask is None or np.sum(mask) == 0:
            return None # type: ignore
            
        # Create a trimap
        trimap = np.zeros_like(mask, dtype=np.uint8)
        
        # Erode the mask to get definite foreground
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        
        # Dilate the mask to get the unknown region
        dilated = cv2.dilate(mask, kernel, iterations=1)
        
        # Set trimap values
        trimap[eroded > 0] = 2  # Definite foreground
        trimap[dilated > 0] = 1  # Unknown region
        
        return trimap
    
    def apply_alpha_matting(self, image: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Apply alpha matting to get a more precise separation of eyebrow hairs
        
        Args:
            image: Input image
            mask: Binary mask of the eyebrow region
            bbox: Bounding box of the eyebrow (x_min, y_min, x_max, y_max)
            
        Returns:
            alpha_matte: Alpha matte of the eyebrow region
            cropped_alpha: Cropped alpha matte to the bounding box
        """
        if mask is None or bbox is None or np.sum(mask) == 0:
            return None, None
            
        # Extract the region of interest
        x_min, y_min, x_max, y_max = bbox
        roi = image[y_min:y_max, x_min:x_max]
        mask_roi = mask[y_min:y_max, x_min:x_max]
        
        # Create a trimap
        trimap = self.create_trimap(mask_roi)
        if trimap is None:
            return None, None
        
        # Normalize to 0-1 range for alpha matte
        alpha_matte = np.zeros_like(mask_roi, dtype=np.float32)
        alpha_matte[trimap == 2] = 1.0  # Definite foreground
        
        # For unknown regions, calculate a gradient
        unknown_mask = (trimap == 1)
        if np.sum(unknown_mask) > 0:
            # Simple distance-based alpha calculation
            dist_bg = cv2.distanceTransform((trimap != 0).astype(np.uint8), cv2.DIST_L2, 3)
            dist_fg = cv2.distanceTransform((trimap != 2).astype(np.uint8), cv2.DIST_L2, 3)
            
            # Normalize distances
            alpha_unknown = dist_bg / (dist_bg + dist_fg + 1e-10)
            alpha_matte[unknown_mask] = alpha_unknown[unknown_mask]
        
        # Create full-size alpha matte
        full_alpha = np.zeros_like(mask, dtype=np.float32)
        full_alpha[y_min:y_max, x_min:x_max] = alpha_matte
        
        return full_alpha, alpha_matte
    
    def recolor_eyebrow(self, image: np.ndarray, alpha_matte: np.ndarray, bbox: Tuple[int, int, int, int], target_color: Tuple[int, int, int]) -> np.ndarray:
        """Recolor the eyebrow region using the alpha matte
        
        Args:
            image: Input image
            alpha_matte: Alpha matte of the eyebrow region
            bbox: Bounding box of the eyebrow (x_min, y_min, x_max, y_max)
            target_color: Target color for recoloring (B, G, R)
            
        Returns:
            recolored_image: Image with recolored eyebrow
        """
        if alpha_matte is None or bbox is None:
            return image.copy()
            
        # Extract the region of interest
        x_min, y_min, x_max, y_max = bbox
        roi = image[y_min:y_max, x_min:x_max].copy()
        alpha_roi = alpha_matte[y_min:y_max, x_min:x_max]
        
        # Create a color layer with the target color
        color_layer = np.ones_like(roi) * np.array(target_color, dtype=np.uint8)
        
        # Blend the original image with the color layer using the alpha matte
        for c in range(3):  # For each color channel
            roi[:, :, c] = roi[:, :, c] * (1 - alpha_roi) + color_layer[:, :, c] * alpha_roi
        
        # Create the output image
        recolored = image.copy()
        recolored[y_min:y_max, x_min:x_max] = roi
        
        return recolored
    
    def combine_recolored_eyebrows(self, image: np.ndarray, left_recolored: np.ndarray, right_recolored: np.ndarray, 
                                  left_alpha: np.ndarray, right_alpha: np.ndarray,
                                  left_bbox: Tuple[int, int, int, int], right_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Combine two recolored eyebrows into a single image
        
        Args:
            image: Original image
            left_recolored: Left recolored eyebrow
            right_recolored: Right recolored eyebrow
            left_alpha: Left alpha matte
            right_alpha: Right alpha matte
            left_bbox: Left eyebrow bounding box
            right_bbox: Right eyebrow bounding box
            
        Returns:
            combined: Combined image with both eyebrows recolored
        """
        if left_recolored is None or right_recolored is None:
            return image.copy()
            
        combined = image.copy()
        
        # Apply left eyebrow
        if left_bbox and left_alpha is not None:
            lx_min, ly_min, lx_max, ly_max = left_bbox
            left_region = left_recolored[ly_min:ly_max, lx_min:lx_max]
            left_alpha_roi = left_alpha[ly_min:ly_max, lx_min:lx_max]
            
            for c in range(3):
                combined[ly_min:ly_max, lx_min:lx_max, c] = (
                    combined[ly_min:ly_max, lx_min:lx_max, c] * (1 - left_alpha_roi) + 
                    left_region[:, :, c] * left_alpha_roi
                )
        
        # Apply right eyebrow
        if right_bbox and right_alpha is not None:
            rx_min, ry_min, rx_max, ry_max = right_bbox
            right_region = right_recolored[ry_min:ry_max, rx_min:rx_max]
            right_alpha_roi = right_alpha[ry_min:ry_max, rx_min:rx_max]
            
            for c in range(3):
                combined[ry_min:ry_max, rx_min:rx_max, c] = (
                    combined[ry_min:ry_max, rx_min:rx_max, c] * (1 - right_alpha_roi) + 
                    right_region[:, :, c] * right_alpha_roi
                )
        
        return combined
    
    def generate_color_variations(self, base_color: Tuple[int, int, int], num_variations: int = 5) -> List[Tuple[int, int, int]]:
        """Generate color variations based on a base color
        
        Args:
            base_color: Base color (B, G, R)
            num_variations: Number of variations to generate
            
        Returns:
            variations: List of color variations
        """
        variations = []
        
        # Convert to HSV for better color manipulation
        base_color_np = np.array([[base_color]], dtype=np.uint8)
        hsv_color = cv2.cvtColor(base_color_np, cv2.COLOR_BGR2HSV)[0][0]
        
        h, s, v = hsv_color
        
        # Generate variations by modifying hue and saturation
        for i in range(num_variations):
            # Modify hue (wrap around 180 for OpenCV's HSV)
            new_h = (h + (i * 15)) % 180
            
            # Alternate between increasing and decreasing saturation
            if i % 2 == 0:
                new_s = min(255, s + 30)
            else:
                new_s = max(0, s - 30)
            
            # Create new HSV color
            new_hsv = np.array([[[new_h, new_s, v]]], dtype=np.uint8)
            
            # Convert back to BGR
            new_bgr = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)[0][0]
            variations.append((int(new_bgr[0]), int(new_bgr[1]), int(new_bgr[2])))
        
        return variations
