import cv2
import numpy as np

class EyebrowRecoloring:
    """
    Class for recoloring eyebrows in images using segmentation masks.
    """
    
    def __init__(self):
        """Initialize the eyebrow recoloring module."""
        pass
        
    def apply_color_to_eyebrows(self, image, eyebrow_mask, target_color_rgb, 
                               preserve_highlights=True, preserve_texture=True, 
                               opacity=0.8):
        """
        Apply a target RGB color to eyebrows with advanced hair strand isolation.
        Uses multiple techniques to extract only the hair strands and not the skin.
        
        Args:
            image: Input image (BGR format)
            eyebrow_mask: Binary mask of eyebrow region
            target_color_rgb: Target RGB color as (r, g, b) tuple
            preserve_highlights: Whether to preserve highlight areas
            preserve_texture: Whether to preserve eyebrow texture
            opacity: Opacity of color application (0.0 to 1.0)
            
        Returns:
            Recolored image
        """
        # Check if mask is valid
        if eyebrow_mask is None or np.max(eyebrow_mask) == 0:
            return image.copy()
        
        # Make sure mask is binary - fix for Pylance type checking
        binary_mask = np.where(eyebrow_mask > 0, 1, 0).astype(np.uint8)
        
        # Convert target RGB to BGR for OpenCV
        target_color_bgr = (target_color_rgb[2], target_color_rgb[1], target_color_rgb[0])
        
        # Create a copy of the image
        result = image.copy()
        
        # Extract the eyebrow region
        eyebrow_region = cv2.bitwise_and(image, image, mask=binary_mask)
        
        # ADVANCED HAIR ISOLATION
        # Convert to different color spaces for better hair detection
        gray_region = cv2.cvtColor(eyebrow_region, cv2.COLOR_BGR2GRAY)
        hsv_region = cv2.cvtColor(eyebrow_region, cv2.COLOR_BGR2HSV)
        lab_region = cv2.cvtColor(eyebrow_region, cv2.COLOR_BGR2Lab)
        
        # 1. MULTI-CHANNEL HAIR DETECTION
        # Extract relevant channels
        v_channel = hsv_region[:,:,2]  # Value channel (darkness)
        a_channel = lab_region[:,:,1]  # a channel (green-red)
        
        # 2. ADAPTIVE THRESHOLDING - more aggressive for hair strands
        block_size = 7  # Smaller block size for finer details
        c_value = 3     # Higher constant for more aggressive thresholding
        
        # Apply adaptive threshold to get hair strands from value channel
        v_hair_mask = cv2.adaptiveThreshold(
            v_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, c_value
        )
        
        # 3. EDGE DETECTION - capture hair edges
        # Use Canny edge detection with tighter thresholds
        edges = cv2.Canny(gray_region, 30, 100)
        
        # 4. COLOR-BASED SEGMENTATION
        # Use a-channel to help distinguish hair from skin (works well for eyebrows)
        _, a_thresh = cv2.threshold(a_channel, 128, 255, cv2.THRESH_BINARY)
        
        # 5. COMBINE MULTIPLE TECHNIQUES
        # Combine all masks
        combined_mask = cv2.bitwise_or(v_hair_mask, edges)
        combined_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=binary_mask)
        
        # 6. CLEAN UP with morphological operations
        # Create kernels for different operations
        small_kernel = np.ones((2,2), np.uint8)
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
        
        # Clean up noise while preserving hair-like structures
        hair_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, small_kernel)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, line_kernel)
        
        # 7. HIGHLIGHT PRESERVATION
        if preserve_highlights:
            # Create highlight mask (very bright areas)
            highlight_threshold = 220 if np.max(v_channel) > 230 else 200 # type: ignore
            highlight_mask = np.where(v_channel > highlight_threshold, 255, 0).astype(np.uint8) # type: ignore
            
            # Remove highlights from the hair mask
            hair_mask = cv2.bitwise_and(hair_mask, cv2.bitwise_not(highlight_mask))
        
        # Convert hair_mask to binary - fix for Pylance type checking
        hair_mask = np.where(hair_mask > 0, 255, 0).astype(np.uint8) # type: ignore
        
        # Create a color overlay image
        color_overlay = np.zeros_like(image)
        color_overlay[:] = target_color_bgr
        
        # ENHANCED TEXTURE PRESERVATION
        if preserve_texture:
            # Convert to grayscale for texture
            gray = cv2.cvtColor(eyebrow_region, cv2.COLOR_BGR2GRAY)
            
            # Create a texture mask with enhanced contrast for more visible texture
            # Normalize and apply gamma correction to enhance texture visibility
            texture_mask = gray.astype(float) / 255.0
            
            # Apply gamma correction to enhance texture contrast
            gamma = 0.7  # Value < 1 increases contrast in darker regions
            texture_mask = np.power(texture_mask, gamma)
            
            # For blonde colors, enhance the texture effect
            is_blonde = (target_color_rgb[0] > 200 and target_color_rgb[1] > 150)
            if is_blonde:
                # Increase texture contrast for blonde colors
                texture_mask = np.power(texture_mask, 0.8)  # Further enhance contrast
                
                # Boost the color intensity for blonde
                color_boost = 1.2  # Boost factor
                boosted_color = np.clip(np.array(target_color_bgr) * color_boost, 0, 255).astype(np.uint8)
                color_overlay[:] = boosted_color
            
            # Apply the enhanced texture to the color overlay
            for c in range(3):
                color_overlay[:,:,c] = (color_overlay[:,:,c] * texture_mask).astype(np.uint8)
        
        # Apply the color to the hair mask region
        colored_eyebrow = np.zeros_like(image)
        colored_eyebrow = cv2.bitwise_and(color_overlay, color_overlay, mask=hair_mask)
        
        # IMPROVED BLENDING
        # For blonde colors, use a stronger opacity to make them more noticeable
        effective_opacity = opacity
        if target_color_rgb[0] > 200 and target_color_rgb[1] > 150:  # Is blonde
            effective_opacity = min(opacity * 1.3, 1.0)  # Increase opacity for blonde, but cap at 1.0
        
        # Blend the colored eyebrow with the original image
        result = cv2.addWeighted(
            result, 1.0, 
            colored_eyebrow, effective_opacity, 
            0
        )
        
        return result
        
    def apply_lab_color_to_eyebrows(self, image, eyebrow_mask, target_color_lab, 
                                   preserve_highlights=True, preserve_texture=True, 
                                   opacity=0.8):
        """
        Apply a target LAB color to eyebrows.
        
        Args:
            image: Input image (BGR format)
            eyebrow_mask: Binary mask of eyebrow region
            target_color_lab: Target LAB color as (L, a, b) tuple
            preserve_highlights: Whether to preserve highlight areas
            preserve_texture: Whether to preserve eyebrow texture
            opacity: Opacity of color application (0.0 to 1.0)
            
        Returns:
            Recolored image
        """
        # Convert LAB color to RGB
        lab_array = np.array([[target_color_lab]], dtype=np.float32)
        bgr_array = cv2.cvtColor(lab_array, cv2.COLOR_Lab2BGR)
        rgb_values = (int(bgr_array[0, 0, 2]), int(bgr_array[0, 0, 1]), int(bgr_array[0, 0, 0]))
        
        # Use the RGB function
        return self.apply_color_to_eyebrows(image, eyebrow_mask, rgb_values, 
                                          preserve_highlights, preserve_texture, opacity)
    
    def recolor_both_eyebrows(self, image, left_mask, right_mask, target_color_rgb,
                             preserve_highlights=True, preserve_texture=True, 
                             opacity=0.8):
        """
        Recolor both eyebrows with the same target color.
        
        Args:
            image: Input image (BGR format)
            left_mask: Binary mask for the left eyebrow
            right_mask: Binary mask for the right eyebrow
            target_color_rgb: Target color in RGB format (tuple of 3 values 0-255)
            preserve_highlights: Whether to preserve highlights in the eyebrows
            preserve_texture: Whether to preserve the texture of the eyebrows
            opacity: Opacity of the color application (0.0-1.0)
            
        Returns:
            recolored_image: Image with both eyebrows recolored
        """
        # Combine both masks
        combined_mask = np.zeros_like(left_mask)
        if left_mask is not None:
            combined_mask = cv2.bitwise_or(combined_mask, left_mask)
        if right_mask is not None:
            combined_mask = cv2.bitwise_or(combined_mask, right_mask)
            
        # Apply color to the combined mask
        return self.apply_color_to_eyebrows(
            image, combined_mask, target_color_rgb,
            preserve_highlights, preserve_texture, opacity
        )
        
    def create_color_palette(self, n_colors=6):
        """
        Create a palette of predefined eyebrow colors.
        
        Args:
            n_colors: Number of colors to include in the palette
            
        Returns:
            colors: List of RGB colors [(r,g,b), ...]
        """
        # Common eyebrow colors (RGB format) - adjusted for better visual appearance
        palette = [
            (51, 25, 0),     # Dark brown
            (102, 51, 0),    # Medium brown
            (153, 102, 51),  # Light brown
            (32, 32, 32),    # Black
            (218, 165, 85),  # Blonde (more vibrant golden blonde)
            (240, 205, 140), # Light blonde (brighter, more noticeable)
            (120, 100, 80),  # Taupe
            (78, 52, 46),    # Auburn
            (93, 64, 55),    # Chestnut
            (150, 75, 0),    # Copper
            (65, 42, 42),    # Dark auburn
            (180, 150, 100)  # Tan (adjusted)
        ]
        
        # Return requested number of colors
        return palette[:n_colors]
