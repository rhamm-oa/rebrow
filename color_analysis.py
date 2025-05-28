import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import color
import matplotlib.pyplot as plt

# Import OpenCV's extra modules
try:
    # Try to import ximgproc module
    from cv2 import ximgproc
except ImportError:
    # If not available, create a placeholder that will raise a more helpful error
    class XimgprocPlaceholder:
        def __getattr__(self, name):
            raise ImportError("OpenCV's ximgproc module is not available. "
                             "Install opencv-contrib-python package to use this feature.")
    ximgproc = XimgprocPlaceholder()
from colormath.color_objects import sRGBColor, LabColor, LCHabColor, HSVColor
from colormath.color_conversions import convert_color

class ColorAnalysis:
    def __init__(self):
        pass
    
    def extract_dominant_colors(self, image, mask, n_colors=3, exclude_skin=True):
        """Extract dominant colors from the masked region using KMeans clustering
        
        Args:
            image: Input image (BGR format)
            mask: Binary mask for the region of interest
            n_colors: Number of dominant colors to extract
            exclude_skin: Whether to apply additional filtering to exclude skin tones
            
        Returns:
            colors: Array of dominant colors in RGB format
            percentages: Percentage of each color in the mask
        """
        if mask is None or np.sum(mask) == 0:
            return None, None
        
        # Apply mask to image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Simple approach: just use the masked pixels directly
        # Get all non-zero pixels (where the mask is applied)
        non_zero_pixels = masked_image[np.where(mask > 0)]
        
        # If we don't have enough pixels, return None
        if len(non_zero_pixels) == 0:
            return None, None
            
        # Reshape the image to be a list of pixels
        pixels = non_zero_pixels.reshape(-1, 3)
        
        # Convert from BGR to RGB
        pixels = pixels[:, ::-1]
        
        # Check if we have enough pixels
        if len(pixels) < n_colors:
            return None, None
        
        # Apply KMeans clustering with explicit n_init to avoid warning
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get the colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Get the percentage of each color
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / len(labels) * 100
        
        # Sort colors by percentage
        sorted_indices = np.argsort(percentages)[::-1]
        colors = colors[sorted_indices]
        percentages = percentages[sorted_indices]
        
        return colors, percentages
    
    def create_color_palette(self, colors, percentages):
        """Create a color palette image showing the dominant colors"""
        if colors is None or percentages is None:
            return None
        
        # Create a 100x300 image to display the colors
        palette = np.zeros((100, 300, 3), dtype=np.uint8)
        
        # Calculate the width of each color based on its percentage
        start_x = 0
        for i, (color, percentage) in enumerate(zip(colors, percentages)):
            width = int(300 * (percentage / 100))
            end_x = start_x + width
            
            # Draw the color
            palette[:, start_x:end_x] = color
            
            start_x = end_x
        
        # Convert to RGB for displaying
        palette = cv2.cvtColor(palette, cv2.COLOR_RGB2BGR)
        
        return palette
    
    def get_color_info(self, colors, percentages):
        """Get color information in RGB, HEX, LAB, LCH, and HSV format"""
        if colors is None or percentages is None:
            return []
        
        color_info = []
        for i, (color, percentage) in enumerate(zip(colors, percentages)):
            r, g, b = color
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            
            # Convert to LAB
            rgb_color = sRGBColor(r/255, g/255, b/255)
            lab_color = convert_color(rgb_color, LabColor)
            lch_color = convert_color(lab_color, LCHabColor)
            
            # Convert to HSV directly using OpenCV
            hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0] # type: ignore
            
            color_info.append({
                'rgb': (r, g, b),
                'hex': hex_color,
                'lab': (round(lab_color.lab_l, 1), round(lab_color.lab_a, 1), round(lab_color.lab_b, 1)), # type: ignore
                'lch': (round(lch_color.lch_l, 1), round(lch_color.lch_c, 1), round(lch_color.lch_h, 1)), # type: ignore
                'hsv': (int(hsv[0]), int(hsv[1]), int(hsv[2])),
                'percentage': f'{percentage:.1f}%'
            })
        
        return color_info
    
    def extract_eyebrow_hair_colors(self, image, mask, n_colors=3, darkness_threshold=100):
        """Extract dominant colors of eyebrow hairs by focusing on darker pixels in the masked region
        
        Args:
            image: Input image (BGR format)
            mask: Binary mask for the region of interest
            n_colors: Number of dominant colors to extract
            darkness_threshold: Maximum brightness value to consider as hair (0-255)
                               Lower values = darker pixels only
            
        Returns:
            colors: Array of dominant colors in RGB format
            percentages: Percentage of each color in the mask
        """
        if mask is None or np.sum(mask) == 0:
            return None, None
        
        # Apply mask to image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Convert to grayscale to get brightness
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        
        # Create a mask of darker pixels (likely to be hair rather than skin)
        dark_mask = np.zeros_like(mask)
        dark_mask[(gray < darkness_threshold) & (mask > 0)] = 255
        
        # If we don't have enough dark pixels, gradually increase the threshold
        while np.sum(dark_mask) < 100 and darkness_threshold < 180:
            darkness_threshold += 15
            dark_mask = np.zeros_like(mask)
            dark_mask[(gray < darkness_threshold) & (mask > 0)] = 255
        
        # Apply the dark mask to the original image
        dark_pixels = cv2.bitwise_and(image, image, mask=dark_mask)
        
        # Get all non-zero pixels (where the dark mask is applied)
        non_zero_pixels = dark_pixels[np.where(dark_mask > 0)]
        
        # If we don't have enough pixels, return None
        if len(non_zero_pixels) < n_colors * 5:  # Need at least a few pixels per cluster
            return None, None
            
        # Reshape the image to be a list of pixels
        pixels = non_zero_pixels.reshape(-1, 3)
        
        # Convert from BGR to RGB
        pixels = pixels[:, ::-1]
        
        # Apply KMeans clustering with explicit n_init to avoid warning
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get the colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Get the percentage of each color
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / len(labels) * 100
        
        # Sort colors by percentage
        sorted_indices = np.argsort(percentages)[::-1]
        colors = colors[sorted_indices]
        percentages = percentages[sorted_indices]
        
        return colors, percentages
        
    def extract_reliable_hair_colors(self, image, mask, n_colors=3):
        """
        A reliable method to extract multiple eyebrow hair colors using Otsu's thresholding
        and K-means clustering. This method is designed to consistently return multiple colors.
        
        Args:
            image: Input image (BGR format)
            mask: Binary mask for the region of interest
            n_colors: Number of dominant colors to extract
            
        Returns:
            colors: Array of dominant colors in RGB format
            percentages: Percentage of each color in the mask
            debug_images: Dictionary of debug images showing the processing steps
        """
        if mask is None or np.sum(mask) == 0:
            return None, None, None
            
        # Create debug images dictionary
        debug_images = {}
        
        # Apply mask to image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        debug_images['original_masked'] = masked_image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        
        # Use Otsu's thresholding to automatically find threshold for dark pixels
        # This is more adaptive than a fixed threshold
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_mask = cv2.bitwise_and(otsu_mask, mask)
        debug_images['otsu_mask'] = cv2.cvtColor(otsu_mask, cv2.COLOR_GRAY2BGR)
        
        # Clean up with morphological operations
        kernel = np.ones((2,2), np.uint8)
        refined_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        debug_images['refined_mask'] = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)
        
        # Extract hair pixels using the refined mask
        hair_pixels_img = cv2.bitwise_and(image, image, mask=refined_mask)
        debug_images['hair_pixels'] = hair_pixels_img.copy()
        
        # Get coordinates of non-zero pixels
        y_coords, x_coords = np.nonzero(refined_mask)
        
        # If we don't have enough pixels, try a less aggressive approach
        if len(y_coords) < n_colors * 10:
            # Try a fixed threshold instead
            dark_mask = np.zeros_like(mask)
            dark_condition = np.less(gray, 130)  # Less aggressive threshold
            mask_condition = np.greater(mask, 0)
            dark_pixels = np.logical_and(dark_condition, mask_condition)
            dark_mask[dark_pixels] = 255
            
            # Clean up
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
            debug_images['fallback_mask'] = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR)
            
            # Extract hair pixels
            hair_pixels_img = cv2.bitwise_and(image, image, mask=dark_mask)
            debug_images['fallback_pixels'] = hair_pixels_img.copy()
            
            # Get new coordinates
            y_coords, x_coords = np.nonzero(dark_mask)
            
            # If still not enough pixels, use the whole masked region
            if len(y_coords) < n_colors * 10:
                debug_images['using_whole_mask'] = masked_image.copy()
                y_coords, x_coords = np.nonzero(mask)
        
        # Extract BGR values at the non-zero coordinates
        hair_pixels = image[y_coords, x_coords]
        
        # Convert from BGR to RGB for display and clustering
        hair_pixels_rgb = hair_pixels[:, ::-1]
        
        # Apply KMeans clustering with forced n_colors
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(hair_pixels_rgb)
        
        # Get the colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Get the percentage of each color
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / len(labels) * 100
        
        # Sort colors by darkness (sum of RGB values, lower is darker)
        # This often gives better results for eyebrow hairs as the darkest colors are usually the actual hairs
        darkness = np.sum(colors, axis=1)
        sorted_indices = np.argsort(darkness)
        colors = colors[sorted_indices]
        percentages = percentages[sorted_indices]
        
        return colors, percentages, debug_images
        
    def extract_texture_based_hairs(self, image, mask, n_colors=3):
        """
        Extract eyebrow hairs using texture-based analysis and DBSCAN clustering.
        This method focuses on the textural differences between hair and skin.
        
        Args:
            image: Input image (BGR format)
            mask: Binary mask for the region of interest
            n_colors: Number of dominant colors to extract
            
        Returns:
            colors: Array of dominant colors in RGB format
            percentages: Percentage of each color in the mask
            debug_images: Dictionary of debug images showing the processing steps
        """
        if mask is None or np.sum(mask) == 0:
            return None, None, None
            
        # Create debug images dictionary
        debug_images = {}
        
        # Apply mask to image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        debug_images['original_masked'] = masked_image.copy()
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate local standard deviation (texture measure)
        kernel_size = 3  # Smaller kernel for fine hair details
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        # Ensure we don't have negative values before sqrt (can happen due to floating point precision)
        variance = np.maximum(0, sqr_mean - mean**2)  # Clamp to zero
        std_dev = np.sqrt(variance)
        
        # Normalize std_dev for visualization
        std_dev_norm = cv2.normalize(std_dev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        debug_images['texture_map'] = cv2.applyColorMap(std_dev_norm, cv2.COLORMAP_JET)
        
        # Hair has higher texture than skin - use adaptive threshold based on percentile
        # Higher percentile = more selective (only highest texture areas)
        texture_percentile = 70  # Adjust as needed
        texture_threshold = np.percentile(std_dev[mask > 0], texture_percentile)
        
        # Create texture-based mask
        texture_mask = np.zeros_like(mask)
        texture_mask[std_dev > texture_threshold] = 255
        texture_mask = cv2.bitwise_and(texture_mask, mask)  # Only within the eyebrow region
        debug_images['texture_mask'] = cv2.cvtColor(texture_mask, cv2.COLOR_GRAY2BGR)
        
        # Also use darkness as a secondary feature - hair is typically darker
        dark_mask = np.zeros_like(mask)
        dark_condition = np.less(gray, 100)  # Aggressive threshold
        mask_condition = np.greater(mask, 0)
        dark_pixels = np.logical_and(dark_condition, mask_condition)
        dark_mask[dark_pixels] = 255
        debug_images['dark_mask'] = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR)
        
        # Combine texture and darkness
        combined_mask = cv2.bitwise_or(texture_mask, dark_mask)
        combined_mask = cv2.bitwise_and(combined_mask, mask)  # Ensure we stay within eyebrow region
        debug_images['combined_mask'] = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        
        # Clean up with morphological operations
        kernel = np.ones((2,2), np.uint8)
        refined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        debug_images['refined_mask'] = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)
        
        # Extract hair pixels using the refined mask
        hair_pixels_img = cv2.bitwise_and(image, image, mask=refined_mask)
        debug_images['hair_pixels'] = hair_pixels_img.copy()
        
        # Get coordinates of non-zero pixels
        y_coords, x_coords = np.nonzero(refined_mask)
        
        # If we don't have enough pixels, try a less aggressive approach
        if len(y_coords) < n_colors * 10:
            # Fall back to Otsu's thresholding for automatic threshold detection
            _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            otsu_mask = cv2.bitwise_and(otsu_mask, mask)
            debug_images['otsu_mask'] = cv2.cvtColor(otsu_mask, cv2.COLOR_GRAY2BGR)
            
            # Clean up with morphological operations
            otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel)
            otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)
            
            # Extract hair pixels using Otsu mask
            hair_pixels_img = cv2.bitwise_and(image, image, mask=otsu_mask)
            debug_images['fallback_pixels'] = hair_pixels_img.copy()
            
            # Get new coordinates
            y_coords, x_coords = np.nonzero(otsu_mask)
            
            # If still not enough pixels, return None
            if len(y_coords) < n_colors * 5:
                return None, None, debug_images
        
        # Extract BGR values at the non-zero coordinates
        hair_pixels = image[y_coords, x_coords]
        
        # Convert from BGR to RGB for display and clustering
        hair_pixels_rgb = hair_pixels[:, ::-1]
        
        # Use DBSCAN for better clustering of hair colors (handles irregular distributions better)
        try:
            from sklearn.cluster import DBSCAN
            
            # Normalize pixel values for DBSCAN
            pixels_normalized = hair_pixels_rgb.astype(float) / 255.0
            
            # Apply DBSCAN clustering with more relaxed parameters
            # Higher eps = larger neighborhood, lower min_samples = smaller clusters allowed
            dbscan = DBSCAN(eps=0.1, min_samples=max(3, len(pixels_normalized) // 200))
            labels = dbscan.fit_predict(pixels_normalized)
            
            # Check if we have enough non-noise points
            non_noise_count = np.sum(labels != -1)
            if non_noise_count < len(pixels_normalized) * 0.1:  # If less than 10% of points are clustered
                # Try again with even more relaxed parameters
                dbscan = DBSCAN(eps=0.15, min_samples=max(2, len(pixels_normalized) // 300))
                labels = dbscan.fit_predict(pixels_normalized)
            
            # Filter out noise points (label -1)
            valid_pixels = hair_pixels_rgb[labels != -1]
            valid_labels = labels[labels != -1]
            
            # If we have valid clusters
            if len(valid_pixels) > 0 and len(np.unique(valid_labels)) >= 1:
                # Find the n largest clusters
                unique_labels, counts = np.unique(valid_labels, return_counts=True)
                sorted_indices = np.argsort(counts)[::-1]  # Sort by count, descending
                
                # Take up to n_colors largest clusters
                largest_clusters = sorted_indices[:min(n_colors, len(sorted_indices))]
                
                # Get the mean color of each cluster
                colors = []
                percentages = []
                
                for cluster_idx in largest_clusters:
                    cluster_label = unique_labels[cluster_idx]
                    cluster_pixels = valid_pixels[valid_labels == cluster_label]
                    cluster_color = np.mean(cluster_pixels, axis=0).astype(int)
                    colors.append(cluster_color)
                    percentages.append(counts[cluster_idx] / len(valid_labels) * 100)
                
                # If we have fewer than n_colors, try to add more by using KMeans on the largest cluster
                if len(colors) < n_colors and len(colors) > 0:
                    largest_cluster_pixels = valid_pixels[valid_labels == unique_labels[sorted_indices[0]]]
                    if len(largest_cluster_pixels) > n_colors * 10:  # Enough pixels to subdivide
                        sub_kmeans = KMeans(n_clusters=min(n_colors - len(colors) + 1, 3), random_state=42, n_init=10)
                        sub_kmeans.fit(largest_cluster_pixels)
                        sub_colors = sub_kmeans.cluster_centers_.astype(int)
                        sub_labels = sub_kmeans.labels_
                        sub_counts = np.bincount(sub_labels)
                        sub_percentages = sub_counts / len(sub_labels) * percentages[0] / 100  # Scale by parent cluster percentage
                        
                        # Add these colors, skipping the first one which is already represented
                        for i in range(1, len(sub_colors)):
                            colors.append(sub_colors[i])
                            percentages.append(sub_percentages[i] * 100)
                
                # Convert to numpy arrays
                colors = np.array(colors)
                percentages = np.array(percentages)
                
                # Sort by percentages again after possible additions
                if len(colors) > 1:
                    sorted_indices = np.argsort(percentages)[::-1]
                    colors = colors[sorted_indices]
                    percentages = percentages[sorted_indices]
                
                return colors, percentages, debug_images
            
        except (ImportError, ValueError) as e:
            # Fall back to KMeans if DBSCAN fails or isn't available
            debug_images['clustering_error'] = np.zeros((50, 200, 3), dtype=np.uint8)
            cv2.putText(debug_images['clustering_error'], f"DBSCAN error: {str(e)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Fallback to KMeans clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(hair_pixels_rgb)
        
        # Get the colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Get the percentage of each color
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / len(labels) * 100
        
        # Sort colors by percentage
        sorted_indices = np.argsort(percentages)[::-1]
        colors = colors[sorted_indices]
        percentages = percentages[sorted_indices]
        
        return colors, percentages, debug_images
        
    def extract_eyebrow_hairs_only(self, image, mask, n_colors=3):
        """
        Extract ONLY eyebrow hairs using aggressive filtering to exclude skin tones.
        This method is specifically designed for eyebrows where we want to focus
        exclusively on the hair strands and completely ignore surrounding skin.
        
        Args:
            image: Input image (BGR format)
            mask: Binary mask for the region of interest
            n_colors: Number of dominant colors to extract
            
        Returns:
            colors: Array of dominant colors in RGB format
            percentages: Percentage of each color in the mask
            debug_images: Dictionary of debug images showing the processing steps
        """
        if mask is None or np.sum(mask) == 0:
            return None, None, None
            
        # Create debug images dictionary
        debug_images = {}
        
        # Apply mask to image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        debug_images['original_masked'] = masked_image.copy()
        
        # Convert to different color spaces
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        
        # 1. Create a very aggressive darkness threshold to isolate only the darkest pixels
        # This is based on the observation that eyebrow hairs are significantly darker than skin
        dark_mask = np.zeros_like(mask)
        # Use np.less for proper array comparison
        dark_condition = np.less(gray, 100)  # More aggressive threshold
        mask_condition = np.greater(mask, 0)
        dark_pixels = np.logical_and(dark_condition, mask_condition)
        dark_mask[dark_pixels] = 255
        debug_images['dark_mask'] = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR)
        
        # 2. Apply edge detection to find hair strands
        edges = cv2.Canny(masked_image, 50, 150)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        debug_images['edges'] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 3. Use color information in HSV space to exclude skin tones
        h, s, v = cv2.split(hsv)
        
        # Create a saturation mask - eyebrow hairs often have higher saturation than skin
        # Use np arrays for bounds to satisfy type checking
        lower_s = np.array([30], dtype=np.uint8)  # Minimum saturation
        sat_mask = np.zeros_like(mask)
        sat_condition = np.greater(s, lower_s[0])
        sat_mask[np.logical_and(sat_condition, mask_condition)] = 255
        debug_images['saturation_mask'] = cv2.cvtColor(sat_mask, cv2.COLOR_GRAY2BGR)
        
        # 4. Combine dark mask with edge information
        combined_mask = cv2.bitwise_or(dark_mask, edges)
        combined_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=mask)
        debug_images['combined_mask'] = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        
        # 5. Apply morphological operations to clean up the mask
        kernel = np.ones((2,2), np.uint8)
        hair_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
        debug_images['hair_mask'] = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR)
        
        # 6. Get the final hair pixels
        hair_pixels = cv2.bitwise_and(image, image, mask=hair_mask)
        debug_images['hair_pixels'] = hair_pixels.copy()
        
        # If we don't have enough pixels, try a less aggressive approach
        non_zero_indices = np.nonzero(hair_mask)
        if len(non_zero_indices[0]) < n_colors * 5:
            # Fall back to a slightly less aggressive threshold
            dark_mask = np.zeros_like(mask)
            less_dark_condition = np.less(gray, 120)
            dark_mask[np.logical_and(less_dark_condition, mask_condition)] = 255
            hair_pixels = cv2.bitwise_and(image, image, mask=dark_mask)
            debug_images['fallback_mask'] = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR)
            non_zero_indices = np.nonzero(dark_mask)
            
            # If still not enough pixels, return None
            if len(non_zero_indices[0]) < n_colors * 5:
                return None, None, debug_images
        
        # Extract pixels at the non-zero indices
        pixels = hair_pixels[non_zero_indices[0], non_zero_indices[1]]
        
        # Convert from BGR to RGB for display
        pixels_rgb = pixels[:, ::-1]
        
        # Apply KMeans clustering with explicit n_init to avoid warning
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels_rgb)
        
        # Get the colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Get the percentage of each color
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / len(labels) * 100
        
        # Sort colors by percentage
        sorted_indices = np.argsort(percentages)[::-1]
        colors = colors[sorted_indices]
        percentages = percentages[sorted_indices]
        
        return colors, percentages, debug_images
        
    def extract_hair_with_matting(self, image, mask, n_colors=3):
        """Extract eyebrow hairs using advanced matting techniques to preserve fine details
        
        Args:
            image: Input image (BGR format)
            mask: Binary mask for the region of interest
            n_colors: Number of dominant colors to extract
            
        Returns:
            colors: Array of dominant colors in RGB format
            percentages: Percentage of each color in the mask
            debug_images: Dictionary of debug images showing the processing steps
        """
        if mask is None or np.sum(mask) == 0:
            return None, None, None
        
        # Create debug images dictionary
        debug_images = {}
        
        # Apply mask to image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        debug_images['masked_original'] = masked_image.copy()
        
        # Convert to different color spaces
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        
        # 1. Apply bilateral filter to preserve edges while smoothing
        bilateral = cv2.bilateralFilter(masked_image, 9, 75, 75)
        debug_images['bilateral'] = bilateral.copy()
        
        # 2. Use Laplacian for edge enhancement (focusing on hair strands)
        gray_bilateral = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_bilateral, cv2.CV_8U, ksize=5)
        debug_images['laplacian'] = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        
        # 3. Apply adaptive thresholding with a small block size to capture fine details
        thresh = cv2.adaptiveThreshold(gray_bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 7, 2)
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
        debug_images['adaptive_threshold'] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        # 4. Create a hair mask using color information in HSV space
        # Extract channels
        h, s, v = cv2.split(hsv)
        
        # Create a mask for likely hair pixels (darker and less saturated than skin)
        hair_mask = np.zeros_like(mask)
        # Use proper comparison for NumPy arrays
        dark_pixels = np.where(v < 120, 255, 0).astype(np.uint8)
        valid_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        hair_mask = cv2.bitwise_and(dark_pixels, valid_mask)
        debug_images['value_mask'] = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR)
        
        # 5. Combine edge information with color information
        # Enhance edges in the original image
        edge_enhanced = cv2.addWeighted(masked_image, 0.7, cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR), 0.3, 0)
        debug_images['edge_enhanced'] = edge_enhanced.copy()
        
        # 6. Apply guided filter for edge-preserving smoothing (matting-like effect)
        try:
            guided_filter = cv2.ximgproc.guidedFilter(masked_image, masked_image, 3, 0.1) # type: ignore
        except (AttributeError, ImportError):
            # Fallback if ximgproc is not available
            guided_filter = bilateral.copy()  # Use bilateral filter as fallback
        debug_images['guided_filter'] = guided_filter.copy()
        
        # 7. Create final hair mask by combining techniques
        # Combine threshold with hair mask
        combined_mask = cv2.bitwise_or(thresh, hair_mask)
        
        # Clean up with morphological operations
        kernel = np.ones((2,2), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        debug_images['combined_mask'] = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        
        # 8. Apply the mask to the edge-enhanced image to get hair pixels
        hair_pixels = cv2.bitwise_and(edge_enhanced, edge_enhanced, mask=combined_mask)
        debug_images['hair_pixels'] = hair_pixels.copy()
        
        # 9. Create a matting-like result by blending
        # Create a 3-channel mask for blending
        mask_3ch = cv2.merge([combined_mask, combined_mask, combined_mask])
        # Normalize to 0-1 range for alpha blending
        alpha = mask_3ch.astype(float) / 255.0
        # Blend original with black background using the mask as alpha
        matting_result = (masked_image * alpha).astype(np.uint8)
        debug_images['matting_result'] = matting_result.copy()
        
        # Get all non-zero pixels from the matting result
        # Create a mask of pixels that have at least one channel > 10
        # Use np.greater to compare arrays properly
        greater_than_10 = np.greater(matting_result, 10)
        non_zero_mask = np.any(greater_than_10, axis=2)
        non_zero_pixels = matting_result[non_zero_mask]
        
        # If we don't have enough pixels, try a less aggressive approach
        if len(non_zero_pixels) < n_colors * 5:
            # Fall back to simpler dark pixel extraction
            dark_mask = np.zeros_like(mask)
            dark_mask[(gray < 130) & (mask > 0)] = 255
            hair_pixels = cv2.bitwise_and(image, image, mask=dark_mask)
            debug_images['fallback_mask'] = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR)
            non_zero_pixels = hair_pixels[np.where(dark_mask > 0)]
            
            # If still not enough pixels, return None
            if len(non_zero_pixels) < n_colors * 5:
                return None, None, debug_images
        
        # Reshape the pixels for clustering
        pixels = non_zero_pixels.reshape(-1, 3)
        
        # Convert from BGR to RGB for display
        pixels_rgb = pixels[:, ::-1]
        
        # Apply KMeans clustering with explicit n_init to avoid warning
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels_rgb)
        
        # Get the colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Get the percentage of each color
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / len(labels) * 100
        
        # Sort colors by percentage
        sorted_indices = np.argsort(percentages)[::-1]
        colors = colors[sorted_indices]
        percentages = percentages[sorted_indices]
        
        return colors, percentages, debug_images
        
    def extract_fine_grained_hair_colors(self, image, mask, n_colors=3):
        """Advanced method for extracting fine-grained eyebrow hair colors using multiple techniques
        
        Args:
            image: Input image (BGR format)
            mask: Binary mask for the region of interest
            n_colors: Number of dominant colors to extract
            
        Returns:
            colors: Array of dominant colors in RGB format
            percentages: Percentage of each color in the mask
            debug_images: Dictionary of debug images showing the processing steps
        """
        if mask is None or np.sum(mask) == 0:
            return None, None, None
        
        # Create debug images dictionary
        debug_images = {}
        
        # Apply mask to image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        debug_images['masked_original'] = masked_image.copy()
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        
        # 1. Apply edge detection to find hair strands
        edges = cv2.Canny(gray, 50, 150)
        debug_images['edges'] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 2. Apply adaptive thresholding to separate hairs from skin
        # Use a small block size to capture fine details
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        # Only keep thresholded areas within the original mask
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
        debug_images['adaptive_threshold'] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        # 3. Use color filtering in HSV space to isolate hair-like colors
        # Eyebrow hairs are typically darker and less saturated than skin
        h, s, v = cv2.split(hsv)
        # Low value (darkness) is characteristic of hairs
        dark_mask = cv2.inRange(v, 0, 120)  # type: ignore # Adjust the upper bound as needed
        # Low to medium saturation for natural hair colors
        sat_mask = cv2.inRange(s, 0, 100)   # type: ignore # Adjust as needed
        # Combine masks
        hsv_mask = cv2.bitwise_and(dark_mask, sat_mask)
        hsv_mask = cv2.bitwise_and(hsv_mask, mask)
        debug_images['hsv_filtering'] = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
        
        # 4. Combine edge detection with thresholding for better hair detection
        # Dilate edges slightly to connect broken hair strands
        kernel = np.ones((2,2), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        # Combine with threshold
        combined_mask = cv2.bitwise_or(dilated_edges, thresh)
        combined_mask = cv2.bitwise_and(combined_mask, mask)  # Keep within original mask
        debug_images['combined_mask'] = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        
        # 5. Final hair mask: Combine all techniques
        final_mask = cv2.bitwise_or(combined_mask, hsv_mask)
        # Clean up with morphological operations
        kernel = np.ones((2,2), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        debug_images['final_mask'] = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
        
        # Apply final mask to get hair pixels
        hair_pixels = cv2.bitwise_and(image, image, mask=final_mask)
        debug_images['hair_pixels'] = hair_pixels.copy()
        
        # Get all non-zero pixels (where the hair mask is applied)
        non_zero_pixels = hair_pixels[np.where(final_mask > 0)] # type: ignore
        
        # If we don't have enough pixels, try a less aggressive approach
        if len(non_zero_pixels) < n_colors * 5:
            # Fall back to simpler dark pixel extraction
            dark_mask = np.zeros_like(mask)
            dark_mask[(gray < 130) & (mask > 0)] = 255
            hair_pixels = cv2.bitwise_and(image, image, mask=dark_mask)
            debug_images['fallback_mask'] = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR)
            non_zero_pixels = hair_pixels[np.where(dark_mask > 0)]
            
            # If still not enough pixels, return None
            if len(non_zero_pixels) < n_colors * 5:
                return None, None, debug_images
        
        # Reshape the pixels for clustering
        pixels = non_zero_pixels.reshape(-1, 3)
        
        # Convert from BGR to RGB for display
        pixels_rgb = pixels[:, ::-1]
        
        # Apply KMeans clustering with explicit n_init to avoid warning
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels_rgb)
        
        # Get the colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Get the percentage of each color
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / len(labels) * 100
        
        # Sort colors by percentage
        sorted_indices = np.argsort(percentages)[::-1]
        colors = colors[sorted_indices]
        percentages = percentages[sorted_indices]
        
        return colors, percentages, debug_images
        
    def analyze_color_properties(self, colors):
        """Analyze properties of the colors (brightness, saturation, etc.)"""
        if colors is None:
            return None
        
        properties = []
        for color in colors:
            r, g, b = color
            
            # Convert to HSV for better analysis
            color_array = np.array([[color]], dtype=np.uint8)
            hsv = cv2.cvtColor(color_array, cv2.COLOR_RGB2HSV)[0][0]
            h, s, v = hsv
            
            # Calculate brightness (0-100)
            brightness = v / 255 * 100
            
            # Calculate saturation (0-100)
            saturation = s / 255 * 100
            
            # Determine if the color is warm or cool
            # Hue: 0-30 and 150-180 are warm, 31-149 are cool
            is_warm = (h <= 30) or (h >= 150)
            
            # Calculate color intensity
            intensity = np.mean([r, g, b])
            
            properties.append({
                'brightness': f'{brightness:.1f}%',
                'saturation': f'{saturation:.1f}%',
                'intensity': f'{intensity:.1f}/255',
                'tone': 'Warm' if is_warm else 'Cool'
            })
        
        return properties
