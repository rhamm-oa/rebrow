# pylint: disable=no-member
# pyright: reportOperatorIssue=false, reportArgumentType=false
import cv2
import plotly
import numpy as np
from sklearn.cluster import KMeans
from skimage import color
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
from colormath.color_objects import sRGBColor, LabColor, LCHabColor, HSVColor
from colormath.color_conversions import convert_color
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os

# Import metadata handler
try:
    from metadata_handler import MetadataHandler
    METADATA_AVAILABLE = True
except ImportError:
    METADATA_AVAILABLE = False

class ColorAnalysis:
    def __init__(self, metadata_csv_path=None):
        """
        Initialize ColorAnalysis with optional metadata support.
        """
        self.metadata_handler = None
        
        if METADATA_AVAILABLE and metadata_csv_path and os.path.exists(metadata_csv_path):
            try:
                self.metadata_handler = MetadataHandler(metadata_csv_path)
                print("✅ Metadata handler initialized successfully")
            except Exception as e:
                print(f"Could not initialize metadata handler: {e}")
                self.metadata_handler = None
    
    def extract_eyebrow_hair_pixels_only(self, image, mask, debug=True):
        """
        ORIGINAL method - keeping it exactly as it was working!
        ENHANCED method to extract ONLY eyebrow hair pixels, with improved black hair detection.
        Uses multiple techniques including HSV thresholding, edge detection, and enhanced LAB analysis.
        
        Args:
            image: Input image (BGR format)
            mask: Binary mask for the eyebrow region from Facer segmentation
            debug: Whether to return debug images
            
        Returns:
            hair_mask: Binary mask containing only hair pixels (LAB-based final result)
            debug_images: Dictionary of debug images showing the process
        """
        debug_images = {}
        
        if mask is None or np.sum(mask) == 0:
            return None, debug_images
        
        # Apply the eyebrow mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        debug_images['1_masked_original'] = cv2.cvtColor(masked_image.copy(), cv2.COLOR_BGR2RGB)
        
        # Convert to different color spaces
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
        
        debug_images['2_gray'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # --- METHOD 1: Enhanced HSV-based hair detection ---
        h, s, v = cv2.split(hsv)
        
        # Create hair mask based on low brightness (Value channel)
        hair_value_mask = np.zeros_like(mask)
        
        # Enhanced adaptive threshold for black hair
        masked_v = v[mask > 0]
        if len(masked_v) > 0:
            v_min = np.min(masked_v)
            v_10th = np.percentile(masked_v, 10)
            v_mean = np.mean(masked_v)
            
            # Detect if this is likely black hair
            if v_10th < 30 and v_min < 20:
                # Very dark hair - use aggressive threshold
                value_threshold = np.percentile(masked_v, 15)
                value_threshold = min(value_threshold, 40)
            elif v_mean < 60:
                # Dark hair
                value_threshold = np.percentile(masked_v, 20)
                value_threshold = min(value_threshold, 60)
            else:
                # Light hair
                value_threshold = np.percentile(masked_v, 30)
                value_threshold = min(value_threshold, 80)
        else:
            value_threshold = 50
            
        hair_value_mask[(v < value_threshold) & (mask > 0)] = 255
        debug_images['3_hsv_value_mask'] = cv2.cvtColor(hair_value_mask, cv2.COLOR_GRAY2RGB)
        debug_images['3_threshold_used'] = f"HSV Value threshold: {value_threshold}"
        
        # Additional saturation filtering
        hair_sat_mask = np.zeros_like(mask)
        hair_sat_mask[(s > 8) & (mask > 0)] = 255  # Very permissive for black hair
        debug_images['4_hsv_saturation_mask'] = cv2.cvtColor(hair_sat_mask, cv2.COLOR_GRAY2RGB)
        
        # Combine HSV masks
        hsv_hair_mask = cv2.bitwise_and(hair_value_mask, hair_sat_mask)
        debug_images['5_combined_hsv_mask'] = cv2.cvtColor(hsv_hair_mask, cv2.COLOR_GRAY2RGB)
        
        # --- METHOD 2: Edge detection for hair strands ---
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(bilateral, 20, 80)  # Lower thresholds for finer hair detection
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        debug_images['6_edge_detection'] = cv2.cvtColor(edges_dilated, cv2.COLOR_GRAY2RGB)
        
        # --- METHOD 3: Texture-based detection ---
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        gray_float = gray.astype(np.float32)
        mean_img = cv2.filter2D(gray_float, -1, kernel)
        sqr_mean_img = cv2.filter2D(gray_float * gray_float, -1, kernel)
        
        variance = np.maximum(0.0, sqr_mean_img - mean_img * mean_img)
        
        variance_norm = np.zeros_like(variance, dtype=np.uint8)
        cv2.normalize(variance, variance_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        texture_threshold = np.percentile(variance_norm[mask > 0], 60) if np.sum(mask) > 0 else 50
        texture_mask = np.zeros_like(mask)
        texture_mask[(variance_norm > texture_threshold) & (mask > 0)] = 255
        debug_images['7_texture_variance'] = cv2.cvtColor(variance_norm, cv2.COLOR_GRAY2RGB)
        debug_images['8_texture_mask'] = cv2.cvtColor(texture_mask, cv2.COLOR_GRAY2RGB)
        
        # --- METHOD 4: ENHANCED LAB color space analysis ---
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Enhanced adaptive LAB threshold for black hair detection
        def get_adaptive_lab_threshold(l_channel, mask):
            if np.sum(mask) == 0:
                return 50
            
            masked_l = l_channel[mask > 0]
            l_mean = np.mean(masked_l)
            l_min = np.min(masked_l)
            l_10th = np.percentile(masked_l, 10)
            l_25th = np.percentile(masked_l, 25)
            
            # Enhanced detection for very dark/black hair
            if l_10th < 25 and l_min < 15:
                # Definitely black hair - very aggressive
                threshold = np.percentile(masked_l, 8)  # Even more aggressive
                debug_images['9_hair_type'] = "Very dark/black hair detected"
            elif l_10th < 35 and l_25th < 45:
                # Dark brown hair
                threshold = np.percentile(masked_l, 15)
                debug_images['9_hair_type'] = "Dark brown hair detected"
            elif l_mean < 60:
                # Medium brown hair
                threshold = np.percentile(masked_l, 20)
                debug_images['9_hair_type'] = "Medium brown hair detected"
            else:
                # Light hair
                threshold = np.percentile(masked_l, 30)
                debug_images['9_hair_type'] = "Light hair detected"
            
            return min(threshold, 50)  # Cap at 50 for safety
        
        lab_threshold = get_adaptive_lab_threshold(l_channel, mask)
        lab_hair_mask = np.zeros_like(mask)
        lab_hair_mask[(l_channel < lab_threshold) & (mask > 0)] = 255
        debug_images['9_lab_lightness_mask'] = cv2.cvtColor(lab_hair_mask, cv2.COLOR_GRAY2RGB)
        debug_images['9_lab_threshold_used'] = f"LAB L threshold: {lab_threshold:.1f}"
        
        # --- ENHANCED COMBINATION STRATEGY ---
        # For very dark hair, rely primarily on LAB
        # For lighter hair, use combination approach
        if lab_threshold < 25:  # Very dark hair detected
            final_mask = lab_hair_mask.copy()
            debug_images['10_strategy'] = "Using LAB-only strategy for very dark hair"
        else:
            # Combine methods for lighter hair
            combined_mask = hsv_hair_mask.copy()
            combined_mask = cv2.bitwise_or(combined_mask, edges_dilated)
            combined_mask = cv2.bitwise_or(combined_mask, texture_mask)
            combined_mask = cv2.bitwise_and(combined_mask, lab_hair_mask)
            final_mask = combined_mask
            debug_images['10_strategy'] = "Using combined methods for lighter hair"
        
        # Ensure we stay within the original eyebrow mask
        final_mask = cv2.bitwise_and(final_mask, mask)
        debug_images['10_combined_before_morphology'] = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)
        
        # --- MORPHOLOGICAL OPERATIONS ---
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
        
        debug_images['11_final_hair_mask'] = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)
        
        # Create final visualization
        hair_pixels_img = cv2.bitwise_and(image, image, mask=final_mask)
        debug_images['12_detected_hair_pixels'] = cv2.cvtColor(hair_pixels_img, cv2.COLOR_BGR2RGB)
        
        # Enhanced fallback for insufficient pixels
        final_mask_sum = int(np.sum(final_mask))
        if final_mask_sum < 30:  # Lowered threshold
            # More aggressive fallback
            aggressive_threshold = np.percentile(l_channel[mask > 0], 5) if np.sum(mask) > 0 else 30
            fallback_mask = np.zeros_like(mask)
            fallback_mask[(l_channel < aggressive_threshold) & (mask > 0)] = 255
            
            fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_OPEN, kernel_open)
            fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_CLOSE, kernel_close)
            
            debug_images['13_fallback_mask'] = cv2.cvtColor(fallback_mask, cv2.COLOR_GRAY2RGB)
            debug_images['13_fallback_note'] = f"Used aggressive fallback: 5th percentile (threshold: {aggressive_threshold:.1f})"
            
            fallback_mask_sum = int(np.sum(fallback_mask))
            if fallback_mask_sum > final_mask_sum:
                final_mask = fallback_mask
                hair_pixels_img = cv2.bitwise_and(image, image, mask=final_mask)
                debug_images['12_detected_hair_pixels'] = cv2.cvtColor(hair_pixels_img, cv2.COLOR_BGR2RGB)
        
        # Final statistics
        final_pixels = int(np.sum(final_mask))
        debug_images['14_final_stats'] = f"Final threshold: {lab_threshold:.1f}\nPixels detected: {final_pixels}\nStrategy: Enhanced black hair detection"
        
        return final_mask, debug_images
    
    def apply_metadata_adjustments(self, hair_mask, image, mask, metadata):
        """
        OPTIONAL: Apply metadata-based adjustments to the hair mask.
        This is called ONLY if metadata is available and won't break existing functionality.
        """
        if not metadata or hair_mask is None:
            return hair_mask
        
        try:
            ethnicity_name = metadata.get('ethnicity_name', 'others')
            skin_cluster = metadata.get('skin_cluster', 3)
            
            # Only apply very light adjustments that won't break working cases
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Slight adjustment based on ethnicity (very conservative)
            if ethnicity_name in ['black_african_american', 'asian'] and skin_cluster >= 4:
                # For very dark hair + dark skin, slightly expand the mask
                expansion_threshold = np.percentile(l_channel[mask > 0], 8) if np.sum(mask) > 0 else 30
                expansion_mask = (l_channel < expansion_threshold) & (mask > 0)
                
                # Only add pixels, never remove
                enhanced_mask = cv2.bitwise_or(hair_mask, expansion_mask.astype(np.uint8) * 255)
                
                # Light morphological cleanup
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel)
                
                return enhanced_mask
            
            return hair_mask
            
        except Exception as e:
            print(f"Metadata adjustment failed, using original: {e}")
            return hair_mask
    
    def _robust_kmeans_clustering(self, hair_pixels_rgb, n_colors):
        """
        IMPROVED: Robust K-means clustering that ensures consistent cluster count.
        """
        # ALWAYS use the exact n_colors specified - no adaptive changes
        n_clusters = n_colors
        
        # Multiple K-means attempts with different initializations for robustness
        best_kmeans = None
        best_inertia = float('inf')
        
        # Try multiple random initializations
        for attempt in range(5):  # 5 attempts for robustness
            try:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=42 + attempt,  # Different seed each attempt
                    n_init=20,  # More initializations per attempt
                    algorithm='lloyd',  # Use Lloyd's algorithm (classic K-means)
                    max_iter=500,  # More iterations for convergence
                    tol=1e-6  # Tighter convergence tolerance
                )
                
                kmeans.fit(hair_pixels_rgb)
                
                # Keep the best result (lowest inertia)
                if kmeans.inertia_ < best_inertia:
                    best_inertia = kmeans.inertia_
                    best_kmeans = kmeans
                    
            except Exception as e:
                print(f"K-means attempt {attempt + 1} failed: {e}")
                continue
        
        if best_kmeans is None:
            raise Exception("All K-means attempts failed")
        
        # Extract results from best K-means
        colors = best_kmeans.cluster_centers_.astype(int)
        labels = best_kmeans.labels_
        
        # Calculate percentages
        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = (counts / len(labels)) * 100
        
        # Ensure we have exactly n_colors results
        if len(colors) < n_colors:
            # If we got fewer clusters than requested, duplicate the dominant cluster
            while len(colors) < n_colors:
                # Find the dominant cluster
                dominant_idx = np.argmax(percentages)
                dominant_color = colors[dominant_idx]
                
                # Add slight variation to create a new "cluster"
                variation = np.random.randint(-10, 11, size=3)
                new_color = np.clip(dominant_color + variation, 0, 255)
                
                colors = np.vstack([colors, new_color])
                percentages = np.append(percentages, 5.0)  # Small percentage for synthetic cluster
                
                # Renormalize percentages
                percentages = (percentages / np.sum(percentages)) * 100
        
        # Sort by percentage (descending)
        sorted_indices = np.argsort(percentages)[::-1]
        colors = colors[sorted_indices][:n_colors]  # Ensure exactly n_colors
        percentages = percentages[sorted_indices][:n_colors]  # Ensure exactly n_colors
        
        # Final renormalization to ensure percentages sum to 100%
        percentages = (percentages / np.sum(percentages)) * 100
        
        return colors, percentages
    
    def extract_colors_from_hair_mask(self, image, hair_mask, n_colors=3):
        """
        ENHANCED: Extract exactly n_colors dominant colors with consistent K-means clustering.
        """
        if hair_mask is None or np.sum(hair_mask) == 0:
            return None, None
        
        # Get hair pixels
        hair_pixels = image[hair_mask > 0]
        
        if len(hair_pixels) < 3:
            return None, None
        
        # Convert from BGR to RGB
        hair_pixels_rgb = hair_pixels[:, ::-1]
        
        # Determine if this is likely black hair by analyzing LAB values
        def is_likely_black_hair(pixels_rgb):
            sample_size = min(len(pixels_rgb), 100)
            sample_pixels = pixels_rgb[:sample_size]
            
            # Convert to LAB for analysis
            sample_bgr = sample_pixels[:, ::-1]  # Convert back to BGR for OpenCV
            sample_reshaped = sample_bgr.reshape(-1, 1, 3)
            sample_lab = cv2.cvtColor(sample_reshaped, cv2.COLOR_BGR2LAB).reshape(-1, 3)
            
            l_values = sample_lab[:, 0]
            l_mean = np.mean(l_values)
            l_10th = np.percentile(l_values, 10)
            
            return l_mean < 40 and l_10th < 25
        
        is_black_hair = is_likely_black_hair(hair_pixels_rgb)
        
        # STRATEGY SELECTION based on number of pixels
        if len(hair_pixels) >= 50:
            # STRATEGY 1: Robust K-means clustering (ALWAYS uses exact n_colors)
            try:
                colors, percentages = self._robust_kmeans_clustering(hair_pixels_rgb, n_colors)
                
                # Black hair validation and correction
                if is_black_hair:
                    colors = self._correct_black_hair_colors(colors)
                
                print(f"✅ K-means successful: {len(colors)} clusters (target: {n_colors})")
                
            except Exception as e:
                print(f"❌ K-means failed: {e}, using manual grouping fallback")
                # Fallback to manual grouping but still ensure n_colors
                colors, percentages = self._manual_grouping_with_fixed_k(hair_pixels_rgb, n_colors, is_black_hair)
        
        elif len(hair_pixels) >= 15:
            # STRATEGY 2: Manual grouping with fixed K
            colors, percentages = self._manual_grouping_with_fixed_k(hair_pixels_rgb, n_colors, is_black_hair)
            print(f"✅ Manual grouping: {len(colors)} clusters (target: {n_colors})")
        
        else:
            # STRATEGY 3: Few pixels - synthetic cluster generation
            colors, percentages = self._generate_synthetic_clusters(hair_pixels_rgb, n_colors, is_black_hair)
            print(f"✅ Synthetic clusters: {len(colors)} clusters (target: {n_colors})")
        
        # FINAL VALIDATION: Ensure exactly n_colors
        if len(colors) != n_colors or len(percentages) != n_colors:
            print(f"⚠️  Cluster count mismatch. Expected: {n_colors}, Got: {len(colors)}. Fixing...")
            colors, percentages = self._force_exact_cluster_count(colors, percentages, n_colors)
        
        # Final processing
        if not isinstance(colors, np.ndarray):
            colors = np.array(colors)
        if not isinstance(percentages, np.ndarray):
            percentages = np.array(percentages)
        
        # Final validation
        assert len(colors) == n_colors, f"Final colors count {len(colors)} != {n_colors}"
        assert len(percentages) == n_colors, f"Final percentages count {len(percentages)} != {n_colors}"
        assert abs(np.sum(percentages) - 100.0) < 0.1, f"Percentages don't sum to 100: {np.sum(percentages)}"
        
        return colors, percentages
    
    def _manual_grouping_with_fixed_k(self, hair_pixels_rgb, n_colors, is_black_hair):
        """
        Manual grouping that ensures exactly n_colors clusters.
        """
        # Convert to LAB for grouping
        hair_pixels_bgr = hair_pixels_rgb[:, ::-1]
        hair_pixels_reshaped = hair_pixels_bgr.reshape(-1, 1, 3)
        hair_lab = cv2.cvtColor(hair_pixels_reshaped, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        l_values = hair_lab[:, 0]
        
        # Create groups based on lightness percentiles
        percentiles = np.linspace(0, 100, n_colors + 1)
        
        groups = []
        group_percentages = []
        
        for i in range(n_colors):
            lower_percentile = percentiles[i]
            upper_percentile = percentiles[i + 1]
            
            lower_threshold = np.percentile(l_values, lower_percentile)
            upper_threshold = np.percentile(l_values, upper_percentile)
            
            if i == n_colors - 1:  # Last group includes the maximum
                mask = (l_values >= lower_threshold) & (l_values <= upper_threshold)
            else:
                mask = (l_values >= lower_threshold) & (l_values < upper_threshold)
            
            if np.sum(mask) >= 1:  # At least 1 pixel
                group_color = np.mean(hair_pixels_rgb[mask], axis=0).astype(int)
                if is_black_hair:
                    group_color = self._correct_black_hair_colors([group_color])[0]
                groups.append(group_color)
                group_percentages.append(np.sum(mask) / len(hair_pixels_rgb) * 100)
            else:
                # No pixels in this range - create synthetic color
                if i == 0:
                    # Darkest synthetic color
                    synthetic_color = np.array([20, 20, 25])
                elif i == n_colors - 1:
                    # Lightest synthetic color  
                    synthetic_color = np.array([60, 55, 50])
                else:
                    # Middle synthetic color
                    synthetic_color = np.array([40, 35, 35])
                
                groups.append(synthetic_color)
                group_percentages.append(100.0 / n_colors)  # Equal distribution
        
        # Ensure we have exactly n_colors
        while len(groups) < n_colors:
            # Duplicate the most common color with slight variation
            dominant_idx = np.argmax(group_percentages)
            dominant_color = groups[dominant_idx]
            variation = np.random.randint(-8, 9, size=3)
            new_color = np.clip(dominant_color + variation, 0, 255)
            groups.append(new_color)
            group_percentages.append(5.0)
        
        # Trim to exact count
        groups = groups[:n_colors]
        group_percentages = group_percentages[:n_colors]
        
        # Normalize percentages
        total = sum(group_percentages)
        if total > 0:
            group_percentages = [(p / total) * 100 for p in group_percentages]
        
        return np.array(groups), np.array(group_percentages)
    
    def _generate_synthetic_clusters(self, hair_pixels_rgb, n_colors, is_black_hair):
        """
        Generate synthetic clusters when there are very few pixels.
        """
        # Get the average color as base
        base_color = np.mean(hair_pixels_rgb, axis=0).astype(int)
        
        if is_black_hair:
            base_color = self._correct_black_hair_colors([base_color])[0]
        
        colors = []
        percentages = []
        
        # Create variations around the base color
        for i in range(n_colors):
            if i == 0:
                # First color is the base color (dominant)
                colors.append(base_color)
                percentages.append(60.0)
            else:
                # Create variations
                variation_factor = (i * 15) - 10  # -10, 5, 20, etc.
                varied_color = np.clip(base_color + variation_factor, 0, 255)
                colors.append(varied_color)
                percentages.append(40.0 / (n_colors - 1))  # Split remaining 40%
        
        # Normalize percentages to sum to 100
        total = sum(percentages)
        percentages = [(p / total) * 100 for p in percentages]
        
        return np.array(colors), np.array(percentages)
    
    def _force_exact_cluster_count(self, colors, percentages, target_count):
        """
        Force the result to have exactly target_count clusters.
        """
        current_count = len(colors)
        
        if current_count == target_count:
            return colors, percentages
        
        elif current_count > target_count:
            # Too many clusters - keep the top ones
            sorted_indices = np.argsort(percentages)[::-1]
            colors = colors[sorted_indices][:target_count]
            percentages = percentages[sorted_indices][:target_count]
            
            # Renormalize percentages
            percentages = (percentages / np.sum(percentages)) * 100
        
        else:
            # Too few clusters - duplicate/synthesize
            colors = list(colors)
            percentages = list(percentages)
            
            while len(colors) < target_count:
                # Duplicate the dominant color with variation
                dominant_idx = np.argmax(percentages)
                dominant_color = colors[dominant_idx]
                
                # Create variation
                variation = np.random.randint(-8, 9, size=3)
                new_color = np.clip(dominant_color + variation, 0, 255)
                
                colors.append(new_color)
                percentages.append(5.0)  # Small percentage
            
            # Convert back to numpy and normalize
            colors = np.array(colors)
            percentages = np.array(percentages)
            percentages = (percentages / np.sum(percentages)) * 100
        
        return colors, percentages
    
    def _correct_black_hair_colors(self, colors):
        """Helper method to correct colors that should be more black"""
        if colors is None:
            return colors
        
        corrected_colors = []
        
        for color in colors:
            r, g, b = color
            
            # Check if this should be more black-like
            # If it's too brown/red (common issue), make it more neutral black
            if r > 70 and (r - b) > 15:  # Too much red relative to blue
                # Make it more neutral and darker
                corrected_r = min(r * 0.7, 55)
                corrected_g = min(g * 0.8, 55) 
                corrected_b = min(b * 1.1, 65)  # Slightly boost blue
                
                corrected_colors.append([int(corrected_r), int(corrected_g), int(corrected_b)])
            elif r > 90:  # Too bright overall for black hair
                # Darken all components
                corrected_r = min(r * 0.6, 60)
                corrected_g = min(g * 0.6, 60)
                corrected_b = min(b * 0.6, 60)
                
                corrected_colors.append([int(corrected_r), int(corrected_g), int(corrected_b)])
            else:
                corrected_colors.append(color)
        
        return corrected_colors

    def extract_reliable_hair_colors(self, image, mask, n_colors=3, filename=None):
        """
        ENHANCED method with optional metadata support - but maintains original functionality!
        
        Args:
            image: Input image (BGR format)
            mask: Binary mask for the eyebrow region (from Facer segmentation)
            n_colors: FIXED number of dominant colors to extract (same for both eyebrows)
            filename: Optional filename for metadata lookup
            
        Returns:
            colors: Array of EXACTLY n_colors dominant colors in RGB format
            percentages: Percentage of each color (always sums to 100%)
            debug_images: Dictionary of debug images showing the process
        """
        # Extract hair pixels with ORIGINAL enhanced black hair detection
        hair_mask, debug_images = self.extract_eyebrow_hair_pixels_only(image, mask, debug=True)
        
        if hair_mask is None:
            return None, None, debug_images
        
        # OPTIONAL: Try to get metadata and apply light adjustments (won't break existing functionality)
        metadata = None
        if filename and self.metadata_handler:
            try:
                metadata = self.metadata_handler.get_person_metadata(filename)
                if metadata:
                    # Apply very conservative metadata-based adjustments
                    hair_mask = self.apply_metadata_adjustments(hair_mask, image, mask, metadata)
                    debug_images['0_metadata_applied'] = f"✅ Light adjustments for {metadata['ethnicity_name']} (skin cluster {metadata['skin_cluster']})"
                else:
                    debug_images['0_metadata'] = "No metadata found for this filename"
            except Exception as e:
                debug_images['0_metadata'] = f"Metadata lookup failed: {e}"
                print(f"Metadata processing failed: {e}")
        else:
            debug_images['0_metadata'] = "No metadata handler or filename provided"
        
        # Extract colors with FIXED cluster count (original algorithm)
        colors, percentages = self.extract_colors_from_hair_mask(image, hair_mask, n_colors)
        
        if colors is None:
            return None, None, debug_images
        
        # Add color extraction info to debug images
        if len(colors) > 0:
            # Validation info
            debug_images['13_cluster_validation'] = f"✅ Clusters: {len(colors)}/{n_colors} | Percentages sum: {np.sum(percentages):.1f}%" # type: ignore
            
            # Create a color palette visualization
            palette_height = 100
            palette_width = 400
            palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
            
            start_x = 0
            for i, (color, percentage) in enumerate(zip(colors, percentages)): # type: ignore
                width = int(palette_width * (percentage / 100))
                end_x = min(start_x + width, palette_width)
                if end_x > start_x:
                    palette[:, start_x:end_x] = color
                start_x = end_x
            
            debug_images['13_extracted_color_palette'] = palette
            
            # Add LAB values for verification
            lab_info = []
            for i, color in enumerate(colors):
                r, g, b = color
                rgb_color = sRGBColor(r/255, g/255, b/255)
                lab_color = convert_color(rgb_color, LabColor)
                lab_info.append(f"C{i+1}: L:{lab_color.lab_l:.1f} a:{lab_color.lab_a:.1f} b:{lab_color.lab_b:.1f} ({percentages[i]:.1f}%)") # type: ignore
            
            debug_images['14_lab_values'] = " | ".join(lab_info)
        
        return colors, percentages, debug_images
    
    # Keep all your existing visualization methods exactly as they were
    def create_color_palette(self, colors, percentages):
        if colors is None or percentages is None:
            return None
        palette = np.zeros((100, 300, 3), dtype=np.uint8)
        start_x = 0
        for i, (color, percentage) in enumerate(zip(colors, percentages)):
            width = int(300 * (percentage / 100))
            end_x = start_x + width
            palette[:, start_x:end_x] = color
            start_x = end_x
        return palette
        
    def create_plotly_pie_chart(self, colors, percentages):
        """Create an interactive Plotly pie chart visualization of the color distribution"""
        if colors is None or percentages is None or len(colors) == 0:
            return None
            
        # Convert RGB colors to hex for Plotly
        hex_colors = [f'rgb({int(r)}, {int(g)}, {int(b)})' for r, g, b in colors]
        
        # Create labels with percentages
        labels = [f'Color {i+1}: {p:.1f}%' for i, p in enumerate(percentages)]
        
        # Create the Plotly pie chart
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=percentages,
            marker=dict(colors=hex_colors, line=dict(color='white', width=2)),
            textinfo='label+percent',
            insidetextorientation='radial',
            textfont=dict(size=14, color='white'),
            hoverinfo='label+percent',
            hole=0.3
        )])
        
        # Update layout for better appearance
        fig.update_layout(
            title={
                'text': 'Eyebrow Hair Color Distribution',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18)
            },
            showlegend=False,
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Convert to JSON for Streamlit
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def create_plotly_lab_3d(self, colors, percentages):
        """Create an interactive 3D visualization of colors in LAB color space"""
        if colors is None or percentages is None or len(colors) == 0:
            return None
            
        # Convert RGB colors to LAB
        lab_values = []
        rgb_hex = []
        sizes = []
        labels = []
        
        for i, (color, percentage) in enumerate(zip(colors, percentages)):
            r, g, b = color
            rgb_hex.append(f'rgb({int(r)}, {int(g)}, {int(b)})')
            
            # Convert to LAB using colormath
            rgb_color = sRGBColor(r/255, g/255, b/255)
            lab_color = convert_color(rgb_color, LabColor)
            
            lab_values.append([
                lab_color.lab_l,  # L - Lightness # type: ignore
                lab_color.lab_a,  # a - green to red # type: ignore
                lab_color.lab_b   # b - blue to yellow # type: ignore
            ])
            
            # Size based on percentage (scaled for visibility)
            sizes.append(percentage * 5)
            labels.append(f'Color {i+1}: {percentage:.1f}%<br>RGB: ({r},{g},{b})<br>LAB: ({lab_color.lab_l:.1f}, {lab_color.lab_a:.1f}, {lab_color.lab_b:.1f})') # type: ignore
        
        # Create a DataFrame for Plotly
        df = pd.DataFrame(lab_values, columns=['L', 'a', 'b'])
        df['color'] = rgb_hex
        df['size'] = sizes
        df['label'] = labels
        df['percentage'] = percentages
        
        # Create 3D scatter plot using actual detected colors
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=df['L'],
            y=df['a'],
            z=df['b'],
            mode='markers+text',
            marker=dict(
                size=df['size'],
                color=df['color'],  # Use actual detected RGB colors
                opacity=0.9,
                line=dict(width=0),
                symbol='circle',
            ),
            text=[f'{p:.1f}%' for p in df['percentage']],
            hovertext=df['label'],
            hoverinfo='text',
            textposition='top center',
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title='Lightness (L)',
                yaxis_title='Green-Red (a)',
                zaxis_title='Blue-Yellow (b)',
                aspectmode='cube',
            ),
            title='Eyebrow Hair Colors in LAB Color Space',
            height=500,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False
        )
        
        # Convert to JSON for Streamlit
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
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