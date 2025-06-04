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

class ColorAnalysis:
    def __init__(self):
        pass
    
    def extract_eyebrow_hair_pixels_only(self, image, mask, debug=True):
        """
        Advanced method to extract ONLY eyebrow hair pixels, excluding skin tones.
        Uses multiple techniques including HSV thresholding, edge detection, and texture analysis.
        
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
        
        # --- METHOD 1: HSV-based hair detection ---
        h, s, v = cv2.split(hsv)
        
        # Create hair mask based on low brightness (Value channel)
        # Hair is typically much darker than skin
        hair_value_mask = np.zeros_like(mask)
        # Use adaptive threshold based on the image content
        masked_v = v[mask > 0]
        if len(masked_v) > 0:
            # Use percentile-based threshold for better adaptation
            value_threshold = np.percentile(masked_v, 30)  # Take darker 70% of pixels
            value_threshold = min(value_threshold, 80)  # Cap at 80 for very light images
        else:
            value_threshold = 100
            
        hair_value_mask[(v < value_threshold) & (mask > 0)] = 255
        debug_images['3_hsv_value_mask'] = cv2.cvtColor(hair_value_mask, cv2.COLOR_GRAY2RGB)
        
        # Additional saturation filtering - hair often has different saturation than skin
        hair_sat_mask = np.zeros_like(mask)
        # Hair can have lower saturation (more neutral) or higher saturation (more colorful)
        # We'll be more permissive here and focus on value
        hair_sat_mask[(s > 10) & (mask > 0)] = 255  # Remove very desaturated pixels (likely skin)
        debug_images['4_hsv_saturation_mask'] = cv2.cvtColor(hair_sat_mask, cv2.COLOR_GRAY2RGB)
        
        # Combine HSV masks
        hsv_hair_mask = cv2.bitwise_and(hair_value_mask, hair_sat_mask)
        debug_images['5_combined_hsv_mask'] = cv2.cvtColor(hsv_hair_mask, cv2.COLOR_GRAY2RGB)
        
        # --- METHOD 2: Edge detection for hair strands ---
        # Apply bilateral filter first to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Use Canny edge detection to find hair boundaries
        edges = cv2.Canny(bilateral, 30, 100)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Dilate edges to make them thicker (connect broken hair strands)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        debug_images['6_edge_detection'] = cv2.cvtColor(edges_dilated, cv2.COLOR_GRAY2RGB)
        
        # --- METHOD 3: Texture-based detection ---
        # Calculate local variance to detect textured regions (hair vs smooth skin)
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        # Convert to float32 for proper computation
        gray_float = gray.astype(np.float32)
        mean_img = cv2.filter2D(gray_float, -1, kernel)
        sqr_mean_img = cv2.filter2D(gray_float * gray_float, -1, kernel)
        
        # Calculate variance with proper type handling
        variance = np.maximum(0.0, sqr_mean_img - mean_img * mean_img)
        
        # Normalize variance with proper destination array
        variance_norm = np.zeros_like(variance, dtype=np.uint8)
        cv2.normalize(variance, variance_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # High variance indicates texture (hair), low variance indicates smooth areas (skin)
        texture_threshold = np.percentile(variance_norm[mask > 0], 70) if np.sum(mask) > 0 else 50
        texture_mask = np.zeros_like(mask)
        texture_mask[(variance_norm > texture_threshold) & (mask > 0)] = 255
        debug_images['7_texture_variance'] = cv2.cvtColor(variance_norm, cv2.COLOR_GRAY2RGB)
        debug_images['8_texture_mask'] = cv2.cvtColor(texture_mask, cv2.COLOR_GRAY2RGB)
        
        # --- METHOD 4: LAB color space analysis ---
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # In LAB space, hair typically has lower L (lightness) values
        lab_threshold = np.percentile(l_channel[mask > 0], 30) if np.sum(mask) > 0 else 120
        lab_hair_mask = np.zeros_like(mask)
        lab_hair_mask[(l_channel < lab_threshold) & (mask > 0)] = 255
        debug_images['9_lab_lightness_mask'] = cv2.cvtColor(lab_hair_mask, cv2.COLOR_GRAY2RGB)
        
        # --- SHOW COMBINED METHODS FOR DEBUGGING (but don't use) ---
        # Start with HSV mask as primary
        combined_mask = hsv_hair_mask.copy()
        
        # Add pixels detected by edge detection
        combined_mask = cv2.bitwise_or(combined_mask, edges_dilated)
        
        # Add pixels detected by texture analysis
        combined_mask = cv2.bitwise_or(combined_mask, texture_mask)
        
        # Intersect with LAB mask to ensure we keep dark pixels
        combined_mask = cv2.bitwise_and(combined_mask, lab_hair_mask)
        
        # Ensure we stay within the original eyebrow mask
        combined_mask = cv2.bitwise_and(combined_mask, mask)
        
        debug_images['10_combined_before_morphology'] = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2RGB)
        
        # --- USE ONLY LAB MASK AS FINAL RESULT ---
        # Instead of using the combined mask, use only the LAB mask
        final_mask = lab_hair_mask.copy()
        
        # Ensure we stay within the original eyebrow mask
        final_mask = cv2.bitwise_and(final_mask, mask)
        
        # --- MORPHOLOGICAL OPERATIONS ---
        # Clean up the mask with morphological operations
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Remove small noise
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
        # Fill small gaps
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
        
        debug_images['11_final_hair_mask'] = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)
        debug_images['11_note'] = "⚠️ NOTE: Final mask uses ONLY LAB lightness filtering (not combined methods)"
        
        # Create final visualization showing detected hair pixels
        hair_pixels_img = cv2.bitwise_and(image, image, mask=final_mask)
        debug_images['12_detected_hair_pixels'] = cv2.cvtColor(hair_pixels_img, cv2.COLOR_BGR2RGB)
        
        # If we don't have enough pixels, try a more aggressive approach
        final_mask_sum = int(np.sum(final_mask))  # Convert to int for comparison
        if final_mask_sum < 50:
            # Fall back to more aggressive LAB threshold
            fallback_threshold = np.percentile(l_channel[mask > 0], 20) if np.sum(mask) > 0 else 100
            fallback_mask = np.zeros_like(mask)
            fallback_mask[(l_channel < fallback_threshold) & (mask > 0)] = 255
            
            # Clean up fallback mask
            fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_OPEN, kernel_open)
            fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_CLOSE, kernel_close)
            
            debug_images['13_fallback_mask'] = cv2.cvtColor(fallback_mask, cv2.COLOR_GRAY2RGB)
            debug_images['13_fallback_note'] = f"Used more aggressive LAB threshold: 20th percentile (threshold: {fallback_threshold})"
            
            fallback_mask_sum = int(np.sum(fallback_mask))  # Convert to int for comparison
            if fallback_mask_sum > final_mask_sum:
                final_mask = fallback_mask
                hair_pixels_img = cv2.bitwise_and(image, image, mask=final_mask)
                debug_images['12_detected_hair_pixels'] = cv2.cvtColor(hair_pixels_img, cv2.COLOR_BGR2RGB)
        
        # Add final statistics
        lab_threshold_used = lab_threshold if final_mask_sum >= 50 else np.percentile(l_channel[mask > 0], 20)
        debug_images['14_final_stats'] = f"LAB L-channel threshold used: {lab_threshold_used:.1f}\nPixels detected: {final_mask_sum}\nMethod: LAB lightness filtering only"
        
        return final_mask, debug_images
        
    def extract_colors_from_hair_mask(self, image, hair_mask, n_colors=3):
        """
        Extract dominant colors from the refined hair mask.
        
        Args:
            image: Input image (BGR format)
            hair_mask: Binary mask containing only hair pixels
            n_colors: Number of dominant colors to extract
            
        Returns:
            colors: Array of dominant colors in RGB format
            percentages: Percentage of each color
        """
        if hair_mask is None or np.sum(hair_mask) == 0:
            return None, None
        
        # Get hair pixels
        hair_pixels = image[hair_mask > 0]
        
        if len(hair_pixels) < n_colors * 5:  # Need minimum pixels for clustering
            return None, None
        
        # Convert from BGR to RGB
        hair_pixels_rgb = hair_pixels[:, ::-1]
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10,algorithm='lloyd',max_iter=3000)
        kmeans.fit(hair_pixels_rgb)
        
        # Get colors and percentages
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / len(labels) * 100
        
        # Sort by percentage (most dominant first)
        sorted_indices = np.argsort(percentages)[::-1]
        colors = colors[sorted_indices]
        percentages = percentages[sorted_indices]
        
        return colors, percentages
    
    def extract_reliable_hair_colors(self, image, mask, n_colors=3):
        """
        Enhanced method to extract reliable eyebrow hair colors using multiple techniques.
        This method focuses exclusively on hair pixels, excluding skin tones.
        
        Args:
            image: Input image (BGR format)
            mask: Binary mask for the eyebrow region (from Facer segmentation)
            n_colors: Number of dominant colors to extract
            
        Returns:
            colors: Array of dominant colors in RGB format
            percentages: Percentage of each color
            debug_images: Dictionary of debug images showing the process
        """
        # Extract hair pixels only
        hair_mask, debug_images = self.extract_eyebrow_hair_pixels_only(image, mask, debug=True)
        
        if hair_mask is None:
            return None, None, debug_images
        
        # Extract colors from hair pixels
        colors, percentages = self.extract_colors_from_hair_mask(image, hair_mask, n_colors)
        
        if colors is None:
            return None, None, debug_images
        
        # Add color extraction info to debug images
        if len(colors) > 0:
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
        
        return colors, percentages, debug_images
    
    
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
        # DO NOT convert to BGR, keep as RGB for Streamlit
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