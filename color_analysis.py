import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from colormath.color_objects import sRGBColor, LabColor, LCHabColor, HSVColor
from colormath.color_conversions import convert_color

class ColorAnalysis:
    def __init__(self):
        pass
    
    def extract_dominant_colors(self, image, mask, n_colors=2):
        """Extract dominant colors from the masked region using KMeans clustering"""
        if mask is None or np.sum(mask) == 0:
            return None, None
        
        # Apply mask to image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Convert to HSV for better hair/skin separation
        hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        
        # Create a more restrictive mask for hair pixels
        # Hair tends to have lower value (darker) than skin
        _, v_channel = cv2.threshold(hsv_image[:,:,2], 150, 255, cv2.THRESH_BINARY_INV)
        
        # Combine with original mask
        hair_mask = cv2.bitwise_and(mask, v_channel)
        
        # If hair mask is too small, fall back to original mask
        if np.sum(hair_mask) < 50:
            hair_mask = mask
        
        # Apply refined hair mask
        hair_pixels = image[hair_mask > 0]
        
        # Reshape the image to be a list of pixels
        pixels = hair_pixels.reshape(-1, 3)
        
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
