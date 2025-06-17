# pylint: disable=no-member
# pyright: reportOperatorIssue=false, reportArgumentType=false
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import ndimage
from skimage import filters
import os

class EnhancedHairAnalysisDashboard:
    def __init__(self):
        """Enhanced hair analysis with more methods + LAB values display"""
        pass
    
    def calculate_method_quality(self, method_mask, gray_image, original_mask):
        """Calculate quality score for a detection method"""
        if np.sum(method_mask) == 0:
            return 0
        
        pixel_ratio = np.sum(method_mask) / max(np.sum(original_mask), 1)
        if pixel_ratio < 0.05:
            count_score = pixel_ratio * 100
        elif pixel_ratio > 0.4:
            count_score = max(0, 100 - (pixel_ratio - 0.4) * 200)
        else:
            count_score = 100
        
        detected_pixels = gray_image[method_mask > 0]
        all_pixels = gray_image[original_mask > 0]
        
        if len(detected_pixels) > 0 and len(all_pixels) > 0:
            avg_detected = np.mean(detected_pixels)
            avg_all = np.mean(all_pixels)
            darkness_score = max(0, min(100, (avg_all - avg_detected) * 2))
        else:
            darkness_score = 0
        
        num_components, _ = cv2.connectedComponents(method_mask)
        if num_components <= 3:
            coherence_score = 100
        else:
            coherence_score = max(0, 100 - (num_components - 3) * 10)
        
        quality_score = (count_score * 0.4 + darkness_score * 0.4 + coherence_score * 0.2)
        return quality_score

    def apply_all_detection_methods(self, image, mask=None):
        """Apply ALL detection methods including new ones"""
        if mask is None:
            mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        
        methods_results = {}
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
        
        print("🔍 Testing ALL detection methods...")
        
        # METHOD 1: HSV Enhanced
        try:
            h, s, v = cv2.split(hsv)
            masked_v = v[mask > 0]
            
            if len(masked_v) > 0:
                v_mean = np.mean(masked_v) # type: ignore
                if v_mean > 120:
                    v_threshold = np.percentile(masked_v, 45) # type: ignore
                    s_threshold = 12
                elif v_mean > 80:
                    v_threshold = np.percentile(masked_v, 35) # type: ignore
                    s_threshold = 10
                else:
                    v_threshold = np.percentile(masked_v, 25) # type: ignore
                    s_threshold = 8
                
                hsv_mask = np.zeros_like(mask)
                hsv_mask[(v < v_threshold) & (s > s_threshold) & (mask > 0)] = 255
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)
                
                methods_results['HSV Enhanced'] = {
                    'mask': hsv_mask,
                    'pixel_count': np.sum(hsv_mask),
                    'quality_score': self.calculate_method_quality(hsv_mask, gray, mask),
                    'description': f'V<{v_threshold:.0f}, S>{s_threshold}',
                    'success': np.sum(hsv_mask) > 20
                }
            else:
                methods_results['HSV Enhanced'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': 'No pixels', 'pixel_count': 0, 'quality_score': 0}
        except Exception as e:
            methods_results['HSV Enhanced'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': str(e), 'pixel_count': 0, 'quality_score': 0}
        
        # METHOD 2: LAB Lightness
        try:
            l_channel = lab[:, :, 0]
            masked_l = l_channel[mask > 0]
            
            if len(masked_l) > 0:
                l_threshold = np.percentile(masked_l, 30) # type: ignore
                lab_mask = np.zeros_like(mask)
                lab_mask[(l_channel < l_threshold) & (mask > 0)] = 255
                
                methods_results['LAB Lightness'] = {
                    'mask': lab_mask,
                    'pixel_count': np.sum(lab_mask),
                    'quality_score': self.calculate_method_quality(lab_mask, gray, mask),
                    'description': f'L<{l_threshold:.1f}',
                    'success': np.sum(lab_mask) > 20
                }
            else:
                methods_results['LAB Lightness'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': 'No pixels', 'pixel_count': 0, 'quality_score': 0}
        except Exception as e:
            methods_results['LAB Lightness'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': str(e), 'pixel_count': 0, 'quality_score': 0}
        
        # METHOD 3: Edge Detection
        try:
            bilateral = cv2.bilateralFilter(gray, 9, 80, 80)
            edges = cv2.Canny(bilateral, 20, 60)
            edges = cv2.bitwise_and(edges, edges, mask=mask)
            
            kernel_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 1))
            kernel_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)
            
            methods_results['Edge Detection'] = {
                'mask': edges,
                'pixel_count': np.sum(edges),
                'quality_score': self.calculate_method_quality(edges, gray, mask),
                'description': 'Canny + morphology',
                'success': np.sum(edges) > 15
            }
        except Exception as e:
            methods_results['Edge Detection'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': str(e), 'pixel_count': 0, 'quality_score': 0}
        
        # METHOD 4: Gabor Filters
        try:
            gabor_response = np.zeros_like(gray, dtype=np.float32)
            for angle in [0, 30, 60, 90, 120, 150]:
                theta = np.radians(angle)
                kernel = cv2.getGaborKernel((11, 11), 3, theta, 6, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                gabor_response += np.abs(filtered)
            
            gabor_response = cv2.normalize(gabor_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # type: ignore
            gabor_threshold = np.percentile(gabor_response[mask > 0], 70) if np.sum(mask) > 0 else 50
            gabor_mask = np.zeros_like(mask)
            gabor_mask[(gabor_response > gabor_threshold) & (mask > 0)] = 255
            
            methods_results['Gabor Filters'] = {
                'mask': gabor_mask,
                'pixel_count': np.sum(gabor_mask),
                'quality_score': self.calculate_method_quality(gabor_mask, gray, mask),
                'description': f'6-direction filters>{gabor_threshold}',
                'success': np.sum(gabor_mask) > 20
            }
        except Exception as e:
            methods_results['Gabor Filters'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': str(e), 'pixel_count': 0, 'quality_score': 0}
        
        # METHOD 5: Texture Variance
        try:
            kernel = np.ones((5, 5), np.float32) / 25
            gray_float = gray.astype(np.float32)
            mean_img = cv2.filter2D(gray_float, -1, kernel)
            sqr_mean_img = cv2.filter2D(gray_float * gray_float, -1, kernel)
            variance = np.maximum(0.0, sqr_mean_img - mean_img * mean_img)
            
            variance_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # type: ignore
            texture_threshold = np.percentile(variance_norm[mask > 0], 60) if np.sum(mask) > 0 else 50
            texture_mask = np.zeros_like(mask)
            texture_mask[(variance_norm > texture_threshold) & (mask > 0)] = 255
            
            methods_results['Texture Variance'] = {
                'mask': texture_mask,
                'pixel_count': np.sum(texture_mask),
                'quality_score': self.calculate_method_quality(texture_mask, gray, mask),
                'description': f'Variance>{texture_threshold}',
                'success': np.sum(texture_mask) > 20
            }
        except Exception as e:
            methods_results['Texture Variance'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': str(e), 'pixel_count': 0, 'quality_score': 0}
        
        # METHOD 6: Statistical Outliers
        try:
            masked_pixels = gray[mask > 0]
            if len(masked_pixels) > 50:
                mean_intensity = np.mean(masked_pixels) # type: ignore
                std_intensity = np.std(masked_pixels) # type: ignore
                outlier_threshold = mean_intensity - (1.5 * std_intensity)
                
                outlier_mask = np.zeros_like(mask)
                outlier_mask[(gray < outlier_threshold) & (mask > 0)] = 255
                
                methods_results['Statistical Outliers'] = {
                    'mask': outlier_mask,
                    'pixel_count': np.sum(outlier_mask),
                    'quality_score': self.calculate_method_quality(outlier_mask, gray, mask),
                    'description': f'< mean-1.5*std ({outlier_threshold:.0f})',
                    'success': np.sum(outlier_mask) > 15
                }
            else:
                methods_results['Statistical Outliers'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': 'Insufficient pixels', 'pixel_count': 0, 'quality_score': 0}
        except Exception as e:
            methods_results['Statistical Outliers'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': str(e), 'pixel_count': 0, 'quality_score': 0}
        
        # METHOD 7: CLAHE Enhanced
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            clahe_enhanced = clahe.apply(gray)
            
            # Apply threshold on enhanced image
            enhanced_threshold = np.percentile(clahe_enhanced[mask > 0], 25) if np.sum(mask) > 0 else 50 # type: ignore
            clahe_mask = np.zeros_like(mask)
            clahe_mask[(clahe_enhanced < enhanced_threshold) & (mask > 0)] = 255
            
            # Clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            clahe_mask = cv2.morphologyEx(clahe_mask, cv2.MORPH_CLOSE, kernel)
            
            methods_results['CLAHE Enhanced'] = {
                'mask': clahe_mask,
                'pixel_count': np.sum(clahe_mask),
                'quality_score': self.calculate_method_quality(clahe_mask, gray, mask),
                'description': f'CLAHE + threshold<{enhanced_threshold:.0f}',
                'success': np.sum(clahe_mask) > 20
            }
        except Exception as e:
            methods_results['CLAHE Enhanced'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': str(e), 'pixel_count': 0, 'quality_score': 0}
        
        # METHOD 8: Morphological Top-Hat
        try:
            kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_tophat)
            tophat_threshold = np.percentile(tophat[mask > 0], 75) if np.sum(mask) > 0 else 20 # type: ignore
            tophat_mask = np.zeros_like(mask)
            tophat_mask[(tophat > tophat_threshold) & (mask > 0)] = 255
            
            methods_results['Top-Hat Transform'] = {
                'mask': tophat_mask,
                'pixel_count': np.sum(tophat_mask),
                'quality_score': self.calculate_method_quality(tophat_mask, gray, mask),
                'description': f'Black top-hat>{tophat_threshold}',
                'success': np.sum(tophat_mask) > 15
            }
        except Exception as e:
            methods_results['Top-Hat Transform'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': str(e), 'pixel_count': 0, 'quality_score': 0}
        
        # METHOD 9: Adaptive Threshold
        try:
            adaptive_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
            adaptive_mask = cv2.bitwise_and(adaptive_mask, mask)
            
            # Clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            adaptive_mask = cv2.morphologyEx(adaptive_mask, cv2.MORPH_OPEN, kernel)
            
            methods_results['Adaptive Threshold'] = {
                'mask': adaptive_mask,
                'pixel_count': np.sum(adaptive_mask),
                'quality_score': self.calculate_method_quality(adaptive_mask, gray, mask),
                'description': 'Gaussian adaptive thresh',
                'success': np.sum(adaptive_mask) > 20
            }
        except Exception as e:
            methods_results['Adaptive Threshold'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': str(e), 'pixel_count': 0, 'quality_score': 0}
        
        # METHOD 10: Gaussian Mixture Model
        try:
            # Use LAB values for GMM
            lab_pixels = lab.reshape(-1, 3)
            mask_flat = mask.flatten()
            masked_lab_pixels = lab_pixels[mask_flat > 0]
            
            if len(masked_lab_pixels) > 100:
                gmm = GaussianMixture(n_components=3, random_state=42)
                labels = gmm.fit_predict(masked_lab_pixels)
                
                # Find the component with lowest L values (likely hair)
                component_means = []
                for i in range(3):
                    component_pixels = masked_lab_pixels[labels == i]
                    if len(component_pixels) > 0:
                        mean_l = np.mean(component_pixels[:, 0]) # type: ignore
                        component_means.append((i, mean_l, len(component_pixels)))
                
                if component_means:
                    # Sort by L value (darkest first)
                    component_means.sort(key=lambda x: x[1])
                    darkest_component = component_means[0][0]
                    
                    # Create mask for darkest component
                    full_labels = np.zeros(len(lab_pixels), dtype=int)
                    full_labels[mask_flat > 0] = labels
                    
                    gmm_segmented = full_labels.reshape(mask.shape)
                    gmm_mask = (gmm_segmented == darkest_component).astype(np.uint8) * 255
                    
                    methods_results['Gaussian Mixture'] = {
                        'mask': gmm_mask,
                        'pixel_count': np.sum(gmm_mask),
                        'quality_score': self.calculate_method_quality(gmm_mask, gray, mask),
                        'description': f'GMM darkest component',
                        'success': np.sum(gmm_mask) > 50
                    }
                else:
                    methods_results['Gaussian Mixture'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': 'No components', 'pixel_count': 0, 'quality_score': 0}
            else:
                methods_results['Gaussian Mixture'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': 'Insufficient pixels', 'pixel_count': 0, 'quality_score': 0}
        except Exception as e:
            methods_results['Gaussian Mixture'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': str(e), 'pixel_count': 0, 'quality_score': 0}
        
        # METHOD 11: Multi-Scale Edges
        try:
            multi_edges = np.zeros_like(gray, dtype=np.float32)
            
            for sigma in [0.5, 1.0, 1.5]:
                blurred = cv2.GaussianBlur(gray, (5, 5), sigma)
                edges = cv2.Canny(blurred, 20, 60)
                multi_edges += edges.astype(np.float32)
            
            multi_edges = cv2.normalize(multi_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # type: ignore
            multi_edges = cv2.bitwise_and(multi_edges, multi_edges, mask=mask)
            
            # Threshold and clean
            _, multi_edges_mask = cv2.threshold(multi_edges, 100, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            multi_edges_mask = cv2.morphologyEx(multi_edges_mask, cv2.MORPH_CLOSE, kernel)
            
            methods_results['Multi-Scale Edges'] = {
                'mask': multi_edges_mask,
                'pixel_count': np.sum(multi_edges_mask),
                'quality_score': self.calculate_method_quality(multi_edges_mask, gray, mask),
                'description': '3-scale edge detection',
                'success': np.sum(multi_edges_mask) > 15
            }
        except Exception as e:
            methods_results['Multi-Scale Edges'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': str(e), 'pixel_count': 0, 'quality_score': 0}
        
        # METHOD 12: LAB A+B Combined
        try:
            a_channel = lab[:, :, 1] - 128  # Convert to actual LAB range
            b_channel = lab[:, :, 2] - 128
            
            # Combine A and B channels for color-based detection
            ab_combined = np.sqrt(a_channel.astype(float)**2 + b_channel.astype(float)**2)
            ab_combined = cv2.normalize(ab_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # type: ignore
            
            ab_threshold = np.percentile(ab_combined[mask > 0], 60) if np.sum(mask) > 0 else 50
            ab_mask = np.zeros_like(mask)
            ab_mask[(ab_combined > ab_threshold) & (mask > 0)] = 255
            
            methods_results['LAB A+B Combined'] = {
                'mask': ab_mask,
                'pixel_count': np.sum(ab_mask),
                'quality_score': self.calculate_method_quality(ab_mask, gray, mask),
                'description': f'sqrt(A²+B²)>{ab_threshold}',
                'success': np.sum(ab_mask) > 20
            }
        except Exception as e:
            methods_results['LAB A+B Combined'] = {'mask': np.zeros_like(mask), 'success': False, 'reason': str(e), 'pixel_count': 0, 'quality_score': 0}
        
        return methods_results

    def extract_colors_with_lab_values(self, image, mask, n_colors=3):
        """Extract colors and return RGB colors WITH their LAB values"""
        if mask is None or np.sum(mask) == 0:
            return None, None, None
        
        hair_pixels = image[mask > 0]
        if len(hair_pixels) < n_colors:
            return None, None, None
        
        hair_pixels_rgb = hair_pixels[:, ::-1]  # BGR to RGB
        
        try:
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(hair_pixels_rgb)
            
            colors_rgb = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = (counts / len(labels)) * 100
            
            sorted_indices = np.argsort(percentages)[::-1]
            colors_rgb = colors_rgb[sorted_indices]
            percentages = percentages[sorted_indices]
            
            # Convert RGB colors to LAB values
            lab_values = []
            for color_rgb in colors_rgb:
                # Create a 1x1 image with this color and convert to LAB
                color_bgr = color_rgb[::-1]  # RGB to BGR for OpenCV
                color_image = np.array([[color_bgr]], dtype=np.uint8)
                color_lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)[0, 0]
                
                # Convert to actual LAB range
                l_val = float(color_lab[0])  # L: 0-100
                a_val = float(color_lab[1]) - 128  # A: -128 to +127
                b_val = float(color_lab[2]) - 128  # B: -128 to +127
                
                lab_values.append([l_val, a_val, b_val])
            
            return colors_rgb, percentages, lab_values
        except:
            return None, None, None

    def analyze_lab_values(self, image, mask=None):
        """Analyze LAB color space values"""
        if mask is not None:
            analysis_image = cv2.bitwise_and(image, image, mask=mask)
        else:
            analysis_image = image.copy()
        
        lab_image = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        
        # Get pixels within mask
        if mask is not None:
            l_values = l_channel[mask > 0]
            a_values = a_channel[mask > 0] - 128
            b_values = b_channel[mask > 0] - 128
        else:
            l_values = l_channel.flatten()
            a_values = a_channel.flatten() - 128
            b_values = b_channel.flatten() - 128
        
        if len(l_values) == 0:
            return None
        
        lab_stats = {
            'L': {
                'min': float(np.min(l_values)),
                'max': float(np.max(l_values)),
                'mean': float(np.mean(l_values)),
                'std': float(np.std(l_values)),
                'median': float(np.median(l_values)),
                'p10': float(np.percentile(l_values, 10)),
                'p25': float(np.percentile(l_values, 25)),
                'p75': float(np.percentile(l_values, 75)),
                'p90': float(np.percentile(l_values, 90))
            },
            'A': {
                'min': float(np.min(a_values)),
                'max': float(np.max(a_values)),
                'mean': float(np.mean(a_values)),
                'std': float(np.std(a_values)),
                'median': float(np.median(a_values))
            },
            'B': {
                'min': float(np.min(b_values)),
                'max': float(np.max(b_values)),
                'mean': float(np.mean(b_values)),
                'std': float(np.std(b_values)),
                'median': float(np.median(b_values))
            }
        }
        
        # Color classification
        if lab_stats['L']['mean'] < 30:
            hair_type = "Very Dark/Black"
        elif lab_stats['L']['mean'] < 50:
            hair_type = "Dark Brown"
        elif lab_stats['L']['mean'] < 70:
            hair_type = "Medium Brown"
        else:
            hair_type = "Light"
        
        # Undertone analysis
        if lab_stats['A']['mean'] > 5 and lab_stats['B']['mean'] > 5:
            undertone = "Warm (Red-Yellow)"
        elif lab_stats['A']['mean'] < -5 and lab_stats['B']['mean'] < -5:
            undertone = "Cool (Green-Blue)"
        elif abs(lab_stats['A']['mean']) < 5 and abs(lab_stats['B']['mean']) < 5:
            undertone = "Neutral"
        else:
            undertone = "Mixed"
        
        # Contrast analysis
        l_range = lab_stats['L']['max'] - lab_stats['L']['min']
        if l_range < 20:
            contrast_level = "Very Low"
        elif l_range < 40:
            contrast_level = "Low"
        elif l_range < 60:
            contrast_level = "Medium"
        else:
            contrast_level = "High"
        
        return {
            'stats': lab_stats,
            'hair_type': hair_type,
            'undertone': undertone,
            'contrast_level': contrast_level,
            'l_range': l_range,
            'l_channel': l_channel,
            'a_channel': a_channel,
            'b_channel': b_channel,
            'lab_image': lab_image
        }

    def create_comprehensive_dashboard(self, image, mask=None, n_colors=3, save_path=None):
        """Create the comprehensive dashboard with LAB values focus"""
        print("🚀 Creating comprehensive hair analysis dashboard with LAB values...")
        
        # Apply ALL detection methods
        methods_results = self.apply_all_detection_methods(image, mask)
        
        # Analyze LAB values
        print("🎨 Analyzing LAB color space...")
        lab_analysis = self.analyze_lab_values(image, mask)
        
        # Create the dashboard
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(5, 6, hspace=0.4, wspace=0.3)
        
        # Convert image for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ============= ROW 1: ORIGINAL + TOP DETECTION METHODS =============
        
        # Original image
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(image_rgb)
        ax_orig.set_title('Original Image', fontsize=12, fontweight='bold')
        ax_orig.axis('off')
        
        # Get successful methods sorted by quality
        successful_methods = [(k, v) for k, v in methods_results.items() if v.get('success', False)]
        successful_methods.sort(key=lambda x: x[1].get('quality_score', 0), reverse=True)
        
        # Show top 5 methods
        for i in range(5):
            ax = fig.add_subplot(gs[0, i+1])
            
            if i < len(successful_methods):
                method_name, method_data = successful_methods[i]
                
                # Create overlay
                overlay = image_rgb.copy()
                method_mask = method_data['mask']
                overlay[method_mask > 0] = [0, 255, 0]  # Green overlay
                blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
                
                ax.imshow(blended)
                
                pixel_count = method_data['pixel_count']
                quality_score = method_data.get('quality_score', 0)
                ax.set_title(f'{method_name}\n✅ {pixel_count}px (Q:{quality_score:.0f})', 
                           fontsize=10, color='green', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No more\nsuccessful\nmethods', 
                       ha='center', va='center', fontsize=10, color='gray')
                ax.set_title('---', fontsize=10, color='gray')
            
            ax.axis('off')
        
        # ============= ROW 2: LAB CHANNELS =============
        
        if lab_analysis:
            # L Channel
            ax_l = fig.add_subplot(gs[1, 0])
            im_l = ax_l.imshow(lab_analysis['l_channel'], cmap='gray', vmin=0, vmax=100)
            ax_l.set_title('L Channel (Lightness)', fontsize=12, fontweight='bold')
            ax_l.axis('off')
            plt.colorbar(im_l, ax=ax_l, fraction=0.046, pad=0.04)
            
            # A Channel
            ax_a = fig.add_subplot(gs[1, 1])
            a_centered = lab_analysis['a_channel'].astype(float) - 128
            im_a = ax_a.imshow(a_centered, cmap='RdYlGn_r', vmin=-128, vmax=127)
            ax_a.set_title('A Channel (Green-Red)', fontsize=12, fontweight='bold')
            ax_a.axis('off')
            plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
            
            # B Channel
            ax_b = fig.add_subplot(gs[1, 2])
            b_centered = lab_analysis['b_channel'].astype(float) - 128
            im_b = ax_b.imshow(b_centered, cmap='coolwarm', vmin=-128, vmax=127)
            ax_b.set_title('B Channel (Blue-Yellow)', fontsize=12, fontweight='bold')
            ax_b.axis('off')
            plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
            
            # LAB Statistics
            ax_lab_stats = fig.add_subplot(gs[1, 3:6])
            ax_lab_stats.axis('off')
            
            stats = lab_analysis['stats']
            stats_text = f"""LAB COLOR ANALYSIS

🎨 HAIR TYPE: {lab_analysis['hair_type']}
🌈 UNDERTONE: {lab_analysis['undertone']}
📊 CONTRAST: {lab_analysis['contrast_level']} (Range: {lab_analysis['l_range']:.1f})

💡 L (Lightness): {stats['L']['mean']:.1f} ± {stats['L']['std']:.1f}
   Range: {stats['L']['min']:.1f} - {stats['L']['max']:.1f}
   P10-P90: {stats['L']['p10']:.1f} - {stats['L']['p90']:.1f}

🎨 A (Green-Red): {stats['A']['mean']:.1f} ± {stats['A']['std']:.1f}
   Range: {stats['A']['min']:.1f} - {stats['A']['max']:.1f}

🌈 B (Blue-Yellow): {stats['B']['mean']:.1f} ± {stats['B']['std']:.1f}
   Range: {stats['B']['min']:.1f} - {stats['B']['max']:.1f}

💡 HAIR DETECTION TIPS:
   • Best L threshold: < {stats['L']['p25']:.1f}
   • A/B separation: A={stats['A']['mean']:.1f}, B={stats['B']['mean']:.1f}
   • Use A+B combined for color-based detection"""
            
            ax_lab_stats.text(0.05, 0.95, stats_text, transform=ax_lab_stats.transAxes,
                            fontsize=10, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # ============= ROW 3 & 4: EXTRACTED HAIR COLORS WITH LAB VALUES =============
        
        row_3_methods = successful_methods[:6]
        row_4_methods = successful_methods[6:12]
        
        # ROW 3: First 6 successful methods
        for i in range(6):
            ax = fig.add_subplot(gs[2, i])
            
            if i < len(row_3_methods):
                method_name, method_data = row_3_methods[i]
                
                # Extract colors WITH LAB values
                colors_rgb, percentages, lab_values = self.extract_colors_with_lab_values(image, method_data['mask'], n_colors)
                
                if colors_rgb is not None and lab_values is not None:
                    # Create color palette
                    palette_height = 40
                    palette_width = 200
                    palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
                    
                    start_x = 0
                    for color, percentage in zip(colors_rgb, percentages):
                        width = int(palette_width * (percentage / 100))
                        end_x = min(start_x + width, palette_width)
                        if end_x > start_x:
                            palette[:, start_x:end_x] = color
                        start_x = end_x
                    
                    ax.imshow(palette)
                    
                    # Add LAB values as text
                    lab_text = []
                    for j, (lab_val, pct) in enumerate(zip(lab_values, percentages)): # type: ignore
                        l_val, a_val, b_val = lab_val
                        lab_text.append(f'C{j+1}: L={l_val:.1f} A={a_val:.1f} B={b_val:.1f}\n({pct:.1f}%)')
                    
                    ax.set_title(f'{method_name}\n' + '\n'.join(lab_text), 
                               fontsize=8, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, f'{method_name}\nColor extraction\nfailed', 
                           ha='center', va='center', fontsize=10)
                    ax.set_title(f'{method_name}', fontsize=10, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No more\nmethods', ha='center', va='center', fontsize=10, color='gray')
                ax.set_title('---', fontsize=10, color='gray')
            
            ax.axis('off')
        
        # ROW 4: Next 6 successful methods
        for i in range(6):
            ax = fig.add_subplot(gs[3, i])
            
            if i < len(row_4_methods):
                method_name, method_data = row_4_methods[i]
                
                # Extract colors WITH LAB values
                colors_rgb, percentages, lab_values = self.extract_colors_with_lab_values(image, method_data['mask'], n_colors)
                
                if colors_rgb is not None and lab_values is not None:
                    # Create color palette
                    palette_height = 40
                    palette_width = 200
                    palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
                    
                    start_x = 0
                    for color, percentage in zip(colors_rgb, percentages):
                        width = int(palette_width * (percentage / 100))
                        end_x = min(start_x + width, palette_width)
                        if end_x > start_x:
                            palette[:, start_x:end_x] = color
                        start_x = end_x
                    
                    ax.imshow(palette)
                    
                    # Add LAB values as text
                    lab_text = []
                    for j, (lab_val, pct) in enumerate(zip(lab_values, percentages)): # type: ignore
                        l_val, a_val, b_val = lab_val
                        lab_text.append(f'C{j+1}: L={l_val:.1f} A={a_val:.1f} B={b_val:.1f}\n({pct:.1f}%)')
                    
                    ax.set_title(f'{method_name}\n' + '\n'.join(lab_text), 
                               fontsize=8, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, f'{method_name}\nColor extraction\nfailed', 
                           ha='center', va='center', fontsize=10)
                    ax.set_title(f'{method_name}', fontsize=10, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No more\nmethods', ha='center', va='center', fontsize=10, color='gray')
                ax.set_title('---', fontsize=10, color='gray')
            
            ax.axis('off')
        
        # ============= ROW 5: 3D LAB PLOT + SUMMARY TABLE =============
        
        if lab_analysis:
            # 3D LAB plot
            ax_3d = fig.add_subplot(gs[4, 0:3], projection='3d')
            
            # Sample pixels for 3D plot
            all_pixels = lab_analysis['lab_image'].reshape(-1, 3)
            if mask is not None:
                mask_flat = mask.flatten()
                all_pixels = all_pixels[mask_flat > 0]
            
            # Sample for performance
            if len(all_pixels) > 2000:
                indices = np.random.choice(len(all_pixels), 2000, replace=False)
                sample_pixels = all_pixels[indices]
            else:
                sample_pixels = all_pixels
            
            L_vals = sample_pixels[:, 0]
            A_vals = sample_pixels[:, 1] - 128
            B_vals = sample_pixels[:, 2] - 128
            
            scatter = ax_3d.scatter(L_vals, A_vals, B_vals, c=L_vals, cmap='viridis', alpha=0.6, s=2)
            ax_3d.set_xlabel('L (Lightness)')
            ax_3d.set_ylabel('A (Green-Red)')
            ax_3d.set_zlabel('B (Blue-Yellow)')
            ax_3d.set_title('LAB Color Space Distribution', fontsize=12, fontweight='bold')
        
        # Summary table
        ax_summary = fig.add_subplot(gs[4, 3:6])
        ax_summary.axis('off')
        
        # Create summary table with LAB info
        table_data = []
        headers = ['Method', 'Success', 'Pixels', 'Quality', 'Hair Colors (LAB)']
        
        for method_name, method_data in methods_results.items():
            success = "✅" if method_data.get('success', False) else "❌"
            pixels = method_data.get('pixel_count', 0)
            quality = f"{method_data.get('quality_score', 0):.0f}" if method_data.get('success', False) else "0"
            
            # Get LAB values for hair colors
            if method_data.get('success', False):
                colors_rgb, percentages, lab_values = self.extract_colors_with_lab_values(image, method_data['mask'], 2)
                if lab_values is not None and len(lab_values) >= 1:
                    l1, a1, b1 = lab_values[0]
                    lab_info = f"L:{l1:.0f} A:{a1:.0f} B:{b1:.0f}"
                else:
                    lab_info = "Failed"
            else:
                lab_info = "N/A"
            
            table_data.append([method_name[:12], success, f"{pixels:,}", quality, lab_info])
        
        table = ax_summary.table(cellText=table_data, colLabels=headers,
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        ax_summary.set_title('Methods Summary with LAB Values', fontsize=12, fontweight='bold')
        
        # Overall title
        fig.suptitle('Enhanced Hair Analysis Dashboard - Hair Extraction with LAB Values', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📸 Enhanced dashboard saved to: {save_path}")
        
        plt.show()
        
        return methods_results, lab_analysis

def run_enhanced_hair_analysis(image_path, region=None, n_colors=3):
    """
    Run enhanced hair analysis with 12 methods + LAB values focus
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"🖼️  Analyzing: {image_path}")
    print(f"📏 Image size: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Apply region if specified
    mask = None
    if region:
        x, y, w, h = region
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255
        print(f"🎯 Using region: ({x}, {y}, {w}, {h})")
    
    # Initialize dashboard
    dashboard = EnhancedHairAnalysisDashboard()
    
    # Create enhanced dashboard
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = f"{base_name}_enhanced_lab_dashboard.png"
    
    methods_results, lab_analysis = dashboard.create_comprehensive_dashboard(
        image, mask, n_colors, save_path
    )
    
    # Print summary
    print("\n" + "="*80)
    print("📊 ENHANCED ANALYSIS SUMMARY")
    print("="*80)
    
    successful_methods = [k for k, v in methods_results.items() if v.get('success', False)]
    print(f"✅ Successful detection methods: {len(successful_methods)}/{len(methods_results)}")
    if successful_methods:
        # Sort by quality and show top 5
        sorted_methods = sorted([(k, v) for k, v in methods_results.items() if v.get('success', False)], 
                              key=lambda x: x[1].get('quality_score', 0), reverse=True)
        print(f"🏆 Top 5 methods:")
        for i, (name, data) in enumerate(sorted_methods[:5]):
            print(f"   {i+1}. {name}: {data['pixel_count']} pixels (Quality: {data.get('quality_score', 0):.0f})")
    else:
        print("   ⚠️  No methods detected hair successfully")
    
    if lab_analysis:
        print(f"\n🎨 LAB Analysis:")
        print(f"   Hair type: {lab_analysis['hair_type']}")
        print(f"   Undertone: {lab_analysis['undertone']}")
        print(f"   Average lightness: {lab_analysis['stats']['L']['mean']:.1f}")
        print(f"   Contrast level: {lab_analysis['contrast_level']} (L range: {lab_analysis['l_range']:.1f})")
        print(f"   A value: {lab_analysis['stats']['A']['mean']:.1f} (Green- / Red+)")
        print(f"   B value: {lab_analysis['stats']['B']['mean']:.1f} (Blue- / Yellow+)")
    
    return methods_results, lab_analysis

# Usage
if __name__ == "__main__":
    # Test with your hair image
    image_path = "/home/user/rebrow/images_data/follicules/e7772399-11df-4e79-bca0-f46b5c835530.png"  # 👈 CHANGE THIS
    
    # Option 1: Analyze entire image
    methods, lab_data = run_enhanced_hair_analysis(image_path, n_colors=3) # type: ignore
    
    # Option 2: Focus on specific region to avoid background
    # methods, lab_data = run_enhanced_hair_analysis(image_path, region=(50, 50, 400, 300), n_colors=3)
    
    # Option 3: Extract more colors for detailed analysis
    # methods, lab_data = run_enhanced_hair_analysis(image_path, n_colors=4)