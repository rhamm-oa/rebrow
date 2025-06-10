# pylint: disable=no-member
# pyright: reportOperatorIssue=false, reportArgumentType=false

"""
Ultra-Robust Eyebrow Analysis Script with Advanced Fallback Methods
Usage: python eyebrow_debug_analysis.py <image_path> [--output-type single|individual|pdf|all]
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import argparse
from matplotlib.backends.backend_pdf import PdfPages
from skimage import filters, morphology, segmentation
from skimage.feature import local_binary_pattern
from scipy import ndimage

# Import your existing modules
from face_detection import FaceDetector
from eyebrow_segmentation import EyebrowSegmentation
from color_analysis import ColorAnalysis
from facer_segmentation import FacerSegmentation

def load_image(image_path):
    """Load image from path"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return image

def create_color_palette_array(colors, percentages, height=100, width=400):
    """Helper function to create color palette array"""
    if colors is None or percentages is None:
        return None
    
    palette = np.zeros((height, width, 3), dtype=np.uint8)
    start_x = 0
    for color, percentage in zip(colors, percentages):
        width_segment = int(width * (percentage / 100))
        end_x = min(start_x + width_segment, width)
        if end_x > start_x:
            palette[:, start_x:end_x] = color
        start_x = end_x
    return palette

def calculate_method_quality(method_mask, gray_image, original_mask):
    """Calculate quality score for a detection method"""
    if np.sum(method_mask) == 0:
        return 0
    
    # Factor 1: Reasonable pixel count (not too few, not too many)
    pixel_ratio = np.sum(method_mask) / np.sum(original_mask)
    if pixel_ratio < 0.05:  # Less than 5%
        count_score = pixel_ratio * 100  # Low score for too few
    elif pixel_ratio > 0.4:  # More than 40%
        count_score = max(0, 100 - (pixel_ratio - 0.4) * 200)  # Penalty for too many
    else:
        count_score = 100
    
    # Factor 2: Detected pixels should be darker than average
    detected_pixels = gray_image[method_mask > 0]
    all_pixels = gray_image[original_mask > 0]
    
    avg_detected = np.mean(detected_pixels)
    avg_all = np.mean(all_pixels)
    
    darkness_score = max(0, min(100, (avg_all - avg_detected) * 2))
    
    # Factor 3: Spatial coherence (connected components)
    num_components, _ = cv2.connectedComponents(method_mask)
    if num_components <= 3:
        coherence_score = 100
    else:
        coherence_score = max(0, 100 - (num_components - 3) * 10)
    
    # Weighted average
    quality_score = (count_score * 0.4 + darkness_score * 0.4 + coherence_score * 0.2)
    
    return quality_score

def extract_hair_with_robust_fallbacks(image, mask):
    """
    🆕 ULTRA-ROBUST: Multiple fallback strategies for difficult images
    """
    methods_results = {}
    
    if mask is None or np.sum(mask) == 0:
        return methods_results
    
    # Apply mask to image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
    
    print("🔬 Running robust hair detection with fallbacks...")
    
    # 🎯 METHOD 1: ENHANCED HSV (with failure detection)
    try:
        h, s, v = cv2.split(hsv)
        masked_v = v[mask > 0]
        masked_s = s[mask > 0]
        
        if len(masked_v) > 0:
            v_range = np.max(masked_v) - np.min(masked_v)
            s_range = np.max(masked_s) - np.min(masked_s)
            
            # Check if HSV will be effective
            if v_range > 30 and s_range > 15:  # Sufficient contrast
                v_mean = np.mean(masked_v)
                if v_mean > 120:  # Light hair
                    v_threshold = np.percentile(masked_v, 45)
                    s_threshold = 12
                elif v_mean > 80:  # Medium hair  
                    v_threshold = np.percentile(masked_v, 35)
                    s_threshold = 10
                else:  # Dark hair
                    v_threshold = np.percentile(masked_v, 25)
                    s_threshold = 8
                
                hsv_mask = np.zeros_like(mask)
                hsv_mask[(v < v_threshold) & (s > s_threshold) & (mask > 0)] = 255
                
                # Clean up
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)
                
                if np.sum(hsv_mask) > 20:  # Only add if successful
                    methods_results['hsv_method'] = {
                        'mask': hsv_mask,
                        'name': 'HSV Enhanced',
                        'pixel_count': np.sum(hsv_mask),
                        'description': f'HSV V<{v_threshold:.0f} + S>{s_threshold}',
                        'quality_score': calculate_method_quality(hsv_mask, gray, mask),
                        'success': True
                    }
                else:
                    methods_results['hsv_method'] = {'success': False, 'reason': 'Insufficient pixels detected'}
            else:
                methods_results['hsv_method'] = {'success': False, 'reason': f'Low contrast: V_range={v_range}, S_range={s_range}'}
        else:
            methods_results['hsv_method'] = {'success': False, 'reason': 'No pixels in mask'}
    except Exception as e:
        methods_results['hsv_method'] = {'success': False, 'reason': f'HSV error: {e}'}
    
    # 🎯 METHOD 2: ROBUST LAB (with failure detection)
    try:
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        masked_l = l_channel[mask > 0]
        
        if len(masked_l) > 0:
            l_range = np.max(masked_l) - np.min(masked_l)
            
            if l_range > 25:  # Sufficient lightness contrast
                l_mean = np.mean(masked_l)
                l_10th = np.percentile(masked_l, 10)
                
                if l_10th < 35:  # Dark hair present
                    l_threshold = np.percentile(masked_l, 30)
                elif l_mean < 90:  # Medium hair
                    l_threshold = np.percentile(masked_l, 40)
                else:  # Light hair
                    l_threshold = np.percentile(masked_l, 50)
                
                lab_mask = np.zeros_like(mask)
                lab_mask[(l_channel < l_threshold) & (mask > 0)] = 255
                
                if np.sum(lab_mask) > 20:
                    methods_results['lab_method'] = {
                        'mask': lab_mask,
                        'name': 'LAB Lightness',
                        'pixel_count': np.sum(lab_mask),
                        'description': f'LAB L<{l_threshold:.1f}',
                        'quality_score': calculate_method_quality(lab_mask, gray, mask),
                        'success': True
                    }
                else:
                    methods_results['lab_method'] = {'success': False, 'reason': 'LAB threshold too restrictive'}
            else:
                methods_results['lab_method'] = {'success': False, 'reason': f'Low L contrast: {l_range}'}
        else:
            methods_results['lab_method'] = {'success': False, 'reason': 'No pixels in mask'}
    except Exception as e:
        methods_results['lab_method'] = {'success': False, 'reason': f'LAB error: {e}'}
    
    # 🎯 METHOD 3: ENHANCED EDGE DETECTION
    try:
        bilateral = cv2.bilateralFilter(gray, 9, 80, 80)
        edges = cv2.Canny(bilateral, 20, 60)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Connect hair strands
        kernel_line_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 1))
        kernel_line_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line_h)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line_v)
        
        if np.sum(edges) > 15:
            methods_results['edge_method'] = {
                'mask': edges,
                'name': 'Hair Edge Detection',
                'pixel_count': np.sum(edges),
                'description': 'Canny edges + strand connection',
                'quality_score': calculate_method_quality(edges, gray, mask),
                'success': True
            }
        else:
            methods_results['edge_method'] = {'success': False, 'reason': 'No significant edges detected'}
    except Exception as e:
        methods_results['edge_method'] = {'success': False, 'reason': f'Edge error: {e}'}
    
    # 🎯 METHOD 4: GABOR FILTER BANK
    try:
        gabor_response = np.zeros_like(gray, dtype=np.float32)
        hair_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
        
        for angle in hair_angles:
            theta = np.radians(angle)
            kernel = cv2.getGaborKernel((11, 11), 3, theta, 6, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            gabor_response += np.abs(filtered)
        
        gabor_response = cv2.normalize(gabor_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # type: ignore
        gabor_threshold = np.percentile(gabor_response[mask > 0], 70) if np.sum(mask) > 0 else 50
        gabor_mask = np.zeros_like(mask)
        gabor_mask[(gabor_response > gabor_threshold) & (mask > 0)] = 255
        
        if np.sum(gabor_mask) > 20:
            methods_results['gabor_method'] = {
                'mask': gabor_mask,
                'name': 'Gabor Hair Texture',
                'pixel_count': np.sum(gabor_mask),
                'description': f'12-direction Gabor bank>{gabor_threshold}',
                'quality_score': calculate_method_quality(gabor_mask, gray, mask),
                'response_image': gabor_response,
                'success': True
            }
        else:
            methods_results['gabor_method'] = {'success': False, 'reason': 'Gabor filters detected insufficient texture'}
    except Exception as e:
        methods_results['gabor_method'] = {'success': False, 'reason': f'Gabor error: {e}'}
    
    # 🎯 METHOD 5: TEXTURE VARIANCE
    try:
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        gray_float = gray.astype(np.float32)
        mean_img = cv2.filter2D(gray_float, -1, kernel)
        sqr_mean_img = cv2.filter2D(gray_float * gray_float, -1, kernel)
        variance = np.maximum(0.0, sqr_mean_img - mean_img * mean_img)
        
        variance_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # type: ignore
        texture_threshold = np.percentile(variance_norm[mask > 0], 60) if np.sum(mask) > 0 else 50
        texture_mask = np.zeros_like(mask)
        texture_mask[(variance_norm > texture_threshold) & (mask > 0)] = 255
        
        if np.sum(texture_mask) > 20:
            methods_results['texture_method'] = {
                'mask': texture_mask,
                'name': 'Texture Variance',
                'pixel_count': np.sum(texture_mask),
                'description': f'Local variance>{texture_threshold}',
                'quality_score': calculate_method_quality(texture_mask, gray, mask),
                'success': True
            }
        else:
            methods_results['texture_method'] = {'success': False, 'reason': 'Insufficient texture variation'}
    except Exception as e:
        methods_results['texture_method'] = {'success': False, 'reason': f'Texture error: {e}'}
    
    # 🎯 METHOD 6: MORPHOLOGICAL TOP-HAT
    try:
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_tophat)
        tophat_threshold = np.percentile(tophat[mask > 0], 75) if np.sum(mask) > 0 else 20 # type: ignore
        tophat_mask = np.zeros_like(mask)
        tophat_mask[(tophat > tophat_threshold) & (mask > 0)] = 255
        
        if np.sum(tophat_mask) > 15:
            methods_results['tophat_method'] = {
                'mask': tophat_mask,
                'name': 'Dark Hair Strands',
                'pixel_count': np.sum(tophat_mask),
                'description': f'Black top-hat>{tophat_threshold}',
                'quality_score': calculate_method_quality(tophat_mask, gray, mask),
                'response_image': tophat,
                'success': True
            }
        else:
            methods_results['tophat_method'] = {'success': False, 'reason': 'No dark strands detected'}
    except Exception as e:
        methods_results['tophat_method'] = {'success': False, 'reason': f'Top-hat error: {e}'}
    
    # 🎯 FALLBACK METHOD 1: STATISTICAL OUTLIER DETECTION
    try:
        masked_pixels = gray[mask > 0]
        if len(masked_pixels) > 50:
            mean_intensity = np.mean(masked_pixels)
            std_intensity = np.std(masked_pixels)
            
            outlier_threshold = mean_intensity - (1.5 * std_intensity)
            
            outlier_mask = np.zeros_like(mask)
            outlier_mask[(gray < outlier_threshold) & (mask > 0)] = 255
            
            if np.sum(outlier_mask) > 15:
                methods_results['outlier_method'] = {
                    'mask': outlier_mask,
                    'name': 'Statistical Outliers',
                    'pixel_count': np.sum(outlier_mask),
                    'description': f'Pixels < mean-1.5*std ({outlier_threshold:.0f})',
                    'quality_score': calculate_method_quality(outlier_mask, gray, mask),
                    'success': True
                }
            else:
                methods_results['outlier_method'] = {'success': False, 'reason': 'No statistical outliers found'}
        else:
            methods_results['outlier_method'] = {'success': False, 'reason': 'Insufficient pixels for statistics'}
    except Exception as e:
        methods_results['outlier_method'] = {'success': False, 'reason': f'Outlier error: {e}'}
    
    # 🎯 FALLBACK METHOD 2: SIMPLE PERCENTILE THRESHOLDING
    try:
        masked_pixels = gray[mask > 0]
        if len(masked_pixels) > 20:
            percentile_threshold = np.percentile(masked_pixels, 25)  # Darkest 25%
            
            percentile_mask = np.zeros_like(mask)
            percentile_mask[(gray < percentile_threshold) & (mask > 0)] = 255
            
            if np.sum(percentile_mask) > 10:
                methods_results['percentile_method'] = {
                    'mask': percentile_mask,
                    'name': 'Darkest 25%',
                    'pixel_count': np.sum(percentile_mask),
                    'description': f'Bottom 25th percentile < {percentile_threshold:.0f}',
                    'quality_score': calculate_method_quality(percentile_mask, gray, mask),
                    'success': True
                }
            else:
                methods_results['percentile_method'] = {'success': False, 'reason': 'Percentile method failed'}
        else:
            methods_results['percentile_method'] = {'success': False, 'reason': 'Insufficient pixels'}
    except Exception as e:
        methods_results['percentile_method'] = {'success': False, 'reason': f'Percentile error: {e}'}
    
    # 🎯 FALLBACK METHOD 3: EROSION-BASED DETECTION
    try:
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
        
        if np.sum(eroded_mask) > 50:
            eroded_pixels = gray[eroded_mask > 0]
            erosion_threshold = np.percentile(eroded_pixels, 60)
            
            erosion_mask = np.zeros_like(mask)
            erosion_mask[(gray < erosion_threshold) & (eroded_mask > 0)] = 255
            
            kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            erosion_mask = cv2.dilate(erosion_mask, kernel_expand, iterations=1)
            erosion_mask = cv2.bitwise_and(erosion_mask, mask)
            
            if np.sum(erosion_mask) > 10:
                methods_results['erosion_method'] = {
                    'mask': erosion_mask,
                    'name': 'Core Region Focus',
                    'pixel_count': np.sum(erosion_mask),
                    'description': f'Core region < {erosion_threshold:.0f}',
                    'quality_score': calculate_method_quality(erosion_mask, gray, mask),
                    'success': True
                }
            else:
                methods_results['erosion_method'] = {'success': False, 'reason': 'Core region detection failed'}
        else:
            methods_results['erosion_method'] = {'success': False, 'reason': 'Mask too small after erosion'}
    except Exception as e:
        methods_results['erosion_method'] = {'success': False, 'reason': f'Erosion error: {e}'}
    
    # 🎯 LAST RESORT: MINIMAL DETECTION
    try:
        masked_pixels = gray[mask > 0]
        if len(masked_pixels) > 10:
            min_threshold = np.percentile(masked_pixels, 5)  # Bottom 5%
            
            minimal_mask = np.zeros_like(mask)
            minimal_mask[(gray < min_threshold) & (mask > 0)] = 255
            
            if np.sum(minimal_mask) >= 5:
                methods_results['minimal_method'] = {
                    'mask': minimal_mask,
                    'name': 'Minimal Detection',
                    'pixel_count': np.sum(minimal_mask),
                    'description': f'Bottom 5% pixels < {min_threshold:.0f}',
                    'quality_score': max(10, np.sum(minimal_mask)),
                    'success': True
                }
            else:
                methods_results['minimal_method'] = {'success': False, 'reason': 'Even minimal detection failed'}
        else:
            methods_results['minimal_method'] = {'success': False, 'reason': 'No pixels available'}
    except Exception as e:
        methods_results['minimal_method'] = {'success': False, 'reason': f'Minimal error: {e}'}
    
    # 🎯 INTELLIGENT COMBINATION (only use successful methods)
    successful_methods = {k: v for k, v in methods_results.items() if v.get('success', False)}
    
    if len(successful_methods) >= 1:
        # Sort successful methods by quality
        sorted_successful = sorted(successful_methods.items(), 
                                 key=lambda x: x[1].get('quality_score', 0), reverse=True)
        
        if len(sorted_successful) == 1:
            # Only one method worked, use it
            best_method = sorted_successful[0]
            combined_mask = best_method[1]['mask'].copy()
            strategy = f"Only {best_method[1]['name']} succeeded"
        else:
            # Multiple methods worked, combine them intelligently
            best_mask = sorted_successful[0][1]['mask']
            second_mask = sorted_successful[1][1]['mask']
            
            # Use union but weight towards better method
            combined_mask = cv2.bitwise_or(best_mask, second_mask)
            strategy = f"Union of {sorted_successful[0][1]['name']} + {sorted_successful[1][1]['name']}"
        
        # Final cleanup
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_clean)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_clean)
        
        methods_results['intelligent_combination'] = {
            'mask': combined_mask,
            'name': 'Intelligent Combination',
            'pixel_count': np.sum(combined_mask),
            'description': strategy,
            'quality_score': calculate_method_quality(combined_mask, gray, mask),
            'success': True
        }
    else:
        # ABSOLUTE LAST RESORT: Return a minimal mask in center of eyebrow
        print("⚠️ ALL METHODS FAILED - Creating emergency fallback")
        center_y, center_x = np.where(mask > 0)
        if len(center_y) > 0:
            cy, cx = int(np.mean(center_y)), int(np.mean(center_x))
            emergency_mask = np.zeros_like(mask)
            
            # Create a small circle in the center
            for y in range(max(0, cy-3), min(mask.shape[0], cy+4)):
                for x in range(max(0, cx-5), min(mask.shape[1], cx+6)):
                    if mask[y, x] > 0:
                        emergency_mask[y, x] = 255
            
            methods_results['emergency_fallback'] = {
                'mask': emergency_mask,
                'name': 'Emergency Fallback',
                'pixel_count': np.sum(emergency_mask),
                'description': 'Center region when all methods fail',
                'quality_score': 5,
                'success': True
            }
    
    # Report what happened
    successful_count = sum(1 for v in methods_results.values() if v.get('success', False))
    failed_count = len(methods_results) - successful_count
    
    print(f"✅ Robust analysis complete: {successful_count} succeeded, {failed_count} failed")
    
    return methods_results

def extract_colors_from_individual_methods(image, mask, methods_results, n_colors=2):
    """
    Extract colors from each individual method (only successful ones)
    """
    color_results = {}
    color_analyzer = ColorAnalysis()
    
    print("🎨 Extracting colors from successful methods...")
    
    for method_name, method_data in methods_results.items():
        if not method_data.get('success', False):
            color_results[method_name] = {
                'colors': None,
                'percentages': None,
                'palette': None,
                'status': f"Method failed: {method_data.get('reason', 'Unknown')}",
                'method_info': method_data
            }
            continue
        
        method_mask = method_data['mask']
        
        if method_mask is None or np.sum(method_mask) < 5:
            color_results[method_name] = {
                'colors': None,
                'percentages': None,
                'palette': None,
                'status': f'Insufficient pixels: {np.sum(method_mask)}',
                'method_info': method_data
            }
            continue
        
        try:
            # Extract colors using the method's mask
            colors, percentages = color_analyzer.extract_colors_from_hair_mask(image, method_mask, n_colors)
            
            if colors is not None:
                palette = create_color_palette_array(colors, percentages, height=60, width=300)
                color_results[method_name] = {
                    'colors': colors,
                    'percentages': percentages,
                    'palette': palette,
                    'status': 'Success',
                    'method_info': method_data
                }
            else:
                color_results[method_name] = {
                    'colors': None,
                    'percentages': None,
                    'palette': None,
                    'status': 'Color extraction failed',
                    'method_info': method_data
                }
        except Exception as e:
            color_results[method_name] = {
                'colors': None,
                'percentages': None,
                'palette': None,
                'status': f'Error: {str(e)}',
                'method_info': method_data
            }
    
    successful_extractions = sum(1 for result in color_results.values() if result['status'] == 'Success')
    print(f"✅ Color extraction completed: {successful_extractions}/{len(color_results)} successful")
    
    return color_results

def create_enhanced_debug_grid(face_crop, left_methods, right_methods, 
                              left_color_results, right_color_results,
                              output_dir, filename):
    """
    Create enhanced debug grid with clean LAB values under color palettes
    """
    # Get all method names (both successful and failed)
    all_methods = set(left_methods.keys()) | set(right_methods.keys())
    method_list = sorted(list(all_methods))
    
    # Create figure - back to 5 columns since we're putting LAB under palette
    fig = plt.figure(figsize=(24, max(16, len(method_list) * 2.5)))
    fig.suptitle(f'Ultra-Robust Eyebrow Hair Detection Analysis - {filename}', fontsize=18, fontweight='bold')
    
    rows = len(method_list) + 2
    cols = 5  # Face, Left mask, Right mask, Method info, Color palette + LAB
    
    # Show original face crop
    ax_face = plt.subplot(rows, cols, (1, 3))  # Span 3 columns
    if face_crop is not None:
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        ax_face.imshow(face_rgb)
    ax_face.set_title('Face Crop', fontweight='bold', fontsize=14)
    ax_face.axis('off')
    
    # Summary info
    ax_summary = plt.subplot(rows, cols, (4, 5))  # Span 2 columns
    left_successful = sum(1 for v in left_methods.values() if v.get('success', False))
    right_successful = sum(1 for v in right_methods.values() if v.get('success', False))
    
    summary_text = f"""DETECTION SUMMARY
Left Eyebrow: {left_successful}/{len(left_methods)} methods succeeded
Right Eyebrow: {right_successful}/{len(right_methods)} methods succeeded

LAB values displayed under color palettes
Success/Failure reasons in method details"""
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
    ax_summary.set_title('Analysis Summary', fontweight='bold', fontsize=12)
    ax_summary.axis('off')
    
    # Show each method
    for i, method_key in enumerate(method_list):
        row = i + 2
        
        # Left eyebrow mask
        ax_left = plt.subplot(rows, cols, row * cols + 1)
        left_method = left_methods.get(method_key, {})
        
        if left_method.get('success', False):
            mask = left_method['mask']
            if np.sum(mask) > 0:
                # Focus on eyebrow region
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) > 0:
                    y_min, y_max = max(0, np.min(y_indices)-5), min(mask.shape[0], np.max(y_indices)+5)
                    x_min, x_max = max(0, np.min(x_indices)-5), min(mask.shape[1], np.max(x_indices)+5)
                    
                    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                    mask_rgb[mask > 0] = [0, 255, 0]  # Green for successful detection
                    
                    cropped_mask = mask_rgb[y_min:y_max, x_min:x_max]
                    ax_left.imshow(cropped_mask)
                else:
                    ax_left.text(0.5, 0.5, '✅ Success\n(No pixels)', ha='center', va='center', 
                                transform=ax_left.transAxes, color='green')
            else:
                ax_left.text(0.5, 0.5, '✅ Success\n(No pixels)', ha='center', va='center', 
                            transform=ax_left.transAxes, color='green')
        else:
            ax_left.text(0.5, 0.5, f'❌ Failed\n{left_method.get("reason", "Unknown")}', 
                        ha='center', va='center', transform=ax_left.transAxes, 
                        color='red', fontsize=8)
        
        method_name = left_method.get('name', method_key.replace('_', ' ').title())
        ax_left.set_title(f'Left: {method_name}', fontsize=10)
        ax_left.axis('off')
        
        # Right eyebrow mask
        ax_right = plt.subplot(rows, cols, row * cols + 2)
        right_method = right_methods.get(method_key, {})
        
        if right_method.get('success', False):
            mask = right_method['mask']
            if np.sum(mask) > 0:
                # Focus on eyebrow region
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) > 0:
                    y_min, y_max = max(0, np.min(y_indices)-5), min(mask.shape[0], np.max(y_indices)+5)
                    x_min, x_max = max(0, np.min(x_indices)-5), min(mask.shape[1], np.max(x_indices)+5)
                    
                    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                    mask_rgb[mask > 0] = [0, 255, 0]  # Green for successful detection
                    
                    cropped_mask = mask_rgb[y_min:y_max, x_min:x_max]
                    ax_right.imshow(cropped_mask)
                else:
                    ax_right.text(0.5, 0.5, '✅ Success\n(No pixels)', ha='center', va='center', 
                                 transform=ax_right.transAxes, color='green')
            else:
                ax_right.text(0.5, 0.5, '✅ Success\n(No pixels)', ha='center', va='center', 
                             transform=ax_right.transAxes, color='green')
        else:
            ax_right.text(0.5, 0.5, f'❌ Failed\n{right_method.get("reason", "Unknown")}', 
                         ha='center', va='center', transform=ax_right.transAxes, 
                         color='red', fontsize=8)
        
        method_name = right_method.get('name', method_key.replace('_', ' ').title())
        ax_right.set_title(f'Right: {method_name}', fontsize=10)
        ax_right.axis('off')
        
        # Method info (cleaned up - removed quality scores)
        ax_info = plt.subplot(rows, cols, row * cols + 3)
        info_text = []
        
        if left_method.get('success', False) or right_method.get('success', False):
            method_info = left_method if left_method.get('success', False) else right_method
            info_text.append(f"Method: {method_info.get('name', 'Unknown')}")
            info_text.append(f"Description: {method_info.get('description', 'N/A')}")
            info_text.append("")
            info_text.append(f"Detection Results:")
            info_text.append(f"Left:  {'✅' if left_method.get('success', False) else '❌'} {left_method.get('pixel_count', 0)} pixels")
            info_text.append(f"Right: {'✅' if right_method.get('success', False) else '❌'} {right_method.get('pixel_count', 0)} pixels")
            
        else:
            info_text.append(f"Method: {method_key.replace('_', ' ').title()}")
            info_text.append("Status: FAILED on both sides")
            info_text.append("")
            info_text.append("Failure reasons:")
            info_text.append(f"Left: {left_method.get('reason', 'Unknown')}")
            info_text.append(f"Right: {right_method.get('reason', 'Unknown')}")
        
        ax_info.text(0.05, 0.95, '\n'.join(info_text), transform=ax_info.transAxes, 
                    fontsize=8, verticalalignment='top', fontfamily='monospace')
        ax_info.set_title('Method Details', fontsize=10)
        ax_info.axis('off')
        
        # 🆕 Color palette with LAB values underneath
        ax_palette = plt.subplot(rows, cols, row * cols + 4)
        
        left_result = left_color_results.get(method_key, {})
        right_result = right_color_results.get(method_key, {})
        
        # Prepare LAB text
        lab_text = []
        
        if (left_result.get('status') == 'Success' and left_result.get('palette') is not None and
            right_result.get('status') == 'Success' and right_result.get('palette') is not None):
            # Show both palettes
            left_palette = left_result['palette']
            right_palette = right_result['palette']
            combined_palette = np.vstack([left_palette, right_palette])
            ax_palette.imshow(combined_palette)
            ax_palette.set_title('Colors (Top:L, Bot:R)', fontsize=9)
            
            # LAB for left eyebrow
            lab_text.append("LAB_LEFT:")
            for i, color_bgr in enumerate(left_result['colors']):
                color_rgb = color_bgr[::-1]  # BGR to RGB
                color_lab = cv2.cvtColor(np.uint8([[color_rgb]]), cv2.COLOR_RGB2LAB)[0][0] # type: ignore
                pct = left_result['percentages'][i]
                lab_text.append(f"C{i+1}({pct:.0f}%): L{color_lab[0]} a{color_lab[1]} b{color_lab[2]}")
            
            lab_text.append("")
            
            # LAB for right eyebrow
            lab_text.append("LAB_RIGHT:")
            for i, color_bgr in enumerate(right_result['colors']):
                color_rgb = color_bgr[::-1]  # BGR to RGB
                color_lab = cv2.cvtColor(np.uint8([[color_rgb]]), cv2.COLOR_RGB2LAB)[0][0] # type: ignore # type: ignore
                pct = right_result['percentages'][i]
                lab_text.append(f"C{i+1}({pct:.0f}%): L{color_lab[0]} a{color_lab[1]} b{color_lab[2]}")
                
        elif left_result.get('status') == 'Success' and left_result.get('palette') is not None:
            # Show only left palette
            ax_palette.imshow(left_result['palette'])
            ax_palette.set_title('Left Colors Only', fontsize=9)
            
            lab_text.append("LAB_LEFT:")
            for i, color_bgr in enumerate(left_result['colors']):
                color_rgb = color_bgr[::-1]  # BGR to RGB
                color_lab = cv2.cvtColor(np.uint8([[color_rgb]]), cv2.COLOR_RGB2LAB)[0][0] # type: ignore
                pct = left_result['percentages'][i]
                lab_text.append(f"C{i+1}({pct:.0f}%): L{color_lab[0]} a{color_lab[1]} b{color_lab[2]}")
            
            lab_text.append("")
            lab_text.append("LAB_RIGHT: No data")
            
        elif right_result.get('status') == 'Success' and right_result.get('palette') is not None:
            # Show only right palette
            ax_palette.imshow(right_result['palette'])
            ax_palette.set_title('Right Colors Only', fontsize=9)
            
            lab_text.append("LAB_LEFT: No data")
            lab_text.append("")
            lab_text.append("LAB_RIGHT:")
            for i, color_bgr in enumerate(right_result['colors']):
                color_rgb = color_bgr[::-1]  # BGR to RGB
                color_lab = cv2.cvtColor(np.uint8([[color_rgb]]), cv2.COLOR_RGB2LAB)[0][0] # type: ignore
                pct = right_result['percentages'][i]
                lab_text.append(f"C{i+1}({pct:.0f}%): L{color_lab[0]} a{color_lab[1]} b{color_lab[2]}")
                
        else:
            # Show failure status
            left_status = left_result.get('status', 'Method failed')
            right_status = right_result.get('status', 'Method failed')
            ax_palette.text(0.5, 0.5, f'Color Extraction Failed\nL: {left_status}\nR: {right_status}', 
                           ha='center', va='center', transform=ax_palette.transAxes, fontsize=7)
            ax_palette.set_title('Color Status', fontsize=9)
            
            lab_text.append("LAB_LEFT: Failed")
            lab_text.append("LAB_RIGHT: Failed")
        
        ax_palette.axis('off')
        
        # Add LAB text below the palette
        if lab_text:
            # Position text below the image
            ax_palette.text(0.02, -0.1, '\n'.join(lab_text), transform=ax_palette.transAxes, 
                           fontsize=7, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'{filename}_robust_debug_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Clean debug grid with LAB values saved: {output_path}")
    
    plt.show()
    return output_path

def create_method_success_report(left_methods, right_methods, left_color_results, right_color_results, output_dir, filename):
    """
    Create a clean text report without quality scores
    """
    report_path = os.path.join(output_dir, f'{filename}_method_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write(f"ULTRA-ROBUST EYEBROW ANALYSIS REPORT - {filename}\n")
        f.write("=" * 60 + "\n\n")
        
        # Summary statistics
        left_successful = sum(1 for v in left_methods.values() if v.get('success', False))
        right_successful = sum(1 for v in right_methods.values() if v.get('success', False))
        left_color_successful = sum(1 for v in left_color_results.values() if v.get('status') == 'Success')
        right_color_successful = sum(1 for v in right_color_results.values() if v.get('status') == 'Success')
        
        f.write("SUMMARY:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Left Eyebrow:  {left_successful}/{len(left_methods)} methods succeeded\n")
        f.write(f"Right Eyebrow: {right_successful}/{len(right_methods)} methods succeeded\n")
        f.write(f"Left Colors:   {left_color_successful}/{len(left_color_results)} extractions successful\n")
        f.write(f"Right Colors:  {right_color_successful}/{len(right_color_results)} extractions successful\n\n")
        
        # Detailed method analysis
        all_methods = set(left_methods.keys()) | set(right_methods.keys())
        
        for method_key in sorted(all_methods):
            f.write(f"METHOD: {method_key.upper().replace('_', ' ')}\n")
            f.write("-" * 40 + "\n")
            
            left_method = left_methods.get(method_key, {})
            right_method = right_methods.get(method_key, {})
            
            # Left side analysis
            f.write("LEFT EYEBROW:\n")
            if left_method.get('success', False):
                f.write(f"  Status: ✅ SUCCESS ({left_method.get('pixel_count', 0)} pixels)\n")
                f.write(f"  Description: {left_method.get('description', 'N/A')}\n")
                
                # Color extraction results with LAB values
                left_color = left_color_results.get(method_key, {})
                if left_color.get('status') == 'Success' and left_color.get('colors') is not None:
                    f.write(f"  Colors extracted: {len(left_color['colors'])}\n")
                    for i, (color_bgr, pct) in enumerate(zip(left_color['colors'], left_color['percentages'])):
                        hex_color = f'#{color_bgr[0]:02x}{color_bgr[1]:02x}{color_bgr[2]:02x}'
                        
                        # Convert BGR to LAB
                        color_rgb = color_bgr[::-1]  # BGR to RGB
                        color_lab = cv2.cvtColor(np.uint8([[color_rgb]]), cv2.COLOR_RGB2LAB)[0][0] # type: ignore
                        
                        f.write(f"    Color {i+1} ({pct:.1f}%): {hex_color} | LAB: L{color_lab[0]} a{color_lab[1]} b{color_lab[2]}\n")
                else:
                    f.write(f"  Color extraction: {left_color.get('status', 'Failed')}\n")
            else:
                f.write(f"  Status: ❌ FAILED - {left_method.get('reason', 'Unknown')}\n")
            
            f.write("\n")
            
            # Right side analysis
            f.write("RIGHT EYEBROW:\n")
            if right_method.get('success', False):
                f.write(f"  Status: ✅ SUCCESS ({right_method.get('pixel_count', 0)} pixels)\n")
                f.write(f"  Description: {right_method.get('description', 'N/A')}\n")
                
                # Color extraction results with LAB values
                right_color = right_color_results.get(method_key, {})
                if right_color.get('status') == 'Success' and right_color.get('colors') is not None:
                    f.write(f"  Colors extracted: {len(right_color['colors'])}\n")
                    for i, (color_bgr, pct) in enumerate(zip(right_color['colors'], right_color['percentages'])):
                        hex_color = f'#{color_bgr[0]:02x}{color_bgr[1]:02x}{color_bgr[2]:02x}'
                        
                        # Convert BGR to LAB
                        color_rgb = color_bgr[::-1]  # BGR to RGB
                        color_lab = cv2.cvtColor(np.uint8([[color_rgb]]), cv2.COLOR_RGB2LAB)[0][0] # type: ignore
                        
                        f.write(f"    Color {i+1} ({pct:.1f}%): {hex_color} | LAB: L{color_lab[0]} a{color_lab[1]} b{color_lab[2]}\n")
                else:
                    f.write(f"  Color extraction: {right_color.get('status', 'Failed')}\n")
            else:
                f.write(f"  Status: ❌ FAILED - {right_method.get('reason', 'Unknown')}\n")
            
            f.write("\n" + "="*40 + "\n\n")
        
        # Quick LAB summary
        f.write("QUICK LAB SUMMARY:\n")
        f.write("-" * 20 + "\n")
        
        all_lab_values = []
        
        # Collect all successful LAB values
        for side_name, color_results in [("Left", left_color_results), ("Right", right_color_results)]:
            for method_key, result in color_results.items():
                if result.get('status') == 'Success' and result.get('colors') is not None:
                    for color_bgr in result['colors']:
                        color_rgb = color_bgr[::-1]
                        color_lab = cv2.cvtColor(np.uint8([[color_rgb]]), cv2.COLOR_RGB2LAB)[0][0] # type: ignore
                        all_lab_values.append(color_lab)
        
        if all_lab_values:
            all_lab_values = np.array(all_lab_values)
            f.write(f"Total colors: {len(all_lab_values)}\n")
            f.write(f"L* (lightness): {np.min(all_lab_values[:, 0])}-{np.max(all_lab_values[:, 0])} (avg: {np.mean(all_lab_values[:, 0]):.0f})\n")
            f.write(f"a* (green-red): {np.min(all_lab_values[:, 1])}-{np.max(all_lab_values[:, 1])} (avg: {np.mean(all_lab_values[:, 1]):.0f})\n")
            f.write(f"b* (blue-yellow): {np.min(all_lab_values[:, 2])}-{np.max(all_lab_values[:, 2])} (avg: {np.mean(all_lab_values[:, 2]):.0f})\n")
        else:
            f.write("No LAB values available\n")
    
    print(f"📄 Clean method analysis report saved: {report_path}")
    return report_path



def process_single_image(image_path, output_type="single"):
    """
    Process image with ultra-robust methods and comprehensive reporting
    Enhanced with better folder organization
    """
    print(f"Processing: {image_path}")
    
    # Create organized output directory structure
    filename = Path(image_path).stem
    base_output_dir = "debug_output"
    output_dir = os.path.join(base_output_dir, filename)  # Create subfolder with image name
    
    # Ensure the directory structure exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Output directory created: {output_dir}")
    
    # Load image
    image = load_image(image_path)
    
    # Initialize modules
    face_detector = FaceDetector(use_gpu=True)
    
    print("🔍 Detecting face...")
    
    # Detect face landmarks
    results = face_detector.detect_face(image)
    if not results.multi_face_landmarks:
        raise ValueError("No face detected in the image")
    
    # Crop face
    face_crop, crop_coords = face_detector.crop_face(image, results)
    if face_crop is None:
        raise ValueError("Could not crop face properly")
    
    print("✂️ Face cropped successfully")
    
    # Get eyebrow landmarks
    left_eyebrow, right_eyebrow = face_detector.get_eyebrow_landmarks(image, results, crop_coords)
    if left_eyebrow is None or right_eyebrow is None:
        left_eyebrow, right_eyebrow = face_detector.get_eyebrow_landmarks(image, results)
        if left_eyebrow is None or right_eyebrow is None:
            raise ValueError("Could not detect eyebrows")
    
    print("🎯 Eyebrow landmarks detected")
    
    # Use Facer segmentation for initial masks
    print("🧠 Running Facer segmentation...")
    try:
        facer_segmenter = FacerSegmentation(use_gpu=True)
        facer_result = facer_segmenter.segment_eyebrows(face_crop, visualize=True)
        
        if facer_result.get('success', False):
            left_initial_mask = facer_result.get('left_eyebrow_mask')
            right_initial_mask = facer_result.get('right_eyebrow_mask')
            print("✅ Facer segmentation successful")
        else:
            raise Exception("Facer segmentation failed")
            
    except Exception as e:
        print(f"⚠️ Facer segmentation failed: {e}. Using traditional method...")
        # Fallback to traditional method
        eyebrow_segmentation = EyebrowSegmentation()
        left_mask, _ = eyebrow_segmentation.create_eyebrow_mask(face_crop, left_eyebrow)
        right_mask, _ = eyebrow_segmentation.create_eyebrow_mask(face_crop, right_eyebrow)
        left_initial_mask = eyebrow_segmentation.refine_eyebrow_mask(face_crop, left_mask)
        right_initial_mask = eyebrow_segmentation.refine_eyebrow_mask(face_crop, right_mask)
    
    # Run ultra-robust hair detection methods
    print("🔬 Running ultra-robust hair detection with fallbacks...")
    left_methods = extract_hair_with_robust_fallbacks(face_crop, left_initial_mask)
    right_methods = extract_hair_with_robust_fallbacks(face_crop, right_initial_mask)
    
    # Extract colors from all successful methods
    print("🎨 Extracting colors from all successful methods...")
    left_color_results = extract_colors_from_individual_methods(face_crop, left_initial_mask, left_methods, n_colors=2)
    right_color_results = extract_colors_from_individual_methods(face_crop, right_initial_mask, right_methods, n_colors=2)
    
    # Generate outputs
    if output_type in ["single", "all"]:
        create_enhanced_debug_grid(face_crop, left_methods, right_methods,
                                  left_color_results, right_color_results,
                                  output_dir, filename)
        
        create_method_success_report(left_methods, right_methods, 
                                   left_color_results, right_color_results,
                                   output_dir, filename)
    
    print("📊 Ultra-robust analysis complete!")
    
    return {
        'face_crop': face_crop,
        'left_methods': left_methods,
        'right_methods': right_methods,
        'left_color_results': left_color_results,
        'right_color_results': right_color_results,
        'output_dir': output_dir
    }

def main():
    parser = argparse.ArgumentParser(description='Ultra-Robust Eyebrow Analysis with Advanced Fallback Methods')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--output-type', choices=['single', 'individual', 'pdf', 'all'], 
                        default='single', help='Type of output to generate')
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting ultra-robust eyebrow analysis...")
        results = process_single_image(args.image_path, args.output_type)
        
        print("\n" + "="*60)
        print("ULTRA-ROBUST ANALYSIS COMPLETE!")
        print("="*60)
        
        # Print method success summary
        print(f"\n📊 METHOD SUCCESS SUMMARY:")
        print("Left Eyebrow Methods:")
        for method_name, method_data in results['left_methods'].items():
            status = "✅" if method_data.get('success', False) else "❌"
            reason = f" ({method_data.get('reason', 'Unknown')})" if not method_data.get('success', False) else ""
            pixels = method_data.get('pixel_count', 0)
            quality = method_data.get('quality_score', 0)
            print(f"  {status} {method_data.get('name', method_name)}: {pixels} pixels, Q={quality:.1f}{reason}")
        
        print("\nRight Eyebrow Methods:")
        for method_name, method_data in results['right_methods'].items():
            status = "✅" if method_data.get('success', False) else "❌"
            reason = f" ({method_data.get('reason', 'Unknown')})" if not method_data.get('success', False) else ""
            pixels = method_data.get('pixel_count', 0)
            quality = method_data.get('quality_score', 0)
            print(f"  {status} {method_data.get('name', method_name)}: {pixels} pixels, Q={quality:.1f}{reason}")
        
        # Print color extraction summary
        print(f"\n🎨 COLOR EXTRACTION SUMMARY:")
        left_successful = sum(1 for v in results['left_color_results'].values() if v.get('status') == 'Success')
        right_successful = sum(1 for v in results['right_color_results'].values() if v.get('status') == 'Success')
        print(f"Left Eyebrow: {left_successful}/{len(results['left_color_results'])} successful")
        print(f"Right Eyebrow: {right_successful}/{len(results['right_color_results'])} successful")
        
        print(f"\n📁 Output files saved in: {results['output_dir']}/")
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()