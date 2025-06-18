"""
Batch Robust Color Analysis Script
Processes multiple images using the new robust color analysis methods
Generates CSV with best method results and dominant colors in LAB format
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse
import time
import gc
import random
from tqdm import tqdm
from datetime import datetime
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the same RGB to LAB conversion function as in main.py
def convert_rgb_to_lab_proper(rgb_color):
    """Convert RGB to proper LAB values with correct scaling"""
    r, g, b = rgb_color
    rgb_color_obj = sRGBColor(r/255.0, g/255.0, b/255.0)
    lab_color_obj = convert_color(rgb_color_obj, LabColor)
    
    return {
        'L': round(lab_color_obj.lab_l, 1), # type: ignore
        'a': round(lab_color_obj.lab_a, 1), # type: ignore
        'b': round(lab_color_obj.lab_b, 1) # type: ignore
    }

# Import custom modules
from rebrow_modules.face_detection import FaceDetector
from rebrow_modules.color_analysis import ColorAnalysis
from rebrow_modules.facer_segmentation import FacerSegmentation
from rebrow_modules.eyebrow_segmentation import EyebrowSegmentation

def process_image(image_path, face_detector, color_analyzer, facer_segmenter, eyebrow_segmentation, k_clusters=2):
    """Process a single image and extract dominant colors using robust analysis"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None
        
        # Detect face landmarks
        results = face_detector.detect_face(image)
        
        if not results.multi_face_landmarks:
            print(f"No face detected in {image_path}")
            return None
        
        # Crop face
        face_crop, crop_coords = face_detector.crop_face(image, results)
        
        if face_crop is None or crop_coords is None:
            print(f"Could not crop face properly in {image_path}")
            return None
        
        # Get face dimensions
        x_min, y_min, x_max, y_max = crop_coords
        
        # Run landmark detection on the cropped face for better accuracy
        face_crop_results = face_detector.detect_face(face_crop)
        
        if not face_crop_results.multi_face_landmarks:
            print(f"Could not detect facial landmarks on the cropped face in {image_path}")
            return None
        
        # Get eyebrow landmarks from the cropped face
        left_eyebrow, right_eyebrow = face_detector.get_eyebrow_landmarks(image, results, crop_coords)
        
        if left_eyebrow is None or right_eyebrow is None:
            left_eyebrow, right_eyebrow = face_detector.get_eyebrow_landmarks(image, results)
            
            if left_eyebrow is None or right_eyebrow is None:
                print(f"Could not detect eyebrows in {image_path}")
                return None
        
        # *** PRIMARY METHOD: Use Facer segmentation for masks ***
        try:
            facer_result = facer_segmenter.segment_eyebrows(face_crop, visualize=False)
            
            if facer_result.get('success', False):
                # Use Facer masks as primary masks
                left_refined_mask = facer_result.get('left_eyebrow_mask')
                right_refined_mask = facer_result.get('right_eyebrow_mask')
                using_facer_masks = True
            else:
                # Fallback to traditional method
                left_mask, left_bbox = eyebrow_segmentation.create_eyebrow_mask(face_crop, left_eyebrow)
                right_mask, right_bbox = eyebrow_segmentation.create_eyebrow_mask(face_crop, right_eyebrow)
                left_refined_mask = eyebrow_segmentation.refine_eyebrow_mask(face_crop, left_mask)
                right_refined_mask = eyebrow_segmentation.refine_eyebrow_mask(face_crop, right_mask)
                using_facer_masks = False
                
        except Exception as e:
            print(f"Facer segmentation failed for {image_path}: {e}. Using traditional method as fallback.")
            # Fallback to traditional method
            left_mask, left_bbox = eyebrow_segmentation.create_eyebrow_mask(face_crop, left_eyebrow)
            right_mask, right_bbox = eyebrow_segmentation.create_eyebrow_mask(face_crop, right_eyebrow)
            left_refined_mask = eyebrow_segmentation.refine_eyebrow_mask(face_crop, left_mask)
            right_refined_mask = eyebrow_segmentation.refine_eyebrow_mask(face_crop, right_mask)
            using_facer_masks = False
        
        # *** ROBUST COLOR ANALYSIS with all methods ***
        print(f"Running robust color analysis for {os.path.basename(image_path)}...")
        
        # Run robust analysis on both eyebrows
        left_robust_results = color_analyzer.extract_robust_eyebrow_colors(
            face_crop, left_refined_mask, k_clusters, filename=os.path.basename(image_path))
        
        right_robust_results = color_analyzer.extract_robust_eyebrow_colors(
            face_crop, right_refined_mask, k_clusters, filename=os.path.basename(image_path))
        
        if not left_robust_results or not right_robust_results:
            print(f"Robust color analysis failed for {image_path}")
            return None
        
        # Extract top 3 methods for each eyebrow
        left_color_results = left_robust_results.get('color_results', {})
        right_color_results = right_robust_results.get('color_results', {})
        left_methods_results = left_robust_results.get('methods_results', {})
        right_methods_results = right_robust_results.get('methods_results', {})
        
        # Get top 3 methods by quality score for left eyebrow
        left_top_methods = sorted(left_methods_results.items(), 
                                key=lambda x: x[1].get('quality_score', 0), reverse=True)[:3]
        
        # Get top 3 methods by quality score for right eyebrow
        right_top_methods = sorted(right_methods_results.items(), 
                                 key=lambda x: x[1].get('quality_score', 0), reverse=True)[:3]
        
        # Prepare result data
        result = {
            'image_filename': os.path.basename(image_path),
            'left_top_methods': left_top_methods,
            'right_top_methods': right_top_methods,
            'using_facer_masks': using_facer_masks,
        }
        
        # Extract dominant colors for each top method
        for i, (method, method_result) in enumerate(left_top_methods):
            result[f'left_top_method{i+1}_dominant_colors'] = left_color_results.get(method, {})
        
        for i, (method, method_result) in enumerate(right_top_methods):
            result[f'right_top_method{i+1}_dominant_colors'] = right_color_results.get(method, {})
        
        # Clean up memory
        del image, face_crop, left_robust_results, right_robust_results
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Batch process images for robust eyebrow color analysis')
    parser.add_argument('input_folder', help='Path to folder containing images')
    parser.add_argument('output_csv', help='Path to output CSV file')
    parser.add_argument('--k_clusters', type=int, default=2, help='Number of clusters for color extraction (default: 2)')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process (for testing)')
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize modules
    print("Initializing modules...")
    face_detector = FaceDetector(use_gpu=True)
    color_analyzer = ColorAnalysis()
    facer_segmenter = FacerSegmentation(use_gpu=True)
    eyebrow_segmentation = EyebrowSegmentation()
    
    # Get list of image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = []
    
    for root, dirs, files in os.walk(args.input_folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No image files found in '{args.input_folder}'")
        return
    
    # Limit number of images if specified
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"Found {len(image_files)} images to process")
    print(f"Using K={args.k_clusters} clusters for color extraction")
    
    # Process images
    results = []
    failed_images = []
    
    for image_path in tqdm(image_files, desc="Processing images"): # type: ignore
        result = process_image(image_path, face_detector, color_analyzer, facer_segmenter, eyebrow_segmentation, args.k_clusters)
        
        if result:
            results.append(result)
        else:
            failed_images.append(os.path.basename(image_path))
    
    print(f"\nSuccessfully processed: {len(results)} images")
    print(f"Failed to process: {len(failed_images)} images")
    
    if failed_images:
        print("Failed images:", failed_images[:10])  # Show first 10 failed images
        if len(failed_images) > 10:
            print(f"... and {len(failed_images) - 10} more")
    
    # Convert results to DataFrame
    if results:
        print("Creating CSV file...")
        csv_data = []
        
        for result in results:
            row = {
                'image_filename': result['image_filename'],
                'using_facer_masks': result['using_facer_masks']
            }
            
            # Add top 3 methods for left eyebrow
            left_top_methods = result['left_top_methods']
            for i, (method, method_result) in enumerate(left_top_methods):
                row[f'left_top_method{i+1}'] = method
                dominant_colors = result[f'left_top_method{i+1}_dominant_colors']
                colors_rgb = dominant_colors['colors']
                percentages = dominant_colors['percentages']
                
                # Convert each RGB color to LAB using the same function as main.py
                for j in range(len(colors_rgb)):
                    rgb_color = colors_rgb[j]
                    lab_values = convert_rgb_to_lab_proper(rgb_color)
                    
                    # Store LAB values for each color
                    row[f'left_top_method{i+1}_color{j+1}_L'] = lab_values['L']
                    row[f'left_top_method{i+1}_color{j+1}_a'] = lab_values['a']
                    row[f'left_top_method{i+1}_color{j+1}_b'] = lab_values['b']
                    row[f'left_top_method{i+1}_color{j+1}_percentage'] = float(percentages[j])
                
                row[f'left_top_method{i+1}_quality_score'] = method_result.get('quality_score')
            
            # Add top 3 methods for right eyebrow
            right_top_methods = result['right_top_methods']
            for i, (method, method_result) in enumerate(right_top_methods):
                row[f'right_top_method{i+1}'] = method
                dominant_colors = result[f'right_top_method{i+1}_dominant_colors']
                colors_rgb = dominant_colors['colors']
                percentages = dominant_colors['percentages']
                
                # Convert each RGB color to LAB using the same function as main.py
                for j in range(len(colors_rgb)):
                    rgb_color = colors_rgb[j]
                    lab_values = convert_rgb_to_lab_proper(rgb_color)
                    
                    # Store LAB values for each color
                    row[f'right_top_method{i+1}_color{j+1}_L'] = lab_values['L']
                    row[f'right_top_method{i+1}_color{j+1}_a'] = lab_values['a']
                    row[f'right_top_method{i+1}_color{j+1}_b'] = lab_values['b']
                    row[f'right_top_method{i+1}_color{j+1}_percentage'] = float(percentages[j])
                
                row[f'right_top_method{i+1}_quality_score'] = method_result.get('quality_score')
            
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(args.output_csv, index=False)
        
        print(f"CSV file saved: {args.output_csv}")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        
        # Print column names for reference
        print("\nColumn names:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        # Show sample of the data
        print(f"\nSample data (first 3 rows):")
        print(df.head(3).to_string())
        
        # Save failed images list if any
        if failed_images:
            failed_file = args.output_csv.replace('.csv', '_failed_images.txt')
            with open(failed_file, 'w') as f:
                f.write('\n'.join(failed_images))
            print(f"Failed images list saved: {failed_file}")
        
        # Generate summary statistics
        print(f"\nSummary Statistics:")
        print(f"  • Success rate: {len(results)/(len(results)+len(failed_images))*100:.1f}%")
        
    else:
        print("No images were successfully processed")

if __name__ == "__main__":
    main()
