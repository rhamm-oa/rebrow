import cv2
import numpy as np
import os
import pandas as pd
import argparse
from tqdm import tqdm
import sys
import gc

# Import custom modules
from face_detection import FaceDetector
from color_analysis import ColorAnalysis
from facer_segmentation import FacerSegmentation

def process_image(image_path, face_detector, color_analyzer, facer_segmenter):
    """Process a single image and extract dominant colors in LAB format"""
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
    
    # Get eyebrow landmarks from the cropped face, passing crop coordinates
    left_eyebrow, right_eyebrow = face_detector.get_eyebrow_landmarks(image, results, crop_coords)
    
    # If landmarks were not detected, try with the original image
    if left_eyebrow is None or right_eyebrow is None:
        left_eyebrow, right_eyebrow = face_detector.get_eyebrow_landmarks(image, results)
        
        if left_eyebrow is None or right_eyebrow is None:
            print(f"Could not detect eyebrows in {image_path}")
            return None
    
    # Process the cropped face with Facer for eyebrow segmentation
    try:
        # Use Facer segmentation for accurate eyebrow masks
        facer_result = facer_segmenter.segment_eyebrows(face_crop, visualize=False)
        
        if not facer_result.get('success', False):
            print(f"Facer segmentation failed for {image_path}: No success flag")
            return None
            
        # Get the eyebrow masks from Facer
        left_eyebrow_mask = facer_result.get('left_eyebrow_mask')
        right_eyebrow_mask = facer_result.get('right_eyebrow_mask')
        
        if left_eyebrow_mask is None or right_eyebrow_mask is None:
            print(f"Facer segmentation failed for {image_path}: Missing eyebrow mask")
            return None
            
        # Extract colors using the reliable hair colors method with Facer masks
        left_colors, left_percentages, _ = color_analyzer.extract_reliable_hair_colors(face_crop, left_eyebrow_mask, n_colors=3)
        right_colors, right_percentages, _ = color_analyzer.extract_reliable_hair_colors(face_crop, right_eyebrow_mask, n_colors=3)
        
        # Get detailed color information for left eyebrow
        left_color_details_list = []
        if left_colors is not None and left_percentages is not None and len(left_colors) > 0:
            left_color_details_list = color_analyzer.get_color_info(left_colors, left_percentages)
        
        # Get detailed color information for right eyebrow
        right_color_details_list = []
        if right_colors is not None and right_percentages is not None and len(right_colors) > 0:
            right_color_details_list = color_analyzer.get_color_info(right_colors, right_percentages)

        if not left_color_details_list and not right_color_details_list:
            print(f"No valid color details extracted for left or right eyebrows in {image_path}.")
            return None

        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        result_data = {'image_filename': filename}

        # Helper function to populate data for an eyebrow
        def _populate_eyebrow_data_for_csv(color_details_list_param, prefix, data_dict_to_fill):
            for i in range(3): # For up to 3 dominant colors
                if i < len(color_details_list_param):
                    details = color_details_list_param[i]
                    # Ensure 'lab' key exists and is not None, and has 3 components
                    if 'lab' in details and details['lab'] and isinstance(details['lab'], (list, tuple)) and len(details['lab']) == 3:
                        data_dict_to_fill[f'{prefix}_color{i+1}_L'] = details['lab'][0]
                        data_dict_to_fill[f'{prefix}_color{i+1}_a'] = details['lab'][1]
                        data_dict_to_fill[f'{prefix}_color{i+1}_b'] = details['lab'][2]
                    else:
                        data_dict_to_fill[f'{prefix}_color{i+1}_L'] = np.nan
                        data_dict_to_fill[f'{prefix}_color{i+1}_a'] = np.nan
                        data_dict_to_fill[f'{prefix}_color{i+1}_b'] = np.nan
                    
                    percentage_str = details.get('percentage') # Safely get percentage
                    if percentage_str:
                        try:
                            data_dict_to_fill[f'{prefix}_color{i+1}_percentage'] = float(percentage_str.strip('%'))
                        except (ValueError, TypeError):
                            data_dict_to_fill[f'{prefix}_color{i+1}_percentage'] = np.nan
                    else: # If percentage key is missing or value is None/empty
                        data_dict_to_fill[f'{prefix}_color{i+1}_percentage'] = np.nan
                else: # If fewer than 3 colors, fill with NaN
                    data_dict_to_fill[f'{prefix}_color{i+1}_L'] = np.nan
                    data_dict_to_fill[f'{prefix}_color{i+1}_a'] = np.nan
                    data_dict_to_fill[f'{prefix}_color{i+1}_b'] = np.nan
                    data_dict_to_fill[f'{prefix}_color{i+1}_percentage'] = np.nan
        
        _populate_eyebrow_data_for_csv(left_color_details_list, 'left_eyebrow', result_data)
        _populate_eyebrow_data_for_csv(right_color_details_list, 'right_eyebrow', result_data)
        
        return result_data
        
    except Exception as e:
        print(f"Facer segmentation or subsequent color processing failed for {image_path}: {e}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process images and extract dominant eyebrow colors')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for processing if available')
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    # Initialize modules
    print("Initializing modules...")
    face_detector = FaceDetector(use_gpu=args.use_gpu)
    color_analyzer = ColorAnalysis()
    
    # Initialize Facer segmentation (may take a moment to load models)
    print("Initializing Facer segmentation (this may take a moment)...")
    facer_segmenter = FacerSegmentation(use_gpu=args.use_gpu)
    
    # Get all image files in the input directory
    image_files = [f for f in os.listdir(args.input_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No image files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image files")
    
    # Process each image
    results = []
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(args.input_dir, image_file)
        result = process_image(image_path, face_detector, color_analyzer, facer_segmenter)
        if result:
            results.append(result)
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    print(f"Successfully processed {len(results)} out of {len(image_files)} images")
    
    # Clean up GPU memory if used
    if args.use_gpu:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("GPU memory cleared")
        except ImportError:
            pass

if __name__ == "__main__":
    main()
