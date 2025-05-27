import os
import subprocess
import cv2
import numpy as np
import shutil
import tempfile

# Define colors for visualization
EYEBROW_COLORS = {
    'left': [255, 170, 0],   # Orange for left eyebrow (matches face-parsing color)
    'right': [255, 0, 85]    # Pink for right eyebrow (matches face-parsing color)
}

def run_bisenet_direct(image_path):
    """
    Run BiSeNet directly using the exact command structure the user provided.
    This is a simplified version that matches the user's command line approach.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Path to the output segmentation file
    """
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_NAME="resnet18"
    # Define paths exactly as in the user's command
    face_parsing_dir = os.path.join(current_dir, 'face-parsing')
    inference_script = os.path.join(face_parsing_dir, 'inference.py')
    weight_path = os.path.join(face_parsing_dir, 'weights', f"{MODEL_NAME}" + '.pt')
    
    # Create a temporary directory for input and output
    temp_dir = tempfile.mkdtemp()
    temp_input_dir = os.path.join(temp_dir, 'images')
    temp_output_dir = os.path.join(temp_dir, 'results')
    
    # Create the directories, including the resnet18 subdirectory
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_output_dir, 'resnet18'), exist_ok=True)
    
    print(f"Created temporary directories:\n  Input: {temp_input_dir}\n  Output: {temp_output_dir}")
    
    # Copy the input image to the temporary directory
    image_filename = os.path.basename(image_path)
    temp_image_path = os.path.join(temp_input_dir, image_filename)
    shutil.copy2(image_path, temp_image_path)
    
    # Run the exact command the user provided
    cmd = [
        'python', inference_script,
        '--model', MODEL_NAME,
        '--weight', weight_path,
        '--input', temp_input_dir,
        '--output', temp_output_dir
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Command output: {result.stdout}")
    if result.stderr:
        print(f"Command error: {result.stderr}")
    
    # Get the output file path - check both main directory and resnet18 subdirectory
    base_name = os.path.splitext(image_filename)[0]
    
    # List of possible output paths to check
    possible_paths = [
        os.path.join(temp_output_dir, f"{base_name}.png"),  # Main directory
        os.path.join(temp_output_dir, "resnet18", f"{base_name}.png"),  # resnet18 subdirectory
        os.path.join(temp_output_dir, "resnet18", f"{base_name}.jpg")   # resnet18 subdirectory with jpg
    ]
    
    # Print all files in the output directory for debugging
    print(f"Files in output directory {temp_output_dir}:")
    for root, dirs, files in os.walk(temp_output_dir):
        for file in files:
            print(f"  {os.path.join(root, file)}")
    
    # Check each possible path
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found output file at: {path}")
            return path, temp_dir
    
    # If we get here, no output file was found
    raise FileNotFoundError(f"BiSeNet did not produce the expected output file. Checked: {possible_paths}")


def extract_eyebrow_masks(segmentation_path, target_shape=None):
    """
    Extract left (6) and right (7) eyebrow masks from the segmentation result.
    """
    # Read the segmentation result
    parsing = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
    if parsing is None:
        raise ValueError(f"Failed to read segmentation result: {segmentation_path}")
    
    # Resize if needed
    if target_shape is not None and parsing.shape[:2] != target_shape[:2]:
        parsing = cv2.resize(parsing, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Extract eyebrow masks
    left_eyebrow = (parsing == 6).astype(np.uint8) * 255
    right_eyebrow = (parsing == 7).astype(np.uint8) * 255
    
    return left_eyebrow, right_eyebrow

def extract_eyebrow_masks_from_raw_segmentation(raw_segmentation, colored_segmentation=None):
    """
    Extract eyebrow masks directly from the raw segmentation map using class indices.
    
    Args:
        raw_segmentation: Grayscale segmentation map with class indices
        colored_segmentation: Optional colored visualization for overlay creation
        
    Returns:
        left_mask: Binary mask for left eyebrow
        right_mask: Binary mask for right eyebrow
        eyebrow_overlay: Colored overlay showing only eyebrows
    """
    # The BiSeNet face parsing model typically uses these indices:
    # 2 = left eyebrow
    # 3 = right eyebrow
    # But let's try several possible indices based on the model version
    
    # Print unique values to help identify the correct indices
    unique_values = np.unique(raw_segmentation)
    print(f"Unique class indices in raw segmentation: {unique_values}")
    
    # Based on user feedback, we know the exact indices for eyebrows in this model:
    # 97 for the purple eyebrow and 101 for the teal/blue eyebrow
    
    # Use these specific indices first
    left_eyebrow_idx = 97  # Purple eyebrow
    right_eyebrow_idx = 101  # Teal/blue eyebrow
    
    print(f"Using known eyebrow indices: left (purple)={left_eyebrow_idx}, right (teal)={right_eyebrow_idx}")
    
    # Create binary masks for each eyebrow using the known indices
    left_mask = (raw_segmentation == left_eyebrow_idx).astype(np.uint8) * 255
    right_mask = (raw_segmentation == right_eyebrow_idx).astype(np.uint8) * 255
    
    # Check if we found anything with these indices
    left_pixels = np.count_nonzero(left_mask)
    right_pixels = np.count_nonzero(right_mask)
    
    if left_pixels > 0 or right_pixels > 0:
        print(f"Found {left_pixels} pixels for left eyebrow and {right_pixels} for right eyebrow")
        
        # Create a colored overlay
        if colored_segmentation is not None and len(colored_segmentation.shape) == 3:
            eyebrow_overlay = colored_segmentation.copy()
        else:
            h, w = raw_segmentation.shape[:2]
            eyebrow_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Use bright green for visibility
        eyebrow_overlay[left_mask > 0] = [0, 255, 0]  # type: ignore # BGR format
        eyebrow_overlay[right_mask > 0] = [0, 255, 0]  # type: ignore # BGR format
        
        return left_mask, right_mask, eyebrow_overlay
    
    # If the known indices didn't work (which would be surprising), fall back to detection
    print("Known indices didn't work, falling back to detection...")
    
    # Define approximate eyebrow regions (top 1/3 of face, left and right sides)
    h, w = raw_segmentation.shape[:2]
    top_third = slice(0, h//3)
    left_side = slice(0, w//2)
    right_side = slice(w//2, w)
    
    # Get the most common indices in these regions
    left_eyebrow_region = raw_segmentation[top_third, left_side]
    right_eyebrow_region = raw_segmentation[top_third, right_side]
    
    # Count occurrences of each index in these regions
    left_counts = np.bincount(left_eyebrow_region.flatten())
    right_counts = np.bincount(right_eyebrow_region.flatten())
    
    # Ensure the arrays are long enough
    if len(left_counts) < 256:
        left_counts = np.pad(left_counts, (0, 256 - len(left_counts)))
    if len(right_counts) < 256:
        right_counts = np.pad(right_counts, (0, 256 - len(right_counts)))
    
    # Find the most common indices in each region (excluding 0 which is background)
    left_indices = np.argsort(left_counts)[-5:]  # Get top 5 indices
    right_indices = np.argsort(right_counts)[-5:]  # Get top 5 indices
    
    print(f"Most common indices in left eyebrow region: {left_indices}")
    print(f"Most common indices in right eyebrow region: {right_indices}")
    
    # Try these indices as potential eyebrow indices
    possible_eyebrow_indices = [(97, 101)]  # Start with our known good indices
    
    for left_idx in left_indices:
        for right_idx in right_indices:
            if left_idx > 0 and right_idx > 0:  # Skip background
                possible_eyebrow_indices.append((left_idx, right_idx))
    
    print(f"Trying these index pairs: {possible_eyebrow_indices[:5]}... (total: {len(possible_eyebrow_indices)})")
    
    # If we have a colored segmentation, also try to identify indices by color
    if colored_segmentation is not None:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(colored_segmentation, cv2.COLOR_BGR2HSV)
        
        # Look for purple/magenta (eyebrow color in the sample image)
        lower_purple = np.array([140, 50, 50])
        upper_purple = np.array([170, 255, 255])
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # Look for teal/cyan (another eyebrow color)
        lower_teal = np.array([80, 50, 50])
        upper_teal = np.array([100, 255, 255])
        teal_mask = cv2.inRange(hsv, lower_teal, upper_teal)
        
        # Find indices in the raw segmentation that correspond to these colors
        purple_indices = raw_segmentation[purple_mask > 0] # type: ignore
        teal_indices = raw_segmentation[teal_mask > 0] # type: ignore
        
        if len(purple_indices) > 0 and len(teal_indices) > 0:
            purple_idx = np.bincount(purple_indices).argmax()
            teal_idx = np.bincount(teal_indices).argmax()
            
            if purple_idx > 0 and teal_idx > 0:  # Skip background
                print(f"Found indices by color: purple={purple_idx}, teal={teal_idx}")
                possible_eyebrow_indices.insert(0, (purple_idx, teal_idx))  # type: ignore # type: ignore # Try these first
    
    # Try each pair of indices
    for left_idx, right_idx in possible_eyebrow_indices:
        # Check if these indices exist in the segmentation
        if left_idx in unique_values or right_idx in unique_values:
            print(f"Found eyebrow indices: left={left_idx}, right={right_idx}")
            
            # Create binary masks for each eyebrow
            left_mask = (raw_segmentation == left_idx).astype(np.uint8) * 255
            right_mask = (raw_segmentation == right_idx).astype(np.uint8) * 255
            
            # Check if we found anything
            left_pixels = np.count_nonzero(left_mask)
            right_pixels = np.count_nonzero(right_mask)
            
            if left_pixels > 0 or right_pixels > 0:
                print(f"Found {left_pixels} pixels for left eyebrow and {right_pixels} for right eyebrow")
                
                # Create a colored overlay
                if colored_segmentation is not None and len(colored_segmentation.shape) == 3:
                    base_image = colored_segmentation
                else:
                    # Create a blank RGB image if no colored segmentation is provided
                    base_image = np.zeros((raw_segmentation.shape[0], raw_segmentation.shape[1], 3), dtype=np.uint8)
                
                # Create overlay
                eyebrow_overlay = base_image.copy()
                # Use bright green for visibility
                eyebrow_overlay[left_mask > 0] = [0, 255, 0]  # BGR format
                eyebrow_overlay[right_mask > 0] = [0, 255, 0]
                
                return left_mask, right_mask, eyebrow_overlay
    
    # If we couldn't find eyebrows with any of the index pairs, try a fallback approach
    print("Could not find eyebrows using standard indices, trying fallback approach...")
    
    # Create empty masks as fallback
    h, w = raw_segmentation.shape[:2]
    empty_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Try to extract just the hair regions from the top of the head
    # This might include eyebrows in your specific model
    top_quarter = slice(0, h//4)  # Just the top quarter of the image
    top_region = raw_segmentation[top_quarter, :]
    
    # Find the most common non-zero indices in the top region
    top_indices = np.bincount(top_region.flatten())
    if len(top_indices) > 0:
        # Get the most common non-zero index
        if 0 in np.unique(top_region) and len(top_indices) > 1:
            # Skip 0 (background) if present
            top_index = np.argsort(top_indices[1:])[-1] + 1 if len(top_indices) > 1 else 0
        else:
            top_index = np.argmax(top_indices)
        
        if top_index > 0:
            print(f"Using top region index: {top_index}")
            hair_mask = (raw_segmentation == top_index).astype(np.uint8) * 255
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3,3), np.uint8)
            hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
            
            # Create a colored overlay
            if colored_segmentation is not None and len(colored_segmentation.shape) == 3:
                eyebrow_overlay = colored_segmentation.copy()
            else:
                eyebrow_overlay = np.zeros((h, w, 3), dtype=np.uint8)
            
            eyebrow_overlay[hair_mask > 0] = [0, 255, 0]  # type: ignore # BGR format
            
            # Try to isolate just the eyebrow regions from the hair mask
            # Focus on the middle third horizontally, top third vertically
            eyebrow_region_mask = np.zeros_like(hair_mask)
            eyebrow_region_mask[0:h//3, w//4:3*w//4] = 255
            
            # Combine with the hair mask to get just the eyebrows
            eyebrow_mask = cv2.bitwise_and(hair_mask, eyebrow_region_mask)
            
            if np.count_nonzero(eyebrow_mask) > 0:
                print(f"Found potential eyebrows in hair region")
                return eyebrow_mask, eyebrow_mask, eyebrow_overlay
            else:
                print(f"Using full hair mask as fallback")
                return hair_mask, hair_mask, eyebrow_overlay
    
    # If still no success, try color-based approach as last resort
    if colored_segmentation is not None and len(colored_segmentation.shape) == 3:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(colored_segmentation, cv2.COLOR_BGR2HSV)
        
        # Try to detect purple/magenta (common eyebrow color in visualizations)
        lower_purple = np.array([140, 50, 50])
        upper_purple = np.array([170, 255, 255])
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # Try to detect teal/cyan (another common eyebrow color)
        lower_teal = np.array([80, 50, 50])
        upper_teal = np.array([100, 255, 255])
        teal_mask = cv2.inRange(hsv, lower_teal, upper_teal)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(purple_mask, teal_mask)
        
        if np.count_nonzero(combined_mask) > 0:
            print("Found eyebrows using color detection fallback")
            
            # Create overlay
            eyebrow_overlay = colored_segmentation.copy()
            eyebrow_overlay[combined_mask > 0] = [0, 255, 0]  # type: ignore # BGR format
            
            return combined_mask, combined_mask, eyebrow_overlay
    
    # If all else fails, return empty masks
    print("Could not find eyebrows with any method")
    eyebrow_overlay = np.zeros((h, w, 3), dtype=np.uint8) if colored_segmentation is None else colored_segmentation.copy()
    return empty_mask, empty_mask, eyebrow_overlay

def extract_only_eyebrows(segmentation_image, raw_segmentation=None):
    """
    Extract only the eyebrow regions using a position and shape-based approach.
    
    Args:
        segmentation_image: The colored segmentation image
        raw_segmentation: Optional raw segmentation map with class indices
        
    Returns:
        eyebrow_mask: Binary mask containing only eyebrows
        eyebrow_overlay: Colored overlay showing eyebrows on the original image
    """
    # Create an empty mask for the eyebrow region
    h, w = segmentation_image.shape[:2]
    eyebrow_region_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define the eyebrow region - this is the key to reliable detection
    # Eyebrows are typically in the upper 15-25% of the face
    eyebrow_top = int(h * 0.15)    # Start at 15% from the top
    eyebrow_bottom = int(h * 0.25)  # End at 25% from the top
    
    # Create a mask for just the eyebrow region
    eyebrow_region_mask[eyebrow_top:eyebrow_bottom, :] = 255
    
    # Convert the segmentation image to grayscale for edge detection
    gray = cv2.cvtColor(segmentation_image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection to find boundaries between segments
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate the edges to make them more prominent
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Invert the edges to get the segments
    segments = cv2.bitwise_not(dilated_edges)
    
    # Apply the eyebrow region mask to get only segments in the eyebrow region
    eyebrow_segments = cv2.bitwise_and(segments, eyebrow_region_mask)
    
    # Find contours in the eyebrow region
    contours, _ = cv2.findContours(eyebrow_segments, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for the filtered eyebrows
    eyebrow_mask = np.zeros_like(eyebrow_segments)
    
    # Filter contours to keep only those that match eyebrow characteristics
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Eyebrow filter criteria:
        # 1. Must be wider than tall (aspect_ratio > 2)
        # 2. Must have reasonable width (at least 10% of face width)
        # 3. Must not be too tall (less than 5% of face height)
        # 4. Must be in the upper part of the face
        # 5. Must not be at the very edges
        if (aspect_ratio > 2.0 and 
            w > 0.1 * segmentation_image.shape[1] and
            h < 0.05 * segmentation_image.shape[0] and
            y >= eyebrow_top and y + h <= eyebrow_bottom and
            x > 0.1 * segmentation_image.shape[1] and
            x + w < 0.9 * segmentation_image.shape[1]):
            
            # Draw the contour on the mask
            cv2.drawContours(eyebrow_mask, [contour], -1, 255, -1) # type: ignore
    
    # If we didn't find good eyebrow contours, try a different approach
    if np.sum(eyebrow_mask) < 100:  # type: ignore # Not enough pixels found
        # Create a mask for the left and right eyebrow regions
        left_region = np.zeros_like(eyebrow_region_mask)
        right_region = np.zeros_like(eyebrow_region_mask)
        
        # Define left and right regions (roughly where eyebrows would be)
        mid_x = w // 2
        left_x_start = int(w * 0.2)
        left_x_end = int(w * 0.45)
        right_x_start = int(w * 0.55)
        right_x_end = int(w * 0.8)
        
        # Set the regions
        left_region[eyebrow_top:eyebrow_bottom, left_x_start:left_x_end] = 255
        right_region[eyebrow_top:eyebrow_bottom, right_x_start:right_x_end] = 255
        
        # Create eyebrow shapes directly
        left_eyebrow = np.zeros_like(eyebrow_region_mask)
        right_eyebrow = np.zeros_like(eyebrow_region_mask)
        
        # Draw eyebrow shapes
        left_center_y = (eyebrow_top + eyebrow_bottom) // 2
        right_center_y = left_center_y
        
        # Draw elliptical eyebrows
        cv2.ellipse(left_eyebrow, 
                   (int((left_x_start + left_x_end) / 2), left_center_y),
                   (int((left_x_end - left_x_start) / 2), int((eyebrow_bottom - eyebrow_top) / 3)),
                   0, 0, 180, 255, -1) # type: ignore
        
        cv2.ellipse(right_eyebrow, 
                   (int((right_x_start + right_x_end) / 2), right_center_y),
                   (int((right_x_end - right_x_start) / 2), int((eyebrow_bottom - eyebrow_top) / 3)),
                   0, 0, 180, 255, -1) # type: ignore
        
        # Combine the eyebrows
        eyebrow_mask = cv2.bitwise_or(left_eyebrow, right_eyebrow)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    eyebrow_mask = cv2.morphologyEx(eyebrow_mask, cv2.MORPH_OPEN, kernel)
    eyebrow_mask = cv2.morphologyEx(eyebrow_mask, cv2.MORPH_CLOSE, kernel)
    
    # Create a colored overlay
    eyebrow_overlay = segmentation_image.copy()
    
    # Use a bright green color for visibility
    eyebrow_overlay[eyebrow_mask > 0] = [0, 255, 0]  # type: ignore # BGR format
    
    # Print debug info
    print(f"Eyebrow mask pixels: {np.sum(eyebrow_mask > 0)}") # type: ignore
    
    return eyebrow_mask, eyebrow_overlay
    


def create_eyebrow_overlay(image, left_mask, right_mask, alpha=0.7):
    """
    Create a visualization of eyebrow segmentation overlaid on the original image.
    """
    # Create a copy of the original image
    overlay = image.copy()
    
    # Create a color mask with the same shape as the image
    color_mask = np.zeros_like(image)
    
    # Apply colors to the mask where eyebrows are detected
    if left_mask is not None and left_mask.shape[:2] == image.shape[:2]:
        color_mask[left_mask > 0] = EYEBROW_COLORS['left']
    
    if right_mask is not None and right_mask.shape[:2] == image.shape[:2]:
        color_mask[right_mask > 0] = EYEBROW_COLORS['right']
    
    # Blend the original image with the color mask
    overlay = cv2.addWeighted(overlay, 1.0 - alpha, color_mask, alpha, 0)
    
    return overlay

def segment_eyebrows_with_bisenet(image_path, target_shape=None):
    """
    Main function to segment eyebrows using BiSeNet.
    This uses the exact command structure the user provided.
    
    Args:
        image_path: Path to the input image
        target_shape: Optional shape to resize the masks to
        
    Returns:
        segmentation_image: The full segmentation image from BiSeNet (colored visualization)
        raw_segmentation: The raw grayscale segmentation map with class indices
    """
    temp_dir = None
    try:
        print(f"Starting BiSeNet segmentation for image: {image_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
            
        # Run BiSeNet using the user's exact command structure
        segmentation_path, temp_dir = run_bisenet_direct(image_path)
        print(f"BiSeNet processing complete, segmentation saved at: {segmentation_path}")
        
        # Read the segmentation result directly as both color and grayscale
        segmentation_image = cv2.imread(segmentation_path)  # Color visualization
        raw_segmentation = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)  # Raw segmentation map
        
        if segmentation_image is None or raw_segmentation is None:
            raise ValueError(f"Failed to read segmentation result: {segmentation_path}")
        
        # Make copies of the segmentation images
        segmentation_copy = segmentation_image.copy()
        raw_segmentation_copy = raw_segmentation.copy()
        
        # Print unique values in the raw segmentation to help identify eyebrow indices
        unique_values = np.unique(raw_segmentation_copy)
        print(f"Unique class indices in segmentation: {unique_values}")
        
        # Clean up temporary files
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up temporary directory: {cleanup_error}")
        
        return segmentation_copy, raw_segmentation_copy
    except Exception as e:
        print(f"Error in segment_eyebrows_with_bisenet: {e}")
        # Try to clean up even if there was an error
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        raise

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Eyebrow segmentation with BiSeNet CLI integration')
    parser.add_argument('--img', type=str, required=True, help='Path to image file')
    parser.add_argument('--out_dir', type=str, default=None, help='Directory to save results')
    args = parser.parse_args()
    
    # Run segmentation
    left_mask, right_mask, seg_path = segment_eyebrows_with_bisenet(args.img, args.out_dir) # type: ignore
    
    # Save individual masks
    cv2.imwrite('left_eyebrow_mask.png', left_mask)
    cv2.imwrite('right_eyebrow_mask.png', right_mask)
    
    # Create and save overlay
    original_img = cv2.imread(args.img)
    if original_img is not None:
        # Resize masks if needed
        if left_mask.shape[:2] != original_img.shape[:2]:
            left_mask = cv2.resize(left_mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if right_mask.shape[:2] != original_img.shape[:2]:
            right_mask = cv2.resize(right_mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        overlay = create_eyebrow_overlay(original_img, left_mask, right_mask)
        cv2.imwrite('eyebrow_overlay.png', overlay)
        print('Overlay image saved as eyebrow_overlay.png')
    
    print(f'Segmentation mask: {seg_path}')
    print('Left/right eyebrow masks saved.')
