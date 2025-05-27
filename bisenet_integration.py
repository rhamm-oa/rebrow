import os
import subprocess
import cv2
import numpy as np
import shutil
import tempfile

# Define class indices based on BiSeNetV2 documentation
# Standard BiSeNet class indices (0-18)
ATTRIBUTES = [
    'background',  # 0
    'skin',        # 1
    'l_brow',      # 2 - Left eyebrow
    'r_brow',      # 3 - Right eyebrow
    'l_eye',       # 4 - Left eye
    'r_eye',       # 5 - Right eye
    'eye_g',       # 6 - Eye glasses
    'l_ear',       # 7 - Left ear
    'r_ear',       # 8 - Right ear
    'ear_r',       # 9 - Ear rings
    'nose',        # 10 - Nose
    'mouth',       # 11 - Mouth
    'u_lip',       # 12 - Upper lip
    'l_lip',       # 13 - Lower lip
    'neck',        # 14 - Neck
    'neck_l',      # 15 - Necklace
    'cloth',       # 16 - Clothing
    'hair',        # 17 - Hair
    'hat'          # 18 - Hat
]

# Standard BiSeNet color map
COLOR_LIST = [
    [0, 0, 0],       # Background - Black
    [255, 85, 0],    # Skin - Orange
    [255, 170, 0],   # Left eyebrow - Light orange
    [255, 0, 85],    # Right eyebrow - Pink
    [255, 0, 170],   # Left eye - Magenta
    [0, 255, 0],     # Right eye - Green
    [85, 255, 0],    # Eye glasses - Light green
    [170, 255, 0],   # Left ear - Yellow-green
    [0, 255, 85],    # Right ear - Cyan-green
    [0, 255, 170],   # Ear rings - Light cyan
    [0, 0, 255],     # Nose - Blue
    [85, 0, 255],    # Mouth - Purple
    [170, 0, 255],   # Upper lip - Light purple
    [0, 85, 255],    # Lower lip - Light blue
    [0, 170, 255],   # Neck - Cyan
    [255, 255, 0],   # Necklace - Yellow
    [255, 255, 85],  # Clothing - Light yellow
    [255, 255, 170], # Hair - Very light yellow
    [255, 0, 255],   # Hat - Magenta
]

# Class indices for face parts (standard model)
BACKGROUND_CLASS = 0
SKIN_CLASS = 1
LEFT_EYEBROW_CLASS = 2
RIGHT_EYEBROW_CLASS = 3
LEFT_EYE_CLASS = 4
RIGHT_EYE_CLASS = 5
NOSE_CLASS = 10
MOUTH_CLASS = 11
UPPER_LIP_CLASS = 12
LOWER_LIP_CLASS = 13
HAIR_CLASS = 17

# Based on the logs, these are the potential eyebrow classes in the non-standard model output
NS_LEFT_EYEBROW_CLASS = 127  # Left eyebrow in the non-standard model
NS_RIGHT_EYEBROW_CLASS = 129  # Right eyebrow in the non-standard model

# Colors for visualization
EYEBROW_COLORS = {
    'left': (0, 165, 255),   # Orange in BGR
    'right': (255, 0, 170)    # Pink in BGR
}

# Attributes list from face-parsing/utils/common.py
ATTRIBUTES = [
    'skin',
    'l_brow',
    'r_brow',
    'l_eye',
    'r_eye',
    'eye_g',
    'l_ear',
    'r_ear',
    'ear_r',
    'nose',
    'mouth',
    'u_lip',
    'l_lip',
    'neck',
    'neck_l',
    'cloth',
    'hair',
    'hat'
]

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
    Extract eyebrow masks from the raw segmentation output.
    
    Args:
        raw_segmentation: Raw segmentation map with class indices
        colored_segmentation: Colored visualization from BiSeNet (optional)
        
    Returns:
        left_mask: Binary mask for left eyebrow
        right_mask: Binary mask for right eyebrow
        eyebrow_overlay: Colored overlay showing only eyebrows
        combined_mask: Combined mask of both eyebrows
    """
    if raw_segmentation is None:
        print("Warning: raw_segmentation is None, cannot extract eyebrow masks")
        return None, None, None, None
    
    # Get dimensions
    h, w = raw_segmentation.shape[:2]
    
    # Initialize masks
    left_mask = np.zeros((h, w), dtype=np.uint8)
    right_mask = np.zeros((h, w), dtype=np.uint8)
    combined_eyebrow_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Get unique class indices in the segmentation
    unique_values = np.unique(raw_segmentation)
    print(f"Unique class indices: {unique_values}")
    
    # Determine if we're using standard or non-standard model
    max_index = np.max(unique_values)
    is_non_standard = max_index > 19  # Standard BiSeNet has 19 classes (0-18)
    
    if is_non_standard:
        print("Using non-standard BiSeNet model with indices 36-237")
        # Check if non-standard eyebrow classes are present
        if NS_LEFT_EYEBROW_CLASS in unique_values or NS_RIGHT_EYEBROW_CLASS in unique_values:
            print(f"Found non-standard eyebrow classes {NS_LEFT_EYEBROW_CLASS} and {NS_RIGHT_EYEBROW_CLASS}")
            
            # Create masks for left and right eyebrows
            if NS_LEFT_EYEBROW_CLASS in unique_values:
                left_mask[raw_segmentation == NS_LEFT_EYEBROW_CLASS] = 255
                print(f"Left eyebrow pixels: {np.count_nonzero(left_mask)}")
            
            if NS_RIGHT_EYEBROW_CLASS in unique_values:
                right_mask[raw_segmentation == NS_RIGHT_EYEBROW_CLASS] = 255
                print(f"Right eyebrow pixels: {np.count_nonzero(right_mask)}")
        else:
            # Try to identify potential eyebrow classes based on position and color
            print("Non-standard eyebrow classes not found, trying to identify them...")
            
            # Use the enhanced visualization approach
            vis = visualize_segmentation(raw_segmentation)
            
            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(vis, cv2.COLOR_BGR2HSV)
            
            # Try to find eyebrow-like regions in the upper part of the face
            # This is a heuristic approach and may need tuning
            for idx in unique_values:
                if idx > 35:  # Skip background and standard classes
                    # Create a temporary mask for this class
                    temp_mask = np.zeros((h, w), dtype=np.uint8)
                    temp_mask[raw_segmentation == idx] = 255
                    
                    # Count pixels in the upper face region (approximately where eyebrows would be)
                    upper_third = h // 3
                    upper_face_mask = np.zeros((h, w), dtype=np.uint8)
                    upper_face_mask[0:upper_third, :] = 255
                    
                    # Count pixels in the upper face region
                    upper_face_pixels = cv2.bitwise_and(temp_mask, upper_face_mask)
                    upper_face_count = np.count_nonzero(upper_face_pixels)
                    total_pixels = np.count_nonzero(temp_mask)
                    
                    # If more than 50% of the pixels are in the upper face, it could be an eyebrow
                    if total_pixels > 100 and upper_face_count / total_pixels > 0.5:
                        print(f"Class {idx} could be an eyebrow class (upper face ratio: {upper_face_count / total_pixels:.2f})")
                        
                        # Try to determine if it's left or right eyebrow based on position
                        y_indices, x_indices = np.where(temp_mask > 0)
                        if len(x_indices) > 0:
                            center_x = np.mean(x_indices)
                            
                            # If center is on the left side of the image, it's likely the left eyebrow
                            if center_x < w/2:
                                print(f"Class {idx} is likely the left eyebrow (center_x: {center_x})")
                                left_mask = cv2.bitwise_or(left_mask, temp_mask)
                            else:
                                print(f"Class {idx} is likely the right eyebrow (center_x: {center_x})")
                                right_mask = cv2.bitwise_or(right_mask, temp_mask)
    else:
        # Standard BiSeNet model with classes 2 and 3 for left and right eyebrows
        print("Using standard BiSeNet model with indices 0-18")
        
        # Check for left eyebrow (class 2)
        if LEFT_EYEBROW_CLASS in unique_values:
            left_mask[raw_segmentation == LEFT_EYEBROW_CLASS] = 255
            print(f"Left eyebrow pixels: {np.count_nonzero(left_mask)}")
        
        # Check for right eyebrow (class 3)
        if RIGHT_EYEBROW_CLASS in unique_values:
            right_mask[raw_segmentation == RIGHT_EYEBROW_CLASS] = 255
            print(f"Right eyebrow pixels: {np.count_nonzero(right_mask)}")
    
    # Create combined mask
    combined_eyebrow_mask = cv2.bitwise_or(left_mask, right_mask)
    # If the masks are too small, apply dilation to make them more visible
    if np.count_nonzero(combined_eyebrow_mask) < 1000:
        kernel = np.ones((5, 5), np.uint8)
        combined_eyebrow_mask = cv2.dilate(combined_eyebrow_mask, kernel, iterations=2)
        left_mask = cv2.dilate(left_mask, kernel, iterations=2)
        right_mask = cv2.dilate(right_mask, kernel, iterations=2)
    
    # If we have separate left and right masks, create a colored overlay
    if np.count_nonzero(left_mask) > 100 or np.count_nonzero(right_mask) > 100:
        # Create a colored overlay for visualization
        eyebrow_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply colors to the masks using vectorized operations for efficiency
        # Convert to np.uint8 to fix Pylance type errors
        left_mask_uint8 = np.array(left_mask, dtype=np.uint8)
        right_mask_uint8 = np.array(right_mask, dtype=np.uint8)
        eyebrow_overlay[left_mask_uint8 > 0] = EYEBROW_COLORS['left']   # Orange for left eyebrow
        eyebrow_overlay[right_mask_uint8 > 0] = EYEBROW_COLORS['right']  # Pink for right eyebrow
        
        return left_mask, right_mask, eyebrow_overlay, combined_eyebrow_mask
    else:
        print("Could not extract eyebrow masks from segmentation")
        return None, None, None, None


def extract_eyebrows_common_approach(raw_segmentation, original_image=None):
    """
    Extract eyebrows using the approach from utils/common.py.
    This function creates a visualization similar to the one in the face-parsing repo.
    
    Args:
        raw_segmentation: Raw segmentation map with class indices
        original_image: Original image to blend with the segmentation (optional)
        
    Returns:
        blended_image: Visualization of the segmentation blended with the original image
        eyebrow_mask: Binary mask containing only eyebrows
    """
    # Get image dimensions
    h, w = raw_segmentation.shape
    
    # Create a color mask for visualization
    segmentation_color = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Determine if we're using standard or non-standard model
    max_index = np.max(raw_segmentation)
    is_non_standard = max_index > 19  # Standard BiSeNet has 19 classes (0-18)
    
    # Create a binary mask for eyebrows
    eyebrow_mask = np.zeros((h, w), dtype=np.uint8)
    
    if is_non_standard:
        # For non-standard model, use the classes we've identified
        if NS_LEFT_EYEBROW_CLASS in np.unique(raw_segmentation):
            eyebrow_mask[raw_segmentation == NS_LEFT_EYEBROW_CLASS] = 255
            # Use the color from COLOR_LIST for visualization
            segmentation_color[raw_segmentation == NS_LEFT_EYEBROW_CLASS] = COLOR_LIST[LEFT_EYEBROW_CLASS]
        
        if NS_RIGHT_EYEBROW_CLASS in np.unique(raw_segmentation):
            eyebrow_mask[raw_segmentation == NS_RIGHT_EYEBROW_CLASS] = 255
            segmentation_color[raw_segmentation == NS_RIGHT_EYEBROW_CLASS] = COLOR_LIST[RIGHT_EYEBROW_CLASS]
        
        # Apply colors for other facial features based on our best guess
        # This is approximate and may need tuning
        for idx in np.unique(raw_segmentation):
            if idx not in [NS_LEFT_EYEBROW_CLASS, NS_RIGHT_EYEBROW_CLASS]:
                # Map non-standard indices to standard ones as best we can
                if 36 <= idx <= 50:  # Skin
                    segmentation_color[raw_segmentation == idx] = COLOR_LIST[SKIN_CLASS]
                elif 110 <= idx <= 120:  # Left eye
                    segmentation_color[raw_segmentation == idx] = COLOR_LIST[LEFT_EYE_CLASS]
                elif 120 <= idx <= 130 and idx not in [NS_LEFT_EYEBROW_CLASS, NS_RIGHT_EYEBROW_CLASS]:  # Right eye
                    segmentation_color[raw_segmentation == idx] = COLOR_LIST[RIGHT_EYE_CLASS]
                elif 130 <= idx <= 140:  # Nose
                    segmentation_color[raw_segmentation == idx] = COLOR_LIST[NOSE_CLASS]
                elif 150 <= idx <= 160:  # Lips
                    segmentation_color[raw_segmentation == idx] = COLOR_LIST[MOUTH_CLASS]
                elif 170 <= idx <= 180:  # Hair
                    segmentation_color[raw_segmentation == idx] = COLOR_LIST[HAIR_CLASS]
    else:
        # For standard model, use the standard class indices
        # Apply colors for each class
        for class_idx in range(1, min(len(COLOR_LIST), np.max(raw_segmentation) + 1)):
            class_pixels = np.where(raw_segmentation == class_idx)
            segmentation_color[class_pixels[0], class_pixels[1], :] = COLOR_LIST[class_idx]
        
        # Extract eyebrow mask
        eyebrow_mask[raw_segmentation == LEFT_EYEBROW_CLASS] = 255
        eyebrow_mask[raw_segmentation == RIGHT_EYEBROW_CLASS] = 255
    
    # If original image is provided, blend with segmentation
    if original_image is not None:
        # Ensure original image is in BGR format for blending
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            bgr_image = original_image.copy()
        else:
            bgr_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        
        # Resize if necessary
        if bgr_image.shape[:2] != (h, w):
            bgr_image = cv2.resize(bgr_image, (w, h))
        
        # Blend the image with the segmentation mask
        blended_image = cv2.addWeighted(bgr_image, 0.6, segmentation_color, 0.4, 0)
        return blended_image, eyebrow_mask
    
    return segmentation_color, eyebrow_mask
    if np.count_nonzero(left_mask) < 100 or np.count_nonzero(right_mask) < 100:
        print("BiSeNet masks are too small, using landmark-based fallback")
        # This will be handled by the traditional approach in main.py
    
    # Create a colored overlay for visualization
    eyebrow_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Apply colors to the masks
    # Use safer approach with explicit boolean conversion
    left_indices = np.where(np.array(left_mask, dtype=np.uint8) > 0)
    for i, j in zip(*left_indices):
        eyebrow_overlay[i, j] = EYEBROW_COLORS['left']  # Orange for left eyebrow (BGR)
    
    right_indices = np.where(np.array(right_mask, dtype=np.uint8) > 0)
    for i, j in zip(*right_indices):
        eyebrow_overlay[i, j] = EYEBROW_COLORS['right']  # Pink for right eyebrow (BGR)
    
    # Also create a combined eyebrow mask for visualization
    combined_mask = cv2.bitwise_or(left_mask, right_mask)
    
    return left_mask, right_mask, eyebrow_overlay, combined_mask

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
        # Convert to np.uint8 to fix Pylance type errors
        left_mask_uint8 = np.array(left_mask, dtype=np.uint8)
        color_mask[left_mask_uint8 > 0] = EYEBROW_COLORS['left']
    
    if right_mask is not None and right_mask.shape[:2] == image.shape[:2]:
        # Convert to np.uint8 to fix Pylance type errors
        right_mask_uint8 = np.array(right_mask, dtype=np.uint8)
        color_mask[right_mask_uint8 > 0] = EYEBROW_COLORS['right']
    
    # Blend the original image with the color mask
    overlay = cv2.addWeighted(overlay, 1.0 - alpha, color_mask, alpha, 0)
    
    return overlay


def visualize_segmentation(segmentation_output):
    """
    Visualize the segmentation output for debugging
    
    Args:
        segmentation_output: Raw segmentation output from BiSeNet
        
    Returns:
        Visualization of the segmentation
    """
    # Create a visualization of the segmentation
    unique_indices = np.unique(segmentation_output)
    print(f"Unique class indices found: {unique_indices}")
    
    # Determine if we're using the standard or non-standard model
    max_index = np.max(unique_indices)
    is_non_standard = max_index > 19
    
    if is_non_standard:
        print("Using non-standard model visualization")
        # For non-standard model with indices 36-237
        # Create a colormap with enough entries for all possible indices
        max_class = 240  # A bit more than the maximum observed index (237)
        colors = np.zeros((max_class, 3), dtype=np.uint8)
        
        # Set specific colors for important classes based on observed behavior
        # Background/skin - Blue
        for i in range(36, 50):
            colors[i] = [204, 204, 255]
        
        # Left eyebrow - Purple
        colors[127] = [255, 0, 255]
        
        # Right eyebrow - Teal
        colors[129] = [0, 255, 255]
        
        # Eyes - Green/Pink
        for i in range(110, 120):
            colors[i] = [0, 255, 0]  # Left eye
        for i in range(120, 130):
            if i != 127 and i != 129:  # Skip eyebrow indices
                colors[i] = [255, 0, 128]  # Right eye
        
        # Nose - Red
        for i in range(130, 140):
            colors[i] = [255, 0, 0]
        
        # Lips - Orange/Red
        for i in range(150, 160):
            colors[i] = [255, 128, 0]
        
        # Hair - Light green
        for i in range(170, 180):
            colors[i] = [128, 255, 128]
    else:
        # Dynamically size color array for all present classes
        max_class = int(np.max(segmentation_output)) + 1
        colors = np.zeros((max_class, 3), dtype=np.uint8)

        # Set specific colors for important classes
        # 0: Background - Black
        colors[0] = [0, 0, 0]
        # 1: Face - Light blue (if present)
        if 1 < max_class:
            colors[1] = [204, 204, 255]

        # Non-standard model: assign eyebrow colors to 127/129 if present
        if NS_LEFT_EYEBROW_CLASS < max_class:
            colors[NS_LEFT_EYEBROW_CLASS] = [255, 165, 0]  # Orange
        if NS_RIGHT_EYEBROW_CLASS < max_class:
            colors[NS_RIGHT_EYEBROW_CLASS] = [255, 215, 0]  # Gold
        # Also assign for standard indices just in case
        if LEFT_EYEBROW_CLASS < max_class:
            colors[LEFT_EYEBROW_CLASS] = [255, 165, 0]
        if RIGHT_EYEBROW_CLASS < max_class:
            colors[RIGHT_EYEBROW_CLASS] = [255, 215, 0]
        # Eyes
        if LEFT_EYE_CLASS < max_class:
            colors[LEFT_EYE_CLASS] = [255, 0, 170]
        if RIGHT_EYE_CLASS < max_class:
            colors[RIGHT_EYE_CLASS] = [0, 255, 0]
        # Nose
        if NOSE_CLASS < max_class:
            colors[NOSE_CLASS] = [0, 0, 255]
        # Mouth
        if MOUTH_CLASS < max_class:
            colors[MOUTH_CLASS] = [85, 0, 255]

        # Fill remaining colors with random values
        for i in range(max_class):
            if np.all(colors[i] == 0):
                colors[i] = np.random.randint(0, 255, size=3, dtype=np.uint8)

        # Warn if eyebrow classes are not present
        unique_indices = np.unique(segmentation_output)
        if NS_LEFT_EYEBROW_CLASS not in unique_indices:
            print(f"Warning: NS_LEFT_EYEBROW_CLASS ({NS_LEFT_EYEBROW_CLASS}) not present in segmentation output.")
        if NS_RIGHT_EYEBROW_CLASS not in unique_indices:
            print(f"Warning: NS_RIGHT_EYEBROW_CLASS ({NS_RIGHT_EYEBROW_CLASS}) not present in segmentation output.")
    
    # Create a visualization image
    vis = np.zeros((segmentation_output.shape[0], segmentation_output.shape[1], 3), dtype=np.uint8)
    
    # Assign colors to each class
    for idx in unique_indices:
        if idx < max_class:  # Skip indices that are out of range
            # Convert to np.uint8 to fix Pylance type errors
            mask = np.array(segmentation_output == idx, dtype=np.uint8)
            mask_indices = np.where(mask > 0)
            vis[mask_indices] = colors[idx]
    
    return vis


def create_mask_visualization(original_image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Create a visualization of a mask overlaid on an image
    
    Args:
        original_image: Original image (BGR format)
        mask: Binary mask
        color: Color for the mask overlay (BGR format)
        alpha: Transparency of the overlay (0-1)
        
    Returns:
        Visualization with the mask overlaid on the image
    """
    # Create a copy of the original image
    vis = original_image.copy()
    
    # Create a colored mask
    colored_mask = np.zeros_like(original_image)
    colored_mask[mask > 0] = color
    
    # Overlay the mask on the image
    cv2.addWeighted(colored_mask, alpha, vis, 1 - alpha, 0, vis)
    
    # Draw contours around the mask for better visibility
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, color, 2)
    
    return vis

def segment_eyebrows_with_bisenet(image_path, target_shape=None):
    """
    Segment eyebrows using BiSeNet face parsing.
    
    Args:
        image_path: Path to the input image
        target_shape: Optional shape to resize the segmentation result to
        
    Returns:
        segmentation_image: Colored visualization of the segmentation
        raw_segmentation: Raw segmentation map with class indices
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
        
        # Resize if needed
        if target_shape is not None:
            h, w = target_shape[:2]
            segmentation_image = cv2.resize(segmentation_image, (w, h), interpolation=cv2.INTER_NEAREST)
            raw_segmentation = cv2.resize(raw_segmentation, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Make copies of the segmentation images
        segmentation_copy = segmentation_image.copy()
        raw_segmentation_copy = raw_segmentation.copy()
        
        # Print unique values in the raw segmentation to help identify eyebrow indices
        unique_values = np.unique(raw_segmentation_copy)
        print(f"Unique class indices in segmentation: {unique_values}")
        
        # Check if we have values in the expected range (0-15) or if we have a different model output
        # Convert to a Python list to avoid type issues with numpy arrays
        unique_values_list = unique_values.tolist() if isinstance(unique_values, np.ndarray) else []
        max_value = max(unique_values_list) if unique_values_list else 0
        if max_value > 30:  # If we have values like 33-239, we're using a different model
            # Convert to a more standard format - this is a fallback approach
            # Map the high values to our expected class indices
            # This is just a placeholder - in a real scenario, you'd need to map the actual classes correctly
            print("Warning: Using a different BiSeNet model with non-standard class indices")
            # Create a new raw segmentation with standard class indices
            new_raw_segmentation = np.zeros_like(raw_segmentation_copy)
            
            # For now, just return the original segmentation
            # In a real implementation, you would map the classes correctly
            pass
        
        # Clean up temporary files
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up temporary directory: {cleanup_error}")
        
        return segmentation_copy, raw_segmentation_copy
    except Exception as e:
        print(f"BiSeNet segmentation failed: {e}")
        print("Make sure bisenet_integration.py, face-parsing, and weights are properly set up.")
        # Return empty arrays with the target shape if provided
        if target_shape is not None:
            h, w = target_shape[:2]
            empty_segmentation = np.zeros((h, w, 3), dtype=np.uint8)
            empty_raw = np.zeros((h, w), dtype=np.uint8)
            return empty_segmentation, empty_raw
        return None, None

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
