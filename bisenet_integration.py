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
    
    # Define paths exactly as in the user's command
    face_parsing_dir = os.path.join(current_dir, 'face-parsing')
    inference_script = os.path.join(face_parsing_dir, 'inference.py')
    weight_path = os.path.join(face_parsing_dir, 'weights', 'resnet18.pt')
    
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
        '--model', 'resnet18',
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
        segmentation_image: The full segmentation image from BiSeNet
        segmentation_path: Path to the segmentation result file (temporary)
    """
    temp_dir = None
    try:
        print(f"Starting BiSeNet segmentation for image: {image_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
            
        # Run BiSeNet using the user's exact command structure
        segmentation_path, temp_dir = run_bisenet_direct(image_path)
        print(f"BiSeNet processing complete, segmentation saved at: {segmentation_path}")
        
        # Read the segmentation result directly
        segmentation_image = cv2.imread(segmentation_path)
        if segmentation_image is None:
            raise ValueError(f"Failed to read segmentation result: {segmentation_path}")
            
        # Resize if needed
        if target_shape is not None and segmentation_image.shape != target_shape:
            segmentation_image = cv2.resize(segmentation_image, (target_shape[1], target_shape[0]))
        
        # Make a copy of the segmentation image
        segmentation_copy = segmentation_image.copy()
        
        # Clean up temporary files
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up temporary directory: {cleanup_error}")
        
        return segmentation_copy
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
    left_mask, right_mask, seg_path = segment_eyebrows_with_bisenet(args.img, args.out_dir)
    
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
