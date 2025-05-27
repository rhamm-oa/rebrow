import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import traceback
from typing import Dict, Any, Tuple, Optional, List, Union, cast
import matplotlib.gridspec as gridspec
import gc
from numpy.typing import NDArray

# Set environment variables for PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

class FacerSegmentation:
    """
    Face segmentation implementation using the facer library.
    Provides face parsing with CUDA acceleration and eyebrow extraction capabilities.
    """
    
    def __init__(self, use_gpu=True, model_name='farl/lapa/448', detector_model='retinaface/resnet50'):
        """
        Initialize the face segmentation with FaRL model
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            model_name: Model name to use for face parsing. Options:
                        - 'farl/celebm/448': Trained on CelebAMask-HQ
                        - 'farl/lapa/448' (default): Trained on LaPa dataset
            detector_model: Face detector model to use. Options:
                        - 'retinaface/mobilenet': Faster but less accurate
                        - 'retinaface/resnet50': More accurate but slower
                        - 'scrfd/10g': Alternative detector that may work better for some faces
        """
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"Facer Segmentation: Using device: {self.device}")
        
        # Initialize variables
        self.model_name = model_name
        self.detector_model = detector_model
        self.facer = None
        self.face_detector = None
        self.face_parser = None
        self.debug_mode = True  # Set to True to save debug images
        
        # Try to import facer and initialize models
        try:
            import facer
            self.facer = facer
            print("Successfully imported facer library")
            
            # Initialize primary face detector
            self.face_detector = facer.face_detector(detector_model, device=self.device)
            print(f"Primary face detector initialized: {detector_model}")
            
            # Initialize backup detector with a different model if primary is not mobilenet
            if detector_model != 'retinaface/mobilenet':
                self.backup_detector = facer.face_detector('retinaface/mobilenet', device=self.device)
                print("Backup face detector initialized: retinaface/mobilenet")
            else:
                self.backup_detector = facer.face_detector('scrfd/10g', device=self.device)
                print("Backup face detector initialized: scrfd/10g")
            
            # Initialize face parser
            self.face_parser = facer.face_parser(model_name, device=self.device)
            print(f"Face parser initialized with model: {model_name}")
        except ImportError as e:
            print(f"Error importing facer: {e}")
            print("Please install facer with: pip install git+https://github.com/FacePerceiver/facer.git@main")
        except Exception as e:
            print(f"Error initializing face parser: {e}")
            if self.device == 'cuda':
                print("CUDA error. Try using CPU instead or check CUDA installation.")
            traceback.print_exc()
    
    def process_face_image(self, image: np.ndarray, skip_detection: bool = True) -> Dict[str, Any]:
        """
        Process a cropped face image and return parsed facial features using the Facer library.

        Args:
            image (np.ndarray): The cropped face image.
            skip_detection (bool): If True, skip face detection and use the entire image as the face.

        Returns:
            Dict[str, Any]: Parsing results including segmentation masks and visualizations.
        """
        # Save debug info
        debug_dir = os.path.join(os.getcwd(), 'debug_images')
        os.makedirs(debug_dir, exist_ok=True)
        try:
            if self.facer is None or self.face_parser is None:
                return {'success': False, 'error': 'Face parser not initialized'}

            # Save original image for debugging
            debug_dir = os.path.join(os.getcwd(), 'debug_images')
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, 'original_input.png'), 
                       cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 and image.shape[2] == 3 else image)

            # Convert to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3 and image.dtype == np.uint8:  # BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert numpy array to torch tensor
            if isinstance(image, np.ndarray):
                image_float = image.astype(np.float32) / 255.0
                tensor_img = torch.from_numpy(image_float).permute(2, 0, 1).unsqueeze(0)
                tensor_img = tensor_img.to(device=self.device)
            else:
                return {'success': False, 'error': 'Input image must be a numpy array'}

            # Create a fake detection result with the entire image as the face
            h, w = image.shape[:2]
            
            # Create fake landmarks - 5 points for eyes, nose, and mouth corners
            # These are standard facial landmark positions scaled to the image size
            landmarks = torch.tensor([
                [[0.3*w, 0.3*h],   # Left eye
                 [0.7*w, 0.3*h],   # Right eye
                 [0.5*w, 0.5*h],   # Nose
                 [0.3*w, 0.7*h],   # Left mouth corner
                 [0.7*w, 0.7*h]]   # Right mouth corner
            ]).to(device=self.device)
            
            # Create a complete fake detection with all required keys
            fake_detection = {
                'bboxes': torch.tensor([[0, 0, w, h]]).to(device=self.device),
                'landmarks': landmarks,
                'scores': torch.tensor([1.0]).to(device=self.device),
                'image_ids': torch.tensor([0]).to(device=self.device),
                'points': landmarks  # This is the key the parser is looking for
            }
            
            print(f"Created fake detection covering entire image: {h}x{w}")
            
            # Run the face parser with our fake detection
            with torch.inference_mode():
                try:
                    faces = self.face_parser(tensor_img, fake_detection)
                    print("Face parsing successful!")
                except Exception as e:
                    print(f"Face parsing error: {e}")
                    traceback.print_exc()
                    return {'success': False, 'error': f'Face parsing failed: {str(e)}'}

            # Get segmentation logits and probabilities
            seg_logits = faces['seg']['logits']
            seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w

            # Get segmentation map
            seg_map = seg_probs.argmax(dim=1).float().cpu().numpy()[0]

            # Get the number of classes
            n_classes = seg_logits.size(1)

            # Move tensors to CPU for processing
            seg_probs_cpu = seg_probs.cpu()
            tensor = seg_probs_cpu.permute(0, 2, 3, 1)
            tensor = tensor.squeeze().numpy()

            # Extract masks for each facial part
            masks = {}

            # Define the class indices for different facial parts based on the model
            if self.model_name == 'farl/lapa/448':
                class_mapping = {
                    'background': 0,
                    'face': 1,
                    'eyebrow1': 2,  # Left eyebrow
                    'eyebrow2': 3,  # Right eyebrow
                    'eye1': 4,      # Left eye
                    'eye2': 5,      # Right eye
                    'nose': 6,
                    'mouth': 7,
                    'lip': 8,
                    'ear1': 9,      # Left ear
                    'ear2': 10,     # Right ear
                    'hair': 11
                }
            else:
                class_mapping = {
                    'background': 0,
                    'face': 1,
                    'eyebrow1': 2,  # Left eyebrow
                    'eyebrow2': 3,  # Right eyebrow
                    'eye1': 4,      # Left eye
                    'eye2': 5,      # Right eye
                    'nose': 6,
                    'mouth': 7,
                    'lip': 8,
                    'ear1': 9,      # Left ear
                    'ear2': 10,     # Right ear
                    'hair': 11,
                    'hat': 12,
                    'eyeglass': 13,
                    'earring': 14,
                    'necklace': 15,
                    'neck': 16,
                    'cloth': 17
                }

            # Create binary masks for each class
            for part_name, class_idx in class_mapping.items():
                if class_idx < tensor.shape[2]:
                    prob_map = tensor[:, :, class_idx]
                    binary_mask = (prob_map >= 0.5).astype(np.uint8) * 255
                    masks[part_name] = binary_mask
                    
                    # Save mask for debugging
                    cv2.imwrite(os.path.join(debug_dir, f'mask_{part_name}.png'), binary_mask)

            # Create a colored visualization
            colored_vis = self.create_colored_visualization(seg_map, n_classes)
            cv2.imwrite(os.path.join(debug_dir, 'colored_vis.png'), cv2.cvtColor(colored_vis, cv2.COLOR_RGB2BGR))

            # Extract eyebrow masks specifically
            left_eyebrow_mask = masks.get('eyebrow1')
            right_eyebrow_mask = masks.get('eyebrow2')

            # Post-process the eyebrow masks
            if left_eyebrow_mask is not None:
                kernel = np.ones((3, 3), np.uint8)
                left_eyebrow_mask = cv2.morphologyEx(left_eyebrow_mask, cv2.MORPH_CLOSE, kernel)
                left_eyebrow_mask = cv2.morphologyEx(left_eyebrow_mask, cv2.MORPH_OPEN, kernel)

            if right_eyebrow_mask is not None:
                kernel = np.ones((3, 3), np.uint8)
                right_eyebrow_mask = cv2.morphologyEx(right_eyebrow_mask, cv2.MORPH_CLOSE, kernel)
                right_eyebrow_mask = cv2.morphologyEx(right_eyebrow_mask, cv2.MORPH_OPEN, kernel)

            # Combine left and right eyebrow masks
            combined_eyebrow_mask = np.zeros_like(left_eyebrow_mask) if left_eyebrow_mask is not None else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            if left_eyebrow_mask is not None:
                combined_eyebrow_mask = cv2.bitwise_or(combined_eyebrow_mask, left_eyebrow_mask)

            if right_eyebrow_mask is not None:
                combined_eyebrow_mask = cv2.bitwise_or(combined_eyebrow_mask, right_eyebrow_mask)
                
            # Save combined mask for debugging
            cv2.imwrite(os.path.join(debug_dir, 'combined_eyebrow_mask.png'), combined_eyebrow_mask)
            
            # Create image with bounding box for visualization
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Return the segmentation results
            return {
                'success': True,
                'left_eyebrow_mask': left_eyebrow_mask,
                'right_eyebrow_mask': right_eyebrow_mask,
                'combined_eyebrow_mask': combined_eyebrow_mask,
                'all_masks': masks,
                'class_mapping': class_mapping,
                'visualization': colored_vis,
                'segmentation_map': seg_map,
                'original_image': image,
                'detected_face_image': image_bgr
                # No visualization_image as vis_img doesn't exist in this code path
            }

        except Exception as e:
            print(f"Error in face parsing: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def create_colored_visualization(self, seg_map: np.ndarray, n_classes: int) -> np.ndarray:
        """Create a colored visualization of the segmentation map
        
        Args:
            seg_map: Segmentation map
            n_classes: Number of classes
            
        Returns:
            Colored visualization
        """
        # Create a colormap
        from matplotlib import cm
        colormap = cm.get_cmap('tab20', n_classes)
        
        # Create colored segmentation map
        colored_seg = np.zeros((int(seg_map.shape[0]), int(seg_map.shape[1]), 3), dtype=np.uint8)
        for i in range(n_classes):
            mask = (seg_map == i)
            if np.any(mask):
                color = np.array(colormap(i)[:3]) * 255
                colored_seg[mask] = color.astype(np.uint8)
        
        return colored_seg
    
    def create_visualization_overlay(self, image: Optional[NDArray], seg_map: Optional[NDArray], n_classes: int, alpha: float = 0.5) -> NDArray:
        """Create a visualization of the face parsing results overlaid on the original image
        
        Args:
            image: Original image
            seg_map: Segmentation map
            n_classes: Number of classes
            alpha: Transparency level for the overlay (0.0 to 1.0)
            
        Returns:
            Blended visualization
        """
        if image is None or seg_map is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)  # Return empty image on error
            
        # Ensure we have numpy arrays
        image_array = np.asarray(image)
        seg_map_array = np.asarray(seg_map)
            
        # Create a colormap
        from matplotlib import cm
        colormap = cm.get_cmap('tab20', n_classes)
        
        # Make sure image is RGB
        image_rgb = None
        if len(image_array.shape) == 2:  # Grayscale
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:  # RGBA
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB or BGR
            # Check if we need to convert from BGR to RGB
            if image_array.dtype == np.uint8:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_array
        else:
            # Fallback for unexpected formats
            image_rgb = np.asarray(image_array, dtype=np.uint8)
        
        # Create colored segmentation map
        colored_seg = np.zeros((int(seg_map_array.shape[0]), int(seg_map_array.shape[1]), 3), dtype=np.uint8)
        for i in range(n_classes):
            mask = (seg_map_array == i)
            if np.any(mask):
                color = np.array(colormap(i)[:3]) * 255
                colored_seg[mask] = color.astype(np.uint8)
        
        # Blend with original image
        blended = cv2.addWeighted(image_rgb, 1-alpha, colored_seg, alpha, 0)
        
        return blended
    
    def extract_eyebrow_area(self, image: Optional[NDArray], eyebrow_mask: Optional[NDArray]) -> NDArray:
        """Extract the eyebrow area from the original image using the mask
        
        Args:
            image: Original image
            eyebrow_mask: Binary mask for the eyebrow
            
        Returns:
            Masked image showing only the eyebrow area
        """
        if eyebrow_mask is None or image is None:
            # Return an empty image if either input is None
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Make sure we're working with numpy arrays
        image_array = np.asarray(image)
        mask_array = np.asarray(eyebrow_mask)
        
        # Make sure the mask is binary (0 or 255)
        binary_mask = mask_array.copy()
        if binary_mask.max() > 1:
            binary_mask = (binary_mask > 0).astype(np.uint8)
        
        # Create a 3-channel mask if the image is colored
        if len(image_array.shape) == 3 and len(binary_mask.shape) == 2:
            binary_mask = np.stack([binary_mask, binary_mask, binary_mask], axis=2)
        
        # Extract the eyebrow area
        masked_image = np.zeros_like(image_array)
        masked_image[binary_mask > 0] = image_array[binary_mask > 0]
        
        return masked_image
    
    def visualize_segmentation_results(self, result: Dict[str, Any], save_path: Optional[str] = None) -> NDArray:
        """Visualize the segmentation results with original image, segmentation, and eyebrow extraction
        
        Args:
            result: Result dictionary from process_face_image
            save_path: Optional path to save the visualization
            
        Returns:
            Visualization image
        """
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown error')
            print(f"Segmentation failed: {error_msg}")
            # Return an empty image on error
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Check if we have a pre-generated visualization image
        if 'visualization_image' in result:
            return result['visualization_image']
        
        # Get the original image
        original_image = result.get('original_image')
        
        # Check if we're using the new color channel approach
        if 'r_channel' in result and 'g_channel' in result and 'b_channel' in result:
            # Create a figure with subplots for color channels
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))
            
            # Original image
            if original_image is not None:
                if len(np.asarray(original_image).shape) == 3:
                    axs[0, 0].imshow(cv2.cvtColor(np.asarray(original_image), cv2.COLOR_BGR2RGB))
                else:
                    axs[0, 0].imshow(original_image, cmap='gray')
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis('off')
            
            # R channel
            axs[0, 1].imshow(result['r_channel'], cmap='Reds')
            axs[0, 1].set_title('Red Channel')
            axs[0, 1].axis('off')
            
            # G channel
            axs[1, 0].imshow(result['g_channel'], cmap='Greens')
            axs[1, 0].set_title('Green Channel')
            axs[1, 0].axis('off')
            
            # B channel
            axs[1, 1].imshow(result['b_channel'], cmap='Blues')
            axs[1, 1].set_title('Blue Channel')
            axs[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Convert the figure to a numpy array
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8) # type: ignore
            w, h = canvas.get_width_height()
            vis_img = buf.reshape(h, w, 3)
            
            plt.close(fig)
            
            # Save the visualization if a path is provided
            if save_path:
                cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                print(f"Visualization saved to {save_path}")
                
            return vis_img
        
        # If we have the traditional segmentation results
        seg_map = result.get('segmentation_map')
        n_classes = len(result.get('class_mapping', {}))
        
        # Get the colored visualization
        colored_vis = result.get('visualization')
        
        # Get the eyebrow masks
        left_eyebrow_mask = result.get('left_eyebrow_mask')
        right_eyebrow_mask = result.get('right_eyebrow_mask')
        combined_eyebrow_mask = result.get('combined_eyebrow_mask')
        
        # Extract eyebrow areas
        eyebrow_area = self.extract_eyebrow_area(original_image, combined_eyebrow_mask)
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
        
        # Original image
        ax1 = plt.subplot(gs[0, 0])
        if original_image is not None and len(np.asarray(original_image).shape) == 3:
            ax1.imshow(cv2.cvtColor(np.asarray(original_image), cv2.COLOR_BGR2RGB))
        elif original_image is not None:
            ax1.imshow(original_image, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Segmentation visualization
        ax2 = plt.subplot(gs[0, 1])
        if colored_vis is not None:
            ax2.imshow(colored_vis)
        ax2.set_title('Face Segmentation')
        ax2.axis('off')
        
        # Overlay visualization
        ax3 = plt.subplot(gs[0, 2])
        overlay = self.create_visualization_overlay(original_image, seg_map, n_classes)
        ax3.imshow(overlay)
        ax3.set_title('Segmentation Overlay')
        ax3.axis('off')
        
        # Left eyebrow mask
        ax4 = plt.subplot(gs[1, 0])
        if left_eyebrow_mask is not None:
            ax4.imshow(left_eyebrow_mask, cmap='gray')
        ax4.set_title('Left Eyebrow Mask')
        ax4.axis('off')
        
        # Right eyebrow mask
        ax5 = plt.subplot(gs[1, 1])
        if right_eyebrow_mask is not None:
            ax5.imshow(right_eyebrow_mask, cmap='gray')
        ax5.set_title('Right Eyebrow Mask')
        ax5.axis('off')
        
        # Combined eyebrow area
        ax6 = plt.subplot(gs[1, 2])
        if eyebrow_area is not None and len(eyebrow_area.shape) == 3:
            ax6.imshow(cv2.cvtColor(eyebrow_area, cv2.COLOR_BGR2RGB))
        elif eyebrow_area is not None:
            ax6.imshow(eyebrow_area)
        ax6.set_title('Extracted Eyebrows')
        ax6.axis('off')
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        # Convert the figure to a numpy array using a robust backend approach
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8) # type: ignore
        w, h = canvas.get_width_height()
        vis_img = buf.reshape(h, w, 3)
        
        plt.close(fig)
        
        return vis_img
    
    def segment_eyebrows(self, image: np.ndarray, visualize: bool = True, skip_detection: bool = True) -> Dict[str, Any]:
        """
        Segment eyebrows from a cropped face image using Facer.

        Args:
            image (np.ndarray): The cropped face image.
            visualize (bool): Whether to generate a visualization image.
            skip_detection (bool): If True, skip face detection and use the entire image as the face.

        Returns:
            Dict[str, Any]: Results including masks and visualization.
        """
        result = self.process_face_image(image, skip_detection=skip_detection)
        if visualize and result.get('success', False):
            vis_img = self.visualize_segmentation_results(result)
            result['visualization_image'] = vis_img
        return result
    
    def cleanup(self) -> None:
        """Clean up resources used by the face parser"""
        try:
            # Clear model from GPU memory if it exists
            if hasattr(self, 'face_parser') and self.face_parser is not None:
                del self.face_parser
                self.face_parser = None
            
            if hasattr(self, 'face_detector') and self.face_detector is not None:
                del self.face_detector
                self.face_detector = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            print("Facer segmentation resources cleaned up")
        except Exception as e:
            print(f"Error during cleanup: {e}")


# Example usage function
def segment_eyebrows_example(image_path: str):
    """Example function to demonstrate eyebrow segmentation"""
    import cv2
    image = cv2.imread(image_path)
    segmenter = FacerSegmentation(use_gpu=True)
    result = segmenter.segment_eyebrows(image, visualize=True)
    if result.get('success', False):
        vis_img = result.get('visualization_image')
        if vis_img is not None:
            cv2.imwrite('segmentation_visualization.png', vis_img[..., ::-1])  # Convert RGB to BGR
            print('Segmentation successful. Visualization saved as segmentation_visualization.png')
        else:
            print('Segmentation successful, but no visualization image produced.')
    else:
        print('Segmentation failed:', result.get('error'))
        return
    
    # Display the visualization
    vis_img = result.get('visualization_image')
    if vis_img is not None:
        plt.figure(figsize=(15, 10))
        plt.imshow(vis_img)
        plt.axis('off')
        plt.show()
    
    # Clean up resources
    segmenter.cleanup()
    
    print("Eyebrow segmentation completed successfully")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Segment eyebrows from a face image")
    parser.add_argument("--image", type=str, required=True, help="Path to the input face image")
    args = parser.parse_args()
    segment_eyebrows_example(args.image)
