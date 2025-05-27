import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import io

# Import custom modules
from face_detection import FaceDetector
from eyebrow_segmentation import EyebrowSegmentation
from color_analysis import ColorAnalysis
from shape_analysis import ShapeAnalysis

# Set page config
st.set_page_config(
    page_title="Eyebrow Analysis App",
    page_icon="👁️",
    layout="wide"
)

# App title and description
st.title("Eyebrow Analysis App")
st.markdown("""
This app analyzes eyebrows in facial images to extract insights about:
- Eyebrow color (dominant colors)
- Eyebrow shape and characteristics
- Detailed visualization of eyebrow features
""")

# Initialize modules
face_detector = FaceDetector()
eyebrow_segmentation = EyebrowSegmentation()
color_analyzer = ColorAnalysis()
shape_analyzer = ShapeAnalysis()

# Function to convert OpenCV image to PIL Image
def cv2_to_pil(cv2_img):
    if cv2_img is None:
        return None
    if len(cv2_img.shape) == 2:  # grayscale
        return Image.fromarray(cv2_img)
    elif len(cv2_img.shape) == 3 and cv2_img.shape[2] == 3:  # BGR
        return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    elif len(cv2_img.shape) == 3 and cv2_img.shape[2] == 4:  # BGRA
        return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGRA2RGBA))
    return None

# Function to convert PIL Image to OpenCV image
def pil_to_cv2(pil_img):
    if pil_img is None:
        return None
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Function to process the uploaded image
def process_image(image):
    # Detect face landmarks
    face_detector = FaceDetector(use_gpu=True)  # Enable GPU support
    results = face_detector.detect_face(image)
    
    if not results.multi_face_landmarks:
        st.error("No face detected in the image. Please upload another image.")
        return None
    
    # Crop face
    face_crop, crop_coords = face_detector.crop_face(image, results)
    
    if face_crop is None or crop_coords is None:
        st.error("Could not crop face properly. Please upload another image.")
        return None
        
    # Get face dimensions
    x_min, y_min, x_max, y_max = crop_coords
    
    # Run landmark detection on the cropped face for better accuracy
    face_crop_results = face_detector.detect_face(face_crop)
    
    if not face_crop_results.multi_face_landmarks:
        st.error("Could not detect facial landmarks on the cropped face. Please upload another image.")
        return None
    
    # Get eyebrow landmarks from the cropped face, passing crop coordinates
    left_eyebrow, right_eyebrow = face_detector.get_eyebrow_landmarks(image, results, crop_coords)
    
    # If landmarks were not detected, try with the original image
    if left_eyebrow is None or right_eyebrow is None:
        left_eyebrow, right_eyebrow = face_detector.get_eyebrow_landmarks(image, results)
        
        if left_eyebrow is None or right_eyebrow is None:
            st.error("Could not detect eyebrows. Please upload another image.")
            return None
    
    # Initialize face parsing visualization variables
    # These will be used only in the Face Parsing tab
    # We'll create temporary variables and add them to the results dictionary later
    face_parsing_success = False
    face_parsing_visualization = None
    face_parsing_masks = {}
    
    # We'll skip the face parsing for now since it's causing issues
    # The traditional eyebrow segmentation will be used for analysis
    
    # For eyebrow segmentation and analysis, use the traditional method
    # This ensures consistent results for color and shape analysis
    try:
        # Create eyebrow masks using traditional method
        left_mask, left_bbox = eyebrow_segmentation.create_eyebrow_mask(face_crop, left_eyebrow)
        right_mask, right_bbox = eyebrow_segmentation.create_eyebrow_mask(face_crop, right_eyebrow)
        
        # Refine masks
        left_refined_mask = eyebrow_segmentation.refine_eyebrow_mask(face_crop, left_mask)
        right_refined_mask = eyebrow_segmentation.refine_eyebrow_mask(face_crop, right_mask)
    except Exception as e:
        st.error(f"Eyebrow segmentation failed: {e}")
        return None
    
    # Get cropped masks
    left_cropped_mask = eyebrow_segmentation.get_cropped_mask(left_refined_mask, left_bbox)
    right_cropped_mask = eyebrow_segmentation.get_cropped_mask(right_refined_mask, right_bbox)
    
    # Extract eyebrow regions
    left_eyebrow_region, left_roi_mask = eyebrow_segmentation.extract_eyebrow_region(face_crop, left_refined_mask, left_bbox)
    right_eyebrow_region, right_roi_mask = eyebrow_segmentation.extract_eyebrow_region(face_crop, right_refined_mask, right_bbox)
    
    # Create alpha matting with cropping
    left_alpha, left_alpha_mask = eyebrow_segmentation.alpha_matting(face_crop, left_refined_mask, left_bbox)
    right_alpha, right_alpha_mask = eyebrow_segmentation.alpha_matting(face_crop, right_refined_mask, right_bbox)
    
    # Check if BiSeNet masks are available (they will be populated later in the BiSeNet tab)
    bisenet_masks_available = False
    
    # Extract dominant colors using traditional masks initially
    left_colors, left_percentages = color_analyzer.extract_dominant_colors(face_crop, left_refined_mask)
    right_colors, right_percentages = color_analyzer.extract_dominant_colors(face_crop, right_refined_mask)
    
    # Create color palettes
    left_palette = color_analyzer.create_color_palette(left_colors, left_percentages)
    right_palette = color_analyzer.create_color_palette(right_colors, right_percentages)
    
    # Get color information
    left_color_info = color_analyzer.get_color_info(left_colors, left_percentages)
    right_color_info = color_analyzer.get_color_info(right_colors, right_percentages)
    
    # Analyze color properties
    left_color_properties = color_analyzer.analyze_color_properties(left_colors)
    right_color_properties = color_analyzer.analyze_color_properties(right_colors)
    
    # Analyze eyebrow shape
    left_shape_info = shape_analyzer.analyze_eyebrow_shape(left_eyebrow)
    right_shape_info = shape_analyzer.analyze_eyebrow_shape(right_eyebrow)
    
    # Visualize shape analysis
    left_shape_vis = shape_analyzer.visualize_shape(face_crop, left_eyebrow, left_shape_info)
    right_shape_vis = shape_analyzer.visualize_shape(face_crop, right_eyebrow, right_shape_info)
    
    # Get shape descriptions
    left_shape_desc = shape_analyzer.get_shape_description(left_shape_info)
    right_shape_desc = shape_analyzer.get_shape_description(right_shape_info)
    
    # Draw landmarks on original image for visualization
    landmarks_image = face_detector.draw_landmarks(image, results)
    
    # Draw landmarks on cropped face for better visualization
    cropped_landmarks_image = face_detector.draw_landmarks(face_crop, face_crop_results)
    
    # Draw eyebrow landmarks on cropped face
    eyebrow_landmarks_image = face_detector.draw_eyebrow_landmarks(face_crop, left_eyebrow, right_eyebrow)
    
    # Return all processed data
    results = {
        'original_image': image,
        'face_crop': face_crop,
        'landmarks_image': landmarks_image,
        'cropped_landmarks_image': cropped_landmarks_image,
        'eyebrow_landmarks_image': eyebrow_landmarks_image,
        'left_eyebrow': left_eyebrow,
        'right_eyebrow': right_eyebrow,
        'face_parsing_visualization': face_parsing_visualization,
        'face_parsing_masks': face_parsing_masks,
        'face_parsing_success': face_parsing_success,
        'left_mask': left_refined_mask,
        'right_mask': right_refined_mask,
        'left_cropped_mask': left_cropped_mask,
        'right_cropped_mask': right_cropped_mask,
        'left_eyebrow_region': left_eyebrow_region,
        'right_eyebrow_region': right_eyebrow_region,
        'left_alpha': left_alpha,
        'right_alpha': right_alpha,
        'left_colors': left_colors,
        'right_colors': right_colors,
        'left_percentages': left_percentages,
        'right_percentages': right_percentages,
        'left_palette': left_palette,
        'right_palette': right_palette,
        'left_color_info': left_color_info,
        'right_color_info': right_color_info,
        'left_color_properties': left_color_properties,
        'right_color_properties': right_color_properties,
        'left_shape_info': left_shape_info,
        'right_shape_info': right_shape_info,
        'left_shape_vis': left_shape_vis,
        'right_shape_vis': right_shape_vis,
        'left_shape_desc': left_shape_desc,
        'right_shape_desc': right_shape_desc,
        'left_refined_mask': left_refined_mask,
        'right_refined_mask': right_refined_mask
    }
    
    # Face parsing results are already added to the results dictionary
    # No need to check for them here
        
    return results

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Process the uploaded image
if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Process the image
    with st.spinner('Processing image...'):
        results = process_image(image)
    
    if results:
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Color Analysis", "Shape Analysis", "Deep Segmentation (BiSeNet)", "Detailed View"])
        
        
        with tab1:
            # Overview tab
            st.header("Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                original_image_pil = cv2_to_pil(results['original_image'])
                if original_image_pil is not None:
                    st.image(original_image_pil, use_column_width=True)
            
            with col2:
                st.subheader("Face Detection")
                face_crop_pil = cv2_to_pil(results['face_crop'])
                if face_crop_pil is not None:
                    st.image(face_crop_pil, use_column_width=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Facial Landmarks (Full Image)")
                landmarks_image_pil = cv2_to_pil(results['landmarks_image'])
                if landmarks_image_pil is not None:
                    st.image(landmarks_image_pil, use_column_width=True)
                
                st.subheader("Facial Landmarks (Cropped Face)")
                cropped_landmarks_image_pil = cv2_to_pil(results['cropped_landmarks_image'])
                if cropped_landmarks_image_pil is not None:
                    st.image(cropped_landmarks_image_pil, use_column_width=True)
            
            with col4:
                st.subheader("Eyebrow Landmarks")
                eyebrow_landmarks_image_pil = cv2_to_pil(results['eyebrow_landmarks_image'])
                if eyebrow_landmarks_image_pil is not None:
                    st.image(eyebrow_landmarks_image_pil, use_column_width=True)
        with tab2:
            # Color Analysis tab
            st.header("Color Analysis")
            

            
            # Check if BiSeNet results are available
            bisenet_available = 'bisenet_left_mask' in results and 'bisenet_right_mask' in results
            
            # Add a toggle to switch between traditional and BiSeNet color analysis
            use_bisenet = False
            if bisenet_available:
                use_bisenet = st.checkbox("Use BiSeNet segmentation for color analysis", value=True, 
                                         help="Toggle to switch between traditional and BiSeNet-based color analysis")
            
            # Select the appropriate data based on the toggle
            if use_bisenet and bisenet_available:
                left_mask_display = results['bisenet_left_mask']
                right_mask_display = results['bisenet_right_mask']
                left_palette_display = results['bisenet_left_palette']
                right_palette_display = results['bisenet_right_palette']
                left_color_info_display = results['bisenet_left_color_info']
                right_color_info_display = results['bisenet_right_color_info']
                left_color_props = color_analyzer.analyze_color_properties(results['bisenet_left_colors']) if 'bisenet_left_colors' in results else []
                right_color_props = color_analyzer.analyze_color_properties(results['bisenet_right_colors']) if 'bisenet_right_colors' in results else []
                st.success("Using BiSeNet segmentation for more accurate color analysis")
            else:
                left_mask_display = results['left_refined_mask']
                right_mask_display = results['right_refined_mask']
                left_palette_display = results['left_palette']
                right_palette_display = results['right_palette']
                left_color_info_display = results['left_color_info']
                right_color_info_display = results['right_color_info']
                left_color_props = results['left_color_properties'] if 'left_color_properties' in results else []
                right_color_props = results['right_color_properties'] if 'right_color_properties' in results else []
                if bisenet_available:
                    st.info("Using traditional segmentation for color analysis")
            
            # Create combined mask for visualization
            combined_mask = None
            if left_mask_display is not None and right_mask_display is not None:
                combined_mask = cv2.bitwise_or(left_mask_display, right_mask_display)
                
                # Show combined mask
                st.subheader("Combined Eyebrow Mask")
                
                # Show original and masked images side by side
                col_orig, col_mask = st.columns(2)
                with col_orig:
                    st.subheader("Original Image")
                    st.image(cv2_to_pil(results['face_crop']) or Image.new('RGB', (100, 100)), use_column_width=True)
                
                with col_mask:
                    st.subheader("Combined Mask")
                    # Apply combined mask to original image
                    if combined_mask is not None:
                        masked_combined = cv2.bitwise_and(results['face_crop'], results['face_crop'], mask=combined_mask)
                        st.image(cv2_to_pil(masked_combined) or Image.new('RGB', (100, 100)), use_column_width=True)
            
            # Display eyebrow regions and color palettes
            col1, col2 = st.columns(2)
            
            # Right eyebrow of the person (appears on left side of the screen)
            with col1:
                st.subheader("Right Eyebrow of Person")
                
                # Show original and masked images side by side
                col_orig_right, col_mask_right = st.columns(2)
                
                with col_orig_right:
                    st.subheader("Original")
                    # Show the original image
                    orig_right_pil = cv2_to_pil(results['face_crop'])
                    if orig_right_pil is not None:
                        st.image(orig_right_pil, use_column_width=True)
                
                with col_mask_right:
                    st.subheader("Mask")
                    # Display the mask applied to the original image
                    if right_mask_display is not None:
                        # Apply mask to original image for better visualization
                        masked_right = cv2.bitwise_and(results['face_crop'], results['face_crop'], mask=right_mask_display)
                        masked_right_pil = cv2_to_pil(masked_right)
                        if masked_right_pil is not None:
                            st.image(masked_right_pil, use_column_width=True)
                
                st.subheader("Dominant Colors")
                if right_palette_display is not None:
                    st.image(right_palette_display, channels="BGR")
                
                st.subheader("Color Information")
                if right_color_info_display:
                    for i, color_info in enumerate(right_color_info_display):
                        st.write(f"Color {i+1}: {color_info['percentage']}")
                        st.write(f"RGB: {color_info['rgb']}, HEX: {color_info['hex']}")
                        st.write(f"LAB: {color_info['lab']}")
                        st.write(f"LCH: {color_info['lch']}")
                        st.write(f"HSV: {color_info['hsv']}")
                        # Create a color swatch using HTML
                        st.markdown(f"<div style='background-color: {color_info['hex']}; width: 100%; height: 30px; margin-bottom: 10px;'></div>", unsafe_allow_html=True)
                        
                        # Add color properties directly under each color
                        if right_color_props and i < len(right_color_props):
                            st.markdown("**Color Properties:**")
                            st.markdown(f"- Brightness: {right_color_props[i]['brightness']}")
                            st.markdown(f"- Saturation: {right_color_props[i]['saturation']}")
                            st.markdown(f"- Intensity: {right_color_props[i]['intensity']}")
                            st.markdown(f"- Tone: {right_color_props[i]['tone']}")
                            st.markdown("---")
            
            # Left eyebrow of the person (appears on right side of the screen)
            with col2:
                st.subheader("Left Eyebrow of Person")
                
                # Show original and masked images side by side
                col_orig_left, col_mask_left = st.columns(2)
                
                with col_orig_left:
                    st.subheader("Original")
                    # Show the original image
                    orig_left_pil = cv2_to_pil(results['face_crop'])
                    if orig_left_pil is not None:
                        st.image(orig_left_pil, use_column_width=True)
                
                with col_mask_left:
                    st.subheader("Mask")
                    # Display the mask applied to the original image
                    if left_mask_display is not None:
                        # Apply mask to original image for better visualization
                        masked_left = cv2.bitwise_and(results['face_crop'], results['face_crop'], mask=left_mask_display)
                        masked_left_pil = cv2_to_pil(masked_left)
                        if masked_left_pil is not None:
                            st.image(masked_left_pil, use_column_width=True)
                
                st.subheader("Dominant Colors")
                if left_palette_display is not None:
                    st.image(left_palette_display, channels="BGR")
                
                st.subheader("Color Information")
                if left_color_info_display:
                    for i, color_info in enumerate(left_color_info_display):
                        st.write(f"Color {i+1}: {color_info['percentage']}")
                        st.write(f"RGB: {color_info['rgb']}, HEX: {color_info['hex']}")
                        st.write(f"LAB: {color_info['lab']}")
                        st.write(f"LCH: {color_info['lch']}")
                        st.write(f"HSV: {color_info['hsv']}")
                        # Create a color swatch using HTML
                        st.markdown(f"<div style='background-color: {color_info['hex']}; width: 100%; height: 30px; margin-bottom: 10px;'></div>", unsafe_allow_html=True)
                        
                        # Add color properties directly under each color
                        if left_color_props and i < len(left_color_props):
                            st.markdown("**Color Properties:**")
                            st.markdown(f"- Brightness: {left_color_props[i]['brightness']}")
                            st.markdown(f"- Saturation: {left_color_props[i]['saturation']}")
                            st.markdown(f"- Intensity: {left_color_props[i]['intensity']}")
                            st.markdown(f"- Tone: {left_color_props[i]['tone']}")
                            st.markdown("---")
        
        with tab3:
            # Shape Analysis tab
            st.header("Shape Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Left Eyebrow Shape")
                left_shape_vis_pil = cv2_to_pil(results['left_shape_vis'])
                if left_shape_vis_pil is not None:
                    st.image(left_shape_vis_pil, use_column_width=True)
                
                if results['left_shape_info']:
                    st.subheader("Shape Description")
                    st.markdown(results['left_shape_desc'])
                    
                    st.subheader("Shape Metrics")
                    st.markdown(f"- Shape Type: **{results['left_shape_info']['shape']}**")
                    st.markdown(f"- Arch Type: **{results['left_shape_info']['arch_type']}**")
                    st.markdown(f"- Aspect Ratio: {results['left_shape_info']['aspect_ratio']:.2f}")
                    st.markdown(f"- Arch Height: {results['left_shape_info']['arch_height']:.2f} pixels")
                    st.markdown(f"- Thickness: {results['left_shape_info']['thickness']:.2f} pixels")
                    st.markdown(f"- Curvature: {results['left_shape_info']['curvature']:.2f} radians")
                    st.markdown(f"- Compactness: {results['left_shape_info']['compactness']:.2f}")
                    st.markdown(f"- Convexity: {results['left_shape_info']['convexity']:.2f}")
            
            with col2:
                st.subheader("Right Eyebrow Shape")
                right_shape_vis_pil = cv2_to_pil(results['right_shape_vis'])
                if right_shape_vis_pil is not None:
                    st.image(right_shape_vis_pil, use_column_width=True)
                
                if results['right_shape_info']:
                    st.subheader("Shape Description")
                    st.markdown(results['right_shape_desc'])
                    
                    st.subheader("Shape Metrics")
                    st.markdown(f"- Shape Type: **{results['right_shape_info']['shape']}**")
                    st.markdown(f"- Arch Type: **{results['right_shape_info']['arch_type']}**")
                    st.markdown(f"- Aspect Ratio: {results['right_shape_info']['aspect_ratio']:.2f}")
                    st.markdown(f"- Arch Height: {results['right_shape_info']['arch_height']:.2f} pixels")
                    st.markdown(f"- Thickness: {results['right_shape_info']['thickness']:.2f} pixels")
                    st.markdown(f"- Curvature: {results['right_shape_info']['curvature']:.2f} radians")
                    st.markdown(f"- Compactness: {results['right_shape_info']['compactness']:.2f}")
                    st.markdown(f"- Convexity: {results['right_shape_info']['convexity']:.2f}")
        
        with tab4:
            # Deep Eyebrow Segmentation (BiSeNet) tab
            st.header("Deep Eyebrow Segmentation (BiSeNet)")
            st.markdown("""
                This tab uses a deep neural network (BiSeNet) for state-of-the-art eyebrow segmentation. Results are robust for all skin tones and lighting conditions.
            """)
            # Create a placeholder for status messages
            status_placeholder = st.empty()
            status_placeholder.info("Preparing to run BiSeNet segmentation...")
            
            try:
                from bisenet_integration import segment_eyebrows_with_bisenet, create_eyebrow_overlay, extract_eyebrow_masks_from_raw_segmentation, extract_only_eyebrows
                import tempfile
                
                # Save cropped face image to a temp file
                cropped_face = results.get('face_crop', None)
                if cropped_face is None:
                    st.error('No cropped face available. Cannot run BiSeNet segmentation.')
                    st.stop()
                
                # Create a temporary file for the cropped face
                tmp_img_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_img:
                        status_placeholder.info("Saving temporary image for processing...")
                        cv2.imwrite(tmp_img.name, cropped_face)
                        tmp_img_path = tmp_img.name
                        status_placeholder.info(f"Temporary image saved at: {tmp_img_path}")
                except Exception as e:
                    st.error(f"Failed to create temporary file: {e}")
                    st.stop()
                
                # Show a spinner while processing
                with st.spinner('Running BiSeNet segmentation...'):
                    status_placeholder.info("Running BiSeNet segmentation - this may take a moment...")
                    
                    # Run BiSeNet segmentation using the simplified function
                    segmentation_image, raw_segmentation = segment_eyebrows_with_bisenet(tmp_img_path, target_shape=cropped_face.shape)
                    status_placeholder.success(f"BiSeNet segmentation completed successfully!")
                    
                    # Clean up the temporary file
                    if tmp_img_path and os.path.exists(tmp_img_path):
                        os.remove(tmp_img_path)
                
                # Display information about the segmentation image
                status_placeholder.info(f"Segmentation image shape: {segmentation_image.shape}, dtype: {segmentation_image.dtype}")
                status_placeholder.info(f"Raw segmentation shape: {raw_segmentation.shape}, dtype: {raw_segmentation.dtype}")
                
                # Store the raw segmentation for visualization
                results['bisenet_segmentation_image'] = segmentation_image
                results['bisenet_raw_segmentation'] = raw_segmentation
                
                # Create a more visually appealing segmentation visualization using the common approach
                from bisenet_integration import visualize_segmentation, extract_eyebrows_common_approach
                
                # Create enhanced visualization using the common approach from face-parsing/utils
                blended_visualization, eyebrow_mask = extract_eyebrows_common_approach(raw_segmentation, cropped_face)
                results['bisenet_enhanced_visualization'] = blended_visualization
                results['bisenet_eyebrow_mask'] = eyebrow_mask
                
                # Also create a standard visualization for comparison
                standard_visualization = visualize_segmentation(raw_segmentation)
                results['bisenet_standard_visualization'] = standard_visualization
                
                # Display the original image and segmentation side by side
                st.subheader("Original Image vs BiSeNet Segmentation")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cropped_face, channels="BGR", caption="Original Image")
                with col2:
                    st.image(segmentation_image, channels="BGR", caption="BiSeNet Face Parsing Result")
                
                # Display enhanced visualization
                st.subheader("Enhanced BiSeNet Visualization")
                if 'bisenet_enhanced_visualization' in results and results['bisenet_enhanced_visualization'] is not None:
                    enhanced_vis_pil = cv2_to_pil(results['bisenet_enhanced_visualization'])
                    if enhanced_vis_pil is not None:
                        st.image(enhanced_vis_pil, caption="Enhanced Segmentation Visualization")
                else:
                    st.warning("Enhanced visualization not available")
                
                # Display the segmentation classes explanation
                st.subheader("Segmentation Classes")
                st.markdown("""
                The BiSeNet face parsing model segments the face into different regions, each with a unique class index:
                
                - Skin (blue)
                - Eyebrows (purple/teal)
                - Eyes (green/pink)
                - Nose (red)
                - Lips (orange/red)
                - Hair (light green)
                
                The exact colors may vary depending on the model version.
                """)
                
                # Add information about the segmentation indices
                st.subheader("Technical Information")
                st.markdown("""
                The BiSeNet model used in this application produces class indices in the range 36-237, 
                which differs from the standard BiSeNet model that uses indices 0-15.
                
                Based on our analysis, the eyebrow classes in this model are approximately:
                - Left eyebrow: class 127
                - Right eyebrow: class 129
                
                However, the segmentation results may vary depending on lighting, pose, and other factors.
                """)
                
                # Add a button to print all indices to console for research
                if st.button("Print segmentation indices to console", key="bisenet_print_indices"):
                    unique_indices = np.unique(raw_segmentation)
                    print("\n\nAll unique indices in raw segmentation:")
                    print(unique_indices)
                    st.success("Indices printed to console. Check your terminal/console output.")
                
                # Add a note about the segmentation result
                st.markdown("---")
                st.write("BiSeNet provides a detailed facial segmentation with different colors representing different facial features.")
                st.write("The extracted eyebrow masks are used for more accurate color analysis.")
                st.write("You can use this segmentation as a reference for further analysis.")
                
                # Add a separator before the explanation
                st.markdown("---")
                
                # Add a debug visualization section
                st.header("Segmentation Debug Visualizations")
                
                # Add a toggle for showing debug visualizations
                show_debug = st.checkbox("Show debug visualizations", value=False)
                
                if show_debug and 'bisenet_raw_segmentation' in results:
                    # Create visualization of raw segmentation
                    from bisenet_integration import visualize_segmentation, create_mask_visualization
                    
                    # Display raw segmentation visualization
                    st.subheader("Raw Segmentation Classes")
                    raw_vis = visualize_segmentation(results['bisenet_raw_segmentation'])
                    st.image(raw_vis, channels="BGR")
                    
                    # Display mask visualizations
                    st.subheader("Eyebrow Mask Visualizations")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Left Eyebrow Mask")
                        if 'left_eyebrow_mask' in results:
                            # Create visualization of left eyebrow mask
                            left_vis = create_mask_visualization(
                                results['face_crop'], 
                                results['left_eyebrow_mask'], 
                                color=(255, 170, 0),  # Orange
                                alpha=0.5
                            )
                            st.image(left_vis, channels="BGR")
                        else:
                            st.warning("Left eyebrow mask not available.")
                    
                    with col2:
                        st.subheader("Right Eyebrow Mask")
                        if 'right_eyebrow_mask' in results:
                            # Create visualization of right eyebrow mask
                            right_vis = create_mask_visualization(
                                results['face_crop'], 
                                results['right_eyebrow_mask'], 
                                color=(255, 0, 85),  # Pink
                                alpha=0.5
                            )
                            st.image(right_vis, channels="BGR")
                        else:
                            st.warning("Right eyebrow mask not available.")
            except Exception as e:
                st.error(f"BiSeNet segmentation failed: {e}\nMake sure bisenet_integration.py, face-parsing, and weights are properly set up.")
        

        with tab5:
            # Detailed View tab
            st.header("Detailed View")
            
            st.subheader("Eyebrow Masks")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("Left Eyebrow Mask")
                if results['left_cropped_mask'] is not None:
                    st.image(results['left_cropped_mask'], use_column_width=True)
            
            with col2:
                st.markdown("Right Eyebrow Mask")
                if results['right_cropped_mask'] is not None:
                    st.image(results['right_cropped_mask'], use_column_width=True)
            
            st.subheader("Alpha Matting")
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("Left Eyebrow Alpha Matte")
                if results['left_alpha'] is not None:
                    left_alpha_pil = cv2_to_pil(results['left_alpha'])
                    if left_alpha_pil is not None:
                        st.image(left_alpha_pil, use_column_width=True)
            
            with col4:
                st.markdown("Right Eyebrow Alpha Matte")
                if results['right_alpha'] is not None:
                    right_alpha_pil = cv2_to_pil(results['right_alpha'])
                    if right_alpha_pil is not None:
                        st.image(right_alpha_pil, use_column_width=True)





# Add information about the app
st.sidebar.title("About")
st.sidebar.info("""
This app analyzes eyebrows in facial images using computer vision techniques.
It extracts information about:
- Eyebrow color and dominant shades
- Eyebrow shape characteristics
- Detailed visualization of eyebrow features

Upload a high-resolution facial image to get started.
""")

# Add instructions
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. Upload a high-resolution facial image
2. The app will automatically detect the face and eyebrows
3. View the analysis results in the different tabs
4. The "Overview" tab shows the basic detection results
5. The "Color Analysis" tab shows the dominant colors in the eyebrows
6. The "Shape Analysis" tab shows the shape characteristics
7. The "Detailed View" tab shows masks and alpha matting
""")

# Add technical notes
st.sidebar.title("Technical Notes")
st.sidebar.markdown("""
- Face detection and landmark extraction using MediaPipe
- Eyebrow segmentation using landmark-based masks
- Color analysis using K-means clustering
- Shape analysis using contour analysis and geometric calculations
- Alpha matting simulation for detailed visualization
""")

if __name__ == "__main__":
    # This will be executed when the script is run directly
    pass
