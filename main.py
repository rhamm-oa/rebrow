import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import io
import json
import plotly.graph_objects as go
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
# Import custom modules
from face_detection import FaceDetector
from eyebrow_segmentation import EyebrowSegmentation
from color_analysis import ColorAnalysis
from shape_analysis import ShapeAnalysis
from facer_segmentation import FacerSegmentation
from eyebrow_recoloring import EyebrowRecoloring

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
facer_segmenter = FacerSegmentation()
eyebrow_recoloring = EyebrowRecoloring()

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
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")

# Reset session state when a new image is uploaded
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

# Check if the uploaded file has changed
if uploaded_file is not None and (st.session_state.last_uploaded_file is None or 
                                 uploaded_file.name != st.session_state.last_uploaded_file):
    # Reset all caching variables when a new image is uploaded
    st.session_state.face_crop_cache = None
    st.session_state.left_mask_cache = None
    st.session_state.right_mask_cache = None
    st.session_state.recolored_images = {}
    st.session_state.has_run_facer = False
    if 'original_results' in st.session_state:
        del st.session_state.original_results
    
    # Update the last uploaded file
    st.session_state.last_uploaded_file = uploaded_file.name
    
    # Display a message about the new image
    st.success(f"New image loaded: {uploaded_file.name}")

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
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Shape Analysis", "Facer Segmentation", "Virtual Try-On", "Detailed View"])
        
        
        with tab1:
            # Overview tab
            st.header("Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                original_image_pil = cv2_to_pil(results['original_image'])
                if original_image_pil is not None:
                    st.image(original_image_pil, use_container_width=True)
            
            with col2:
                st.subheader("Face Detection")
                face_crop_pil = cv2_to_pil(results['face_crop'])
                if face_crop_pil is not None:
                    st.image(face_crop_pil, use_container_width=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Facial Landmarks (Full Image)")
                landmarks_image_pil = cv2_to_pil(results['landmarks_image'])
                if landmarks_image_pil is not None:
                    st.image(landmarks_image_pil, use_container_width=True)
                
                st.subheader("Facial Landmarks (Cropped Face)")
                cropped_landmarks_image_pil = cv2_to_pil(results['cropped_landmarks_image'])
                if cropped_landmarks_image_pil is not None:
                    st.image(cropped_landmarks_image_pil, use_container_width=True)
            
            with col4:
                st.subheader("Eyebrow Landmarks")
                eyebrow_landmarks_image_pil = cv2_to_pil(results['eyebrow_landmarks_image'])
                if eyebrow_landmarks_image_pil is not None:
                    st.image(eyebrow_landmarks_image_pil, use_container_width=True)
        
        with tab2:
            # Shape Analysis tab
            st.header("Shape Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Left Eyebrow Shape")
                left_shape_vis_pil = cv2_to_pil(results['left_shape_vis'])
                if left_shape_vis_pil is not None:
                    st.image(left_shape_vis_pil, use_container_width=True)
                
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
                    st.image(right_shape_vis_pil, use_container_width=True)
                
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

# Commented out BiSeNet code
# try:
#     from bisenet_integration import segment_eyebrows_with_bisenet, create_eyebrow_overlay, extract_eyebrow_masks_from_raw_segmentation, extract_only_eyebrows
#     import tempfile
#     # ... rest of BiSeNet code
# except Exception as e:
#     st.error(f"BiSeNet segmentation failed: {e}\nMake sure bisenet_integration.py, face-parsing, and weights are properly set up.")
        

        with tab3:
            # Facer Segmentation tab
            st.header("Facer Segmentation")
            st.markdown("""
                This tab uses the Facer library for advanced face parsing and eyebrow segmentation. 
                Facer provides more accurate segmentation than traditional methods.
            """)
            
            try:
                # Get the cropped face image
                cropped_face = results.get('face_crop', None)
                
                if cropped_face is not None:
                    # Process the cropped face with Facer
                    with st.spinner('Processing with Facer...'):
                        facer_segmenter = FacerSegmentation(use_gpu=True)
                        facer_result = facer_segmenter.segment_eyebrows(cropped_face, visualize=True)
                    
                    if facer_result.get('success', False):
                        # Get the visualization image
                        vis_img = facer_result.get('visualization_image')
                        if vis_img is not None:
                            # Convert BGR to RGB for correct display in Streamlit
                            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                            st.image(vis_img_rgb, caption="Facer Segmentation Results", use_container_width=True)
                        
                        # Get the eyebrow masks (still useful to store them for other tabs like color analysis)
                        left_eyebrow_mask = facer_result.get('left_eyebrow_mask')
                        right_eyebrow_mask = facer_result.get('right_eyebrow_mask')
                        combined_mask = facer_result.get('combined_eyebrow_mask')
                        
                        # Store masks in results dictionary for potential use elsewhere
                        results['facer_left_mask'] = left_eyebrow_mask
                        results['facer_right_mask'] = right_eyebrow_mask
                        
                        # Removed the display of individual eyebrow masks as per user request
                        
                        # # Display the extracted eyebrow area
                        # st.subheader("Extracted Eyebrows")
                        # if combined_mask is not None:
                        #     # Extract the eyebrow area
                        #     eyebrow_area = facer_segmenter.extract_eyebrow_area(cropped_face, combined_mask)
                        #     st.image(eyebrow_area, caption="Extracted Eyebrow Area", use_container_width=True)
                        
                        # --- Color Analysis Section ---
                        st.subheader("Eyebrow Color Analysis")
                        color_analyzer = ColorAnalysis()
                        
                        color_col1, color_col2 = st.columns(2)
                        
                        # Left Eyebrow Color Analysis (Now in color_col1)
                        with color_col1:
                            st.subheader("Left Eyebrow Colors")
                            if cropped_face is not None and left_eyebrow_mask is not None:
                                masked_left = cv2.bitwise_and(cropped_face, cropped_face, mask=left_eyebrow_mask)
                                masked_left_pil = cv2_to_pil(masked_left)
                                if masked_left_pil is not None:
                                    st.image(masked_left_pil, caption="Masked Left Eyebrow", use_container_width=True)
                                
                                left_colors, left_percentages, left_debug_images = color_analyzer.extract_reliable_hair_colors(cropped_face, left_eyebrow_mask, n_colors=3)
                                results['left_debug_images'] = left_debug_images
                                left_palette = color_analyzer.create_color_palette(left_colors, left_percentages)
                                left_color_info = color_analyzer.get_color_info(left_colors, left_percentages)
                                
                                if left_palette is not None:
                                    st.image(left_palette, channels="BGR", caption="Dominant Colors Palette (by percentage)") # Consistent caption
                                
                                if left_color_info:
                                    st.subheader("Detailed Color Analysis (sorted by percentage)") # Added subheader
                                    for i, color_info in enumerate(left_color_info):
                                        # Standardized layout with swatch on left
                                        detail_col_l1, detail_col_l2 = st.columns([1, 3])
                                        with detail_col_l1:
                                            st.markdown(f"<div style='background-color: {color_info['hex']}; width: 100%; height: 100px; border-radius: 5px;'></div>", unsafe_allow_html=True)
                                        with detail_col_l2:
                                            st.write(f"**Color {i+1}**: {color_info['percentage']}")
                                            st.write(f"RGB: {color_info['rgb']}, HEX: {color_info['hex']}")
                                            st.write(f"LAB: {color_info['lab']}")
                                            st.write(f"LCH: {color_info['lch']}")
                                            st.write(f"HSV: {color_info['hsv']}")
                                        st.markdown("---")
                                    
                                    # Added Interactive Visualizations for Left Eyebrow
                                    st.subheader("Interactive Color Visualizations")
                                    left_plotly_pie = color_analyzer.create_plotly_pie_chart(left_colors, left_percentages)
                                    if left_plotly_pie is not None:
                                        st.plotly_chart(go.Figure(json.loads(left_plotly_pie)), use_container_width=True)
                                    left_lab_3d = color_analyzer.create_plotly_lab_3d(left_colors, left_percentages)
                                    if left_lab_3d is not None:
                                        st.plotly_chart(go.Figure(json.loads(left_lab_3d)), use_container_width=True)

                                    # Debug view for Left Eyebrow
                                    if left_debug_images:
                                        with st.expander("View Hair Detection Process"):
                                            st.write("This shows how the algorithm isolates the hair strands for color analysis:")
                                            if 'original_masked' in left_debug_images:
                                                st.image(left_debug_images['original_masked'], caption="Original Masked Region", use_container_width=True)
                                            if 'otsu_mask' in left_debug_images:
                                                st.image(left_debug_images['otsu_mask'], caption="Otsu's Automatic Thresholding", use_container_width=True)
                                            if 'refined_mask' in left_debug_images:
                                                st.image(left_debug_images['refined_mask'], caption="Refined Hair Mask", use_container_width=True)
                                            if 'hair_pixels' in left_debug_images:
                                                st.image(left_debug_images['hair_pixels'], caption="Isolated Hair Pixels", use_container_width=True)
                                            if 'fallback_mask' in left_debug_images:
                                                st.image(left_debug_images['fallback_mask'], caption="Fallback Mask (Fixed Threshold)", use_container_width=True)
                                            if 'fallback_pixels' in left_debug_images:
                                                st.image(left_debug_images['fallback_pixels'], caption="Fallback Hair Pixels", use_container_width=True)
                                            if 'using_whole_mask' in left_debug_images:
                                                st.image(left_debug_images['using_whole_mask'], caption="Using Whole Mask (Last Resort)", use_container_width=True)
                                else:
                                    st.info("No color information available for the left eyebrow mask.")
                            else:
                                st.warning("Left eyebrow mask or face crop not available.")

                        # Right Eyebrow Color Analysis (Now in color_col2)
                        with color_col2:
                            st.subheader("Right Eyebrow Colors")
                            if cropped_face is not None and right_eyebrow_mask is not None:
                                masked_right = cv2.bitwise_and(cropped_face, cropped_face, mask=right_eyebrow_mask)
                                masked_right_pil = cv2_to_pil(masked_right)
                                if masked_right_pil is not None:
                                    st.image(masked_right_pil, caption="Masked Right Eyebrow", use_container_width=True)
                                
                                right_colors, right_percentages, right_debug_images = color_analyzer.extract_reliable_hair_colors(cropped_face, right_eyebrow_mask, n_colors=3)
                                results['right_debug_images'] = right_debug_images
                                right_palette = color_analyzer.create_color_palette(right_colors, right_percentages)
                                right_color_info = color_analyzer.get_color_info(right_colors, right_percentages)
                                
                                if right_palette is not None:
                                    st.image(right_palette, channels="BGR", caption="Dominant Colors Palette (by percentage)")
                                
                                if right_color_info:
                                    st.subheader("Detailed Color Analysis (sorted by percentage)")
                                    for i, color_info in enumerate(right_color_info):
                                        detail_col_r1, detail_col_r2 = st.columns([1, 3]) # Unique column names
                                        with detail_col_r1:
                                            st.markdown(f"<div style='background-color: {color_info['hex']}; width: 100%; height: 100px; border-radius: 5px;'></div>", unsafe_allow_html=True)
                                        with detail_col_r2:
                                            st.write(f"**Color {i+1}**: {color_info['percentage']}")
                                            st.write(f"RGB: {color_info['rgb']}, HEX: {color_info['hex']}")
                                            st.write(f"LAB: {color_info['lab']}")
                                            st.write(f"LCH: {color_info['lch']}")
                                            st.write(f"HSV: {color_info['hsv']}")
                                        st.markdown("---")
                                    
                                    st.subheader("Interactive Color Visualizations")
                                    right_plotly_pie = color_analyzer.create_plotly_pie_chart(right_colors, right_percentages)
                                    if right_plotly_pie is not None:
                                        st.plotly_chart(go.Figure(json.loads(right_plotly_pie)), use_container_width=True)
                                    right_lab_3d = color_analyzer.create_plotly_lab_3d(right_colors, right_percentages)
                                    if right_lab_3d is not None:
                                        st.plotly_chart(go.Figure(json.loads(right_lab_3d)), use_container_width=True)

                                    # Debug view for Right Eyebrow
                                    if right_debug_images:
                                        with st.expander("View Hair Detection Process"):
                                            st.write("This shows how the algorithm isolates the hair strands for color analysis:")
                                            if 'original_masked' in right_debug_images:
                                                st.image(right_debug_images['original_masked'], caption="Original Masked Region", use_container_width=True)
                                            if 'otsu_mask' in right_debug_images:
                                                st.image(right_debug_images['otsu_mask'], caption="Otsu's Automatic Thresholding", use_container_width=True)
                                            if 'refined_mask' in right_debug_images:
                                                st.image(right_debug_images['refined_mask'], caption="Refined Hair Mask", use_container_width=True)
                                            if 'hair_pixels' in right_debug_images:
                                                st.image(right_debug_images['hair_pixels'], caption="Isolated Hair Pixels", use_container_width=True)
                                            if 'fallback_mask' in right_debug_images:
                                                st.image(right_debug_images['fallback_mask'], caption="Fallback Mask (Fixed Threshold)", use_container_width=True)
                                            if 'fallback_pixels' in right_debug_images:
                                                st.image(right_debug_images['fallback_pixels'], caption="Fallback Hair Pixels", use_container_width=True)
                                            if 'using_whole_mask' in right_debug_images:
                                                st.image(right_debug_images['using_whole_mask'], caption="Using Whole Mask (Last Resort)", use_container_width=True)
                                else:
                                    st.info("No color information available for the right eyebrow mask.")
                            else:
                                st.warning("Right eyebrow mask or face crop not available.")
                        
                        # Technical information
                        with st.expander("Technical Information"):
                            st.markdown("""
                            **Facer Segmentation Details:**
                            - Model: FaRL (Face Representation Learning)
                            - Dataset: LaPa (Large-scale face parsing dataset)
                            - Resolution: 448 x 448
                            - Features: Provides detailed segmentation of facial features including eyebrows, eyes, nose, mouth, etc.
                            """)
                            
                            # Display the class mapping
                            class_mapping = facer_result.get('class_mapping', {})
                            if class_mapping:
                                st.markdown("**Class Mapping:**")
                                for part_name, class_idx in class_mapping.items():
                                    st.markdown(f"- {class_idx}: {part_name}")
                    else:
                        st.error(f"Facer segmentation failed: {facer_result.get('error', 'Unknown error')}")
                else:
                    st.error("No face crop available. Please make sure face detection succeeded.")
            except Exception as e:
                st.error(f"Facer segmentation failed: {e}\nMake sure facer_segmentation.py is properly set up and facer is installed.")
        
        with tab4:
            # Virtual Try-On tab
            st.header("Eyebrow Virtual Try-On")
            st.markdown("""
                This tab allows you to visualize how different eyebrow colors would look on your face.
                Adjust the opacity slider to control the intensity of the color effect.
            """)
            
            # Initialize session state for caching if not already done
            if 'face_crop_cache' not in st.session_state:
                st.session_state.face_crop_cache = None
            if 'left_mask_cache' not in st.session_state:
                st.session_state.left_mask_cache = None
            if 'right_mask_cache' not in st.session_state:
                st.session_state.right_mask_cache = None
            if 'recolored_images' not in st.session_state:
                st.session_state.recolored_images = {}
            if 'has_run_facer' not in st.session_state:
                st.session_state.has_run_facer = False
            
            try:
                # Only run Facer once per session
                if not st.session_state.has_run_facer:
                    st.session_state.face_crop_cache = results.get('face_crop', None)
                    st.session_state.left_mask_cache = results.get('facer_left_mask', None)
                    st.session_state.right_mask_cache = results.get('facer_right_mask', None)
                    st.session_state.has_run_facer = True
                    
                    # Store the original results to prevent recomputation
                    if 'original_results' not in st.session_state:
                        st.session_state.original_results = results.copy()
                else:
                    # Use the cached results instead of recomputing
                    results = st.session_state.original_results
                
                # Use the cached values
                cropped_face = st.session_state.face_crop_cache
                left_eyebrow_mask = st.session_state.left_mask_cache
                right_eyebrow_mask = st.session_state.right_mask_cache
                
                if cropped_face is not None and left_eyebrow_mask is not None and right_eyebrow_mask is not None:
                    # Display original image
                    st.subheader("Original Image")
                    original_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                    st.image(original_face_rgb, caption="Original Face", use_container_width=True)
                    
                    # Pre-defined color palette
                    color_palette = eyebrow_recoloring.create_color_palette(n_colors=6)
                    color_names = ["Dark Brown", "Medium Brown", "Light Brown", "Black", "Blonde", "Light Blonde"]
                    
                    # Opacity control with key to prevent rerunning
                    st.subheader("Color Intensity")
                    opacity = st.slider("Opacity", 0.0, 1.0, 0.8, 0.05,
                                      help="Control how strong the color effect is",
                                      key="opacity_slider")
                    
                    # Pre-render all color variations
                    st.subheader("Eyebrow Color Options")
                    
                    # Create two rows of 3 colors each
                    row1, row2 = st.columns(3), st.columns(3)
                    all_cols = [row1[0], row1[1], row1[2], row2[0], row2[1], row2[2]]
                    
                    # Clear cached images when opacity changes
                    current_opacity_key = f"current_opacity_{opacity}"
                    if 'last_opacity' not in st.session_state or st.session_state.last_opacity != opacity:
                        st.session_state.recolored_images = {}  # Clear cache when opacity changes
                        st.session_state.last_opacity = opacity
                    
                    for i, (color, name, col) in enumerate(zip(color_palette, color_names, all_cols)):
                        # Create a unique key for this color
                        color_key = f"{name}_{opacity}"
                        
                        # Check if we already have this color+opacity combination cached
                        if color_key not in st.session_state.recolored_images:
                            # Apply recoloring with this color
                            recolored = eyebrow_recoloring.recolor_both_eyebrows(
                                cropped_face, left_eyebrow_mask, right_eyebrow_mask, 
                                color, preserve_highlights=True, preserve_texture=True, opacity=opacity
                            )
                            # Convert to RGB and cache the result
                            recolored_rgb = cv2.cvtColor(recolored, cv2.COLOR_BGR2RGB)
                            st.session_state.recolored_images[color_key] = recolored_rgb
                        else:
                            # Use the cached result
                            recolored_rgb = st.session_state.recolored_images[color_key]
                        
                        # Display in the appropriate column
                        with col:
                            st.image(recolored_rgb, caption=name, use_container_width=True)
                            
                            # Show color swatch
                            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                            st.markdown(f"<div style='background-color: {hex_color}; width: 100%; height: 20px; border-radius: 5px; margin-bottom: 15px;'></div>", 
                                      unsafe_allow_html=True)
                        
                        # Technical explanation
                        with st.expander("How it works"):
                            st.markdown("""
                            The virtual try-on feature works by:
                            1. Using the accurate Facer segmentation masks to identify eyebrow pixels
                            2. Applying color transformations only to those pixels
                            3. Preserving the natural texture and highlights of your eyebrows
                            4. Blending the new color with the original image
                            
                            This creates a realistic preview of how different eyebrow colors would look on your face.
                            """)
                    else:
                        st.warning("Eyebrow masks not available. Please make sure Facer segmentation succeeded in the 'Facer Segmentation' tab.")
                else:
                    st.error("No face crop available. Please make sure face detection succeeded.")
            except Exception as e:
                st.error(f"Eyebrow recoloring failed: {e}\nMake sure eyebrow_recoloring.py is properly set up.")
        
        with tab5:
            # Detailed View tab
            st.header("Detailed View")
            st.markdown("This tab provides a comprehensive summary of the eyebrow analysis results.")
            
            # Summary section
            st.subheader("Analysis Summary")
            col_summary1, col_summary2 = st.columns(2)
            
            with col_summary1:
                st.markdown("### Left Eyebrow Summary")
                if 'left_shape_info' in results:
                    st.markdown(f"**Shape Type:** {results['left_shape_info'].get('shape', 'N/A')}")
                    st.markdown(f"**Arch Type:** {results['left_shape_info'].get('arch_type', 'N/A')}")
                    st.markdown(f"**Thickness:** {results['left_shape_info'].get('thickness', 0):.2f} pixels")
                    st.markdown(f"**Curvature:** {results['left_shape_info'].get('curvature', 0):.2f} radians")
                else:
                    st.info("Left eyebrow shape information not available.")
            
            with col_summary2:
                st.markdown("### Right Eyebrow Summary")
                if 'right_shape_info' in results:
                    st.markdown(f"**Shape Type:** {results['right_shape_info'].get('shape', 'N/A')}")
                    st.markdown(f"**Arch Type:** {results['right_shape_info'].get('arch_type', 'N/A')}")
                    st.markdown(f"**Thickness:** {results['right_shape_info'].get('thickness', 0):.2f} pixels")
                    st.markdown(f"**Curvature:** {results['right_shape_info'].get('curvature', 0):.2f} radians")
                else:
                    st.info("Right eyebrow shape information not available.")
            
            # Shape visualization recap
            st.subheader("Shape Analysis Recap")
            col_shape1, col_shape2 = st.columns(2)
            
            with col_shape1:
                st.markdown("### Left Eyebrow Shape")
                if 'left_shape_vis' in results and results['left_shape_vis'] is not None:
                    left_shape_vis_pil = cv2_to_pil(results['left_shape_vis'])
                    if left_shape_vis_pil is not None:
                        st.image(left_shape_vis_pil, use_container_width=True)
                else:
                    st.info("Left eyebrow shape visualization not available.")
            
            with col_shape2:
                st.markdown("### Right Eyebrow Shape")
                if 'right_shape_vis' in results and results['right_shape_vis'] is not None:
                    right_shape_vis_pil = cv2_to_pil(results['right_shape_vis'])
                    if right_shape_vis_pil is not None:
                        st.image(right_shape_vis_pil, use_container_width=True)
                else:
                    st.info("Right eyebrow shape visualization not available.")
            
            # Facer segmentation recap
            st.subheader("Facer Segmentation Recap")
            
            # Check if Facer results are available - using the correct key names from the Facer tab
            # The visualization image is stored in facer_result and accessed with 'visualization_image'
            facer_vis_img = None
            for result_key in results:
                if 'facer_result' in result_key and isinstance(results[result_key], dict):
                    facer_vis_img = results[result_key].get('visualization_image')
                    break
            
            # If we didn't find it that way, try the direct keys that might be stored
            if facer_vis_img is None:
                if 'visualization_image' in results:
                    facer_vis_img = results['visualization_image']
            
            # Display the visualization image if available
            if facer_vis_img is not None:
                st.image(facer_vis_img, caption="Facer Segmentation Results", use_container_width=True)
                
                # Display eyebrow masks if available
                col_facer1, col_facer2 = st.columns(2)
                
                with col_facer1:
                    st.markdown("### Left Eyebrow Mask (Facer)")
                    # Check for the correct mask key names
                    left_mask = None
                    if 'facer_left_mask' in results and results['facer_left_mask'] is not None:
                        left_mask = results['facer_left_mask']
                    
                    if left_mask is not None:
                        # Create masked image for display
                        cropped_face = results.get('face_crop', None)
                        if cropped_face is not None:
                            masked_left = cv2.bitwise_and(cropped_face, cropped_face, mask=left_mask)
                            masked_left_pil = cv2_to_pil(masked_left)
                            if masked_left_pil is not None:
                                st.image(masked_left_pil, use_container_width=True)
                    else:
                        st.info("Left eyebrow Facer mask not available.")
                
                with col_facer2:
                    st.markdown("### Right Eyebrow Mask (Facer)")
                    # Check for the correct mask key names
                    right_mask = None
                    if 'facer_right_mask' in results and results['facer_right_mask'] is not None:
                        right_mask = results['facer_right_mask']
                    
                    if right_mask is not None:
                        # Create masked image for display
                        cropped_face = results.get('face_crop', None)
                        if cropped_face is not None:
                            masked_right = cv2.bitwise_and(cropped_face, cropped_face, mask=right_mask)
                            masked_right_pil = cv2_to_pil(masked_right)
                            if masked_right_pil is not None:
                                st.image(masked_right_pil, use_container_width=True)
                    else:
                        st.info("Right eyebrow Facer mask not available.")
            else:
                # If we can't find the visualization image, look for the direct masks
                if ('facer_left_mask' in results and results['facer_left_mask'] is not None) or \
                   ('facer_right_mask' in results and results['facer_right_mask'] is not None):
                    
                    st.info("Facer segmentation visualization not available, but masks were found.")
                    
                    col_facer1, col_facer2 = st.columns(2)
                    
                    with col_facer1:
                        st.markdown("### Left Eyebrow Mask (Facer)")
                        if 'facer_left_mask' in results and results['facer_left_mask'] is not None:
                            # Display the mask directly
                            st.image(results['facer_left_mask'], caption="Left Eyebrow Mask", use_container_width=True)
                            
                            # Create masked image for display
                            cropped_face = results.get('face_crop', None)
                            if cropped_face is not None:
                                masked_left = cv2.bitwise_and(cropped_face, cropped_face, mask=results['facer_left_mask'])
                                masked_left_pil = cv2_to_pil(masked_left)
                                if masked_left_pil is not None:
                                    st.image(masked_left_pil, caption="Masked Left Eyebrow", use_container_width=True)
                        else:
                            st.info("Left eyebrow Facer mask not available.")
                    
                    with col_facer2:
                        st.markdown("### Right Eyebrow Mask (Facer)")
                        if 'facer_right_mask' in results and results['facer_right_mask'] is not None:
                            # Display the mask directly
                            st.image(results['facer_right_mask'], caption="Right Eyebrow Mask", use_container_width=True)
                            
                            # Create masked image for display
                            cropped_face = results.get('face_crop', None)
                            if cropped_face is not None:
                                masked_right = cv2.bitwise_and(cropped_face, cropped_face, mask=results['facer_right_mask'])
                                masked_right_pil = cv2_to_pil(masked_right)
                                if masked_right_pil is not None:
                                    st.image(masked_right_pil, caption="Masked Right Eyebrow", use_container_width=True)
                        else:
                            st.info("Right eyebrow Facer mask not available.")
                else:
                    st.info("Facer segmentation results not available. Please check the Facer Segmentation tab for details.")
            
            # Landmarks visualization
            st.subheader("Eyebrow Landmarks")
            if 'eyebrow_landmarks_image' in results and results['eyebrow_landmarks_image'] is not None:
                eyebrow_landmarks_image_pil = cv2_to_pil(results['eyebrow_landmarks_image'])
                if eyebrow_landmarks_image_pil is not None:
                    st.image(eyebrow_landmarks_image_pil, caption="Eyebrow Landmarks", use_container_width=True)
            else:
                st.info("Eyebrow landmarks visualization not available.")





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
