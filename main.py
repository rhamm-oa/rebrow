import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import io
import json
import plotly.graph_objects as go
import plotly.express as px
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
from eyebrow_statistics import EyebrowStatistics

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

# Initialize modules with metadata support
face_detector = FaceDetector()
eyebrow_segmentation = EyebrowSegmentation()
color_analyzer = ColorAnalysis(metadata_csv_path="data/MCB_DATA_MERGED.csv")
shape_analyzer = ShapeAnalysis()
facer_segmenter = FacerSegmentation()
eyebrow_stats = EyebrowStatistics()

# Load statistics data
try:
    eyebrow_stats.load_data('data/valid_results.csv')
except Exception as e:
    st.sidebar.error(f"Error loading statistics data: {e}")

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
    return cv2.cvtColor(np.array(pil_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)

# Modified process_image function to pass filename for metadata lookup
def process_image(image, filename=None):
    # Detect face landmarks
    face_detector = FaceDetector(use_gpu=True)
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
    
    # Get eyebrow landmarks (still needed for shape analysis)
    left_eyebrow, right_eyebrow = face_detector.get_eyebrow_landmarks(image, results, crop_coords)
    
    if left_eyebrow is None or right_eyebrow is None:
        left_eyebrow, right_eyebrow = face_detector.get_eyebrow_landmarks(image, results)
        
        if left_eyebrow is None or right_eyebrow is None:
            st.error("Could not detect eyebrows. Please upload another image.")
            return None
    
    # *** PRIMARY METHOD: Use Facer segmentation for masks ***
    try:
        facer_segmenter = FacerSegmentation(use_gpu=True)
        facer_result = facer_segmenter.segment_eyebrows(face_crop, visualize=True)
        
        if facer_result.get('success', False):
            # Use Facer masks as primary masks
            left_refined_mask = facer_result.get('left_eyebrow_mask')
            right_refined_mask = facer_result.get('right_eyebrow_mask')
            using_facer_masks = True
            st.session_state['facer_available'] = True
        else:
            # Fallback to traditional method only if Facer fails completely
            eyebrow_segmentation = EyebrowSegmentation()
            left_mask, left_bbox = eyebrow_segmentation.create_eyebrow_mask(face_crop, left_eyebrow)
            right_mask, right_bbox = eyebrow_segmentation.create_eyebrow_mask(face_crop, right_eyebrow)
            left_refined_mask = eyebrow_segmentation.refine_eyebrow_mask(face_crop, left_mask)
            right_refined_mask = eyebrow_segmentation.refine_eyebrow_mask(face_crop, right_mask)
            using_facer_masks = False
            st.session_state['facer_available'] = False
            st.warning("Facer segmentation failed, using traditional method as fallback")
            
    except Exception as e:
        st.warning(f"Facer segmentation failed: {e}. Using traditional method as fallback.")
        # Fallback to traditional method
        eyebrow_segmentation = EyebrowSegmentation()
        left_mask, left_bbox = eyebrow_segmentation.create_eyebrow_mask(face_crop, left_eyebrow)
        right_mask, right_bbox = eyebrow_segmentation.create_eyebrow_mask(face_crop, right_eyebrow)
        left_refined_mask = eyebrow_segmentation.refine_eyebrow_mask(face_crop, left_mask)
        right_refined_mask = eyebrow_segmentation.refine_eyebrow_mask(face_crop, right_mask)
        using_facer_masks = False
        st.session_state['facer_available'] = False
    
    # *** 🆕 ENHANCED COLOR ANALYSIS with metadata support ***
    # Extract colors using the enhanced method with filename for metadata lookup
    left_colors, left_percentages, left_debug_images = color_analyzer.extract_reliable_hair_colors(
        face_crop, left_refined_mask, n_colors=3, filename=filename)
    right_colors, right_percentages, right_debug_images = color_analyzer.extract_reliable_hair_colors(
        face_crop, right_refined_mask, n_colors=3, filename=filename)
    
    # Create color palettes and info
    left_palette = color_analyzer.create_color_palette(left_colors, left_percentages)
    right_palette = color_analyzer.create_color_palette(right_colors, right_percentages)
    left_color_info = color_analyzer.get_color_info(left_colors, left_percentages)
    right_color_info = color_analyzer.get_color_info(right_colors, right_percentages)
    left_color_properties = color_analyzer.analyze_color_properties(left_colors)
    right_color_properties = color_analyzer.analyze_color_properties(right_colors)
    
    # Analyze eyebrow shape (still uses landmarks)
    shape_analyzer = ShapeAnalysis()
    left_shape_info = shape_analyzer.analyze_eyebrow_shape(left_eyebrow)
    right_shape_info = shape_analyzer.analyze_eyebrow_shape(right_eyebrow)
    left_shape_vis = shape_analyzer.visualize_shape(face_crop, left_eyebrow, left_shape_info)
    right_shape_vis = shape_analyzer.visualize_shape(face_crop, right_eyebrow, right_shape_info)
    left_shape_desc = shape_analyzer.get_shape_description(left_shape_info)
    right_shape_desc = shape_analyzer.get_shape_description(right_shape_info)
    
    # Draw landmarks on original image for visualization
    landmarks_image = face_detector.draw_landmarks(image, results)
    cropped_landmarks_image = face_detector.draw_landmarks(face_crop, face_crop_results)
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
        'using_facer_masks': using_facer_masks,
        'facer_result': facer_result if 'facer_result' in locals() else None,
        'left_refined_mask': left_refined_mask,
        'right_refined_mask': right_refined_mask,
        'left_colors': left_colors,
        'right_colors': right_colors,
        'left_percentages': left_percentages,
        'right_percentages': right_percentages,
        'left_debug_images': left_debug_images,
        'right_debug_images': right_debug_images,
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
        'right_shape_desc': right_shape_desc
    }
    
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
    
    # 🆕 Display metadata banner if available
    if color_analyzer.metadata_handler:
        try:
            # Extract identifier from filename
            identifier = os.path.splitext(uploaded_file.name)[0]
            metadata = color_analyzer.metadata_handler.get_person_metadata(identifier)
            
            if metadata:
                ethnicity_display = {
                    'caucasian': '🏳️ Caucasian',
                    'black_african_american': '🏿 Black/African American', 
                    'asian': '🏯 Asian',
                    'hispanic_latino': '🌮 Hispanic/Latino',
                    'native_american': '🦬 Native American',
                    'others': '🌍 Other'
                }
                
                ethnicity_str = ethnicity_display.get(metadata['ethnicity_name'], '🌍 Unknown')
                age_str = f"📅 Age: {metadata['actual_age']}"
                skin_tone_str = f"🎨 Skin Cluster: {metadata['skin_cluster']}/6"
                
                st.info(f"""
                **Person Metadata Detected: {uploaded_file.name}**  
                {ethnicity_str} | {age_str} | {skin_tone_str}
                
                *Hair color analysis will be optimized for these characteristics*
                """)
            else:
                st.info(f"**New image loaded: {uploaded_file.name}** (No metadata found)")
        except Exception as e:
            st.success(f"New image loaded: {uploaded_file.name}")
    else:
        st.success(f"New image loaded: {uploaded_file.name}")

# Process the uploaded image
if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 🆕 Process the image with filename for metadata lookup
    with st.spinner('Processing image...'):
        results = process_image(image, filename=uploaded_file.name)
    
    if results:
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Shape Analysis", "Facer Segmentation+Color Analysis", "Statistics", "Debugging insights"])
        
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
        
        with tab3:
            # 🆕 Enhanced tab with metadata information
            st.header("Facer Segmentation + Enhanced Hair Color Analysis")
            
            # Check if we have Facer results
            using_facer = results.get('using_facer_masks', False)
            facer_result = results.get('facer_result')
            
            if using_facer and facer_result and facer_result.get('success', False):
                st.success("✅ Using Facer segmentation masks for accurate color analysis")
                
                # Display Facer visualization
                vis_img = facer_result.get('visualization_image')
                if vis_img is not None:
                    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                    st.image(vis_img_rgb, caption="Facer Segmentation Results", use_container_width=True)
                
                # Get the eyebrow masks
                left_eyebrow_mask = facer_result.get('left_eyebrow_mask')
                right_eyebrow_mask = facer_result.get('right_eyebrow_mask')
                
                # 🆕 Enhanced Color Analysis Section with metadata awareness
                st.subheader("🎨 Enhanced Hair-Only Color Analysis with Metadata Optimization")
                st.markdown("""
                **Algorithm Features:**
                - 🎯 **Hair-Only Detection**: Isolates actual hair strands, excluding skin tones
                - 🌈 **Multi-Method Approach**: Combines HSV thresholding, edge detection, texture analysis, and LAB color space
                - 🔍 **Adaptive Thresholding**: Automatically adjusts to different lighting conditions
                - 🧠 **Smart Fallbacks**: Multiple detection strategies ensure reliable results
                - 📊 **Metadata Optimization**: Uses ethnicity and skin cluster data for better accuracy
                """)
                
                color_col1, color_col2 = st.columns(2)
                
                # Left Eyebrow Enhanced Analysis
                with color_col1:
                    st.subheader("👈 Left Eyebrow Analysis")
                    
                    # Get the analysis results
                    left_colors = results.get('left_colors')
                    left_percentages = results.get('left_percentages') 
                    left_debug_images = results.get('left_debug_images', {})
                    
                    # 🆕 Show metadata info if available
                    if '0_metadata_applied' in left_debug_images:
                        st.success(left_debug_images['0_metadata_applied'])
                    elif '0_metadata' in left_debug_images:
                        st.info(left_debug_images['0_metadata'])
                    
                    if left_colors is not None and left_percentages is not None:
                        # Display color palette
                        left_palette = results.get('left_palette')
                        if left_palette is not None:
                            st.image(left_palette, channels="RGB", caption="🎨 Pure Hair Colors (No Skin Tones)", use_container_width=True)
                        
                        # Color information
                        left_color_info = results.get('left_color_info', [])
                        if left_color_info:
                            st.subheader("📊 Detected Hair Colors")
                            for i, color_info in enumerate(left_color_info):
                                detail_col_l1, detail_col_l2 = st.columns([1, 3])
                                with detail_col_l1:
                                    st.markdown(f"<div style='background-color: {color_info['hex']}; width: 100%; height: 80px; border-radius: 5px; border: 2px solid #333;'></div>", unsafe_allow_html=True)
                                with detail_col_l2:
                                    st.write(f"**Hair Color {i+1}**: {color_info['percentage']}")
                                    st.write(f"🎨 RGB: {color_info['rgb']}")
                                    st.write(f"🏷️ HEX: {color_info['hex']}")
                                    st.write(f"🔬 LAB: {color_info['lab']}")
                                st.markdown("---")
                        
                        # Interactive visualizations
                        st.subheader("📈 Interactive Visualizations")
                        left_plotly_pie = color_analyzer.create_plotly_pie_chart(left_colors, left_percentages)
                        if left_plotly_pie is not None:
                            st.plotly_chart(go.Figure(json.loads(left_plotly_pie)), use_container_width=True)

                        left_plotly_lab_3d = color_analyzer.create_plotly_lab_3d(left_colors, left_percentages)
                        if left_plotly_lab_3d is not None:
                            st.plotly_chart(go.Figure(json.loads(left_plotly_lab_3d)), use_container_width=True)

                        # Enhanced Debug Process
                        with st.expander("🔬 **Enhanced Hair Detection Process**", expanded=False):
                            st.markdown("""
                            **This enhanced algorithm uses 4 different methods to isolate hair pixels:**
                            1. **HSV Analysis**: Detects dark pixels (low brightness)
                            2. **Edge Detection**: Finds hair strand boundaries  
                            3. **Texture Analysis**: Identifies textured areas (hair vs smooth skin)
                            4. **LAB Color Space**: Additional lightness-based filtering
                            5. **Metadata Optimization**: Adjusts thresholds based on ethnicity and skin cluster
                            """)
                            
                            # Display debug images in order
                            debug_keys = [
                                ('1_masked_original', 'Original Masked Region'),
                                ('3_hsv_value_mask', 'HSV: Dark Pixel Detection'),
                                ('6_edge_detection', 'Edge Detection: Hair Boundaries'),
                                ('8_texture_mask', 'Texture Analysis: Hair vs Skin'),
                                ('9_lab_lightness_mask', 'LAB: Lightness Filtering'),
                                ('11_final_hair_mask', 'Final Combined Hair Mask'),
                                ('12_detected_hair_pixels', 'Final Detected Hair Pixels'),
                            ]
                            
                            for key, title in debug_keys:
                                if key in left_debug_images:
                                    st.write(f"**{title}**")
                                    st.image(left_debug_images[key], use_container_width=True)
                            
                            # Show enhanced results
                            if '13_cluster_validation' in left_debug_images:
                                st.success(left_debug_images['13_cluster_validation'])
                            
                            if '14_lab_values' in left_debug_images:
                                st.info(f"**LAB Values:** {left_debug_images['14_lab_values']}")
                    else:
                        st.warning("⚠️ Could not extract hair colors from left eyebrow")
                
                # Right Eyebrow Enhanced Analysis  
                with color_col2:
                    st.subheader("👉 Right Eyebrow Analysis")
                    
                    # Get the analysis results
                    right_colors = results.get('right_colors')
                    right_percentages = results.get('right_percentages')
                    right_debug_images = results.get('right_debug_images', {})
                    
                    # 🆕 Show metadata info if available
                    if '0_metadata_applied' in right_debug_images:
                        st.success(right_debug_images['0_metadata_applied'])
                    elif '0_metadata' in right_debug_images:
                        st.info(right_debug_images['0_metadata'])
                    
                    if right_colors is not None and right_percentages is not None:
                        # Display color palette
                        right_palette = results.get('right_palette')
                        if right_palette is not None:
                            st.image(right_palette, channels="RGB", caption="🎨 Pure Hair Colors (No Skin Tones)", use_container_width=True)
                        
                        # Color information
                        right_color_info = results.get('right_color_info', [])
                        if right_color_info:
                            st.subheader("📊 Detected Hair Colors")
                            for i, color_info in enumerate(right_color_info):
                                detail_col_r1, detail_col_r2 = st.columns([1, 3])
                                with detail_col_r1:
                                    st.markdown(f"<div style='background-color: {color_info['hex']}; width: 100%; height: 80px; border-radius: 5px; border: 2px solid #333;'></div>", unsafe_allow_html=True)
                                with detail_col_r2:
                                    st.write(f"**Hair Color {i+1}**: {color_info['percentage']}")
                                    st.write(f"🎨 RGB: {color_info['rgb']}")
                                    st.write(f"🏷️ HEX: {color_info['hex']}")
                                    st.write(f"🔬 LAB: {color_info['lab']}")
                                st.markdown("---")
                        
                        # Interactive visualizations
                        st.subheader("📈 Interactive Visualizations")
                        right_plotly_pie = color_analyzer.create_plotly_pie_chart(right_colors, right_percentages)
                        if right_plotly_pie is not None:
                            st.plotly_chart(go.Figure(json.loads(right_plotly_pie)), use_container_width=True)

                        right_plotly_lab_3d = color_analyzer.create_plotly_lab_3d(right_colors, right_percentages)
                        if right_plotly_lab_3d is not None:
                            st.plotly_chart(go.Figure(json.loads(right_plotly_lab_3d)), use_container_width=True)

                        # Enhanced Debug Process
                        with st.expander("🔬 **Enhanced Hair Detection Process**", expanded=False):
                            st.markdown("""
                            **This enhanced algorithm uses 4 different methods to isolate hair pixels:**
                            1. **HSV Analysis**: Detects dark pixels (low brightness)
                            2. **Edge Detection**: Finds hair strand boundaries  
                            3. **Texture Analysis**: Identifies textured areas (hair vs smooth skin)
                            4. **LAB Color Space**: Additional lightness-based filtering
                            5. **Metadata Optimization**: Adjusts thresholds based on ethnicity and skin cluster
                            """)
                            
                            # Display debug images in order
                            debug_keys = [
                                ('1_masked_original', 'Original Masked Region'),
                                ('3_hsv_value_mask', 'HSV: Dark Pixel Detection'),
                                ('6_edge_detection', 'Edge Detection: Hair Boundaries'),
                                ('8_texture_mask', 'Texture Analysis: Hair vs Skin'),
                                ('9_lab_lightness_mask', 'LAB: Lightness Filtering'),
                                ('11_final_hair_mask', 'Final Combined Hair Mask'),
                                ('12_detected_hair_pixels', 'Final Detected Hair Pixels'),
                            ]
                            
                            for key, title in debug_keys:
                                if key in right_debug_images:
                                    st.write(f"**{title}**")
                                    st.image(right_debug_images[key], use_container_width=True)
                            
                            # Show enhanced results
                            if '13_cluster_validation' in right_debug_images:
                                st.success(right_debug_images['13_cluster_validation'])
                            
                            if '14_lab_values' in right_debug_images:
                                st.info(f"**LAB Values:** {right_debug_images['14_lab_values']}")
                    else:
                        st.warning("⚠️ Could not extract hair colors from right eyebrow")
                
                # Technical Information
                with st.expander("🔧 Technical Information"):
                    st.markdown("""
                    ### **Enhanced Hair Detection Algorithm with Metadata Optimization**
                    
                    **Problem Solved**: Previous methods were detecting skin tones mixed with hair colors, leading to incorrect light brown results instead of the actual darker hair colors.
                    
                    **Solution**: Multi-method approach that focuses exclusively on hair strands:
                    
                    1. **HSV Color Space Analysis**:
                    - Uses Value (brightness) channel to detect dark pixels
                    - Adaptive thresholding based on image content
                    - Filters out bright skin tones automatically
                    
                    2. **Edge Detection (Canny)**:
                    - Detects hair strand boundaries and fine textures
                    - Bilateral filtering preserves edges while reducing noise
                    - Morphological operations connect broken hair strands
                    
                    3. **Texture Analysis**:
                    - Calculates local variance to distinguish textured (hair) vs smooth (skin) areas
                    - Uses sliding window approach for local texture measurement
                    - High variance = hair texture, low variance = smooth skin
                    
                    4. **LAB Color Space**:
                    - Uses L channel (lightness) for additional dark pixel detection
                    - More perceptually uniform than RGB for color analysis
                    - Provides robust hair vs skin separation
                    
                    5. **🆕 Metadata Optimization**:
                    - Uses ethnicity and skin cluster data from your CSV file
                    - Automatically adjusts thresholds for different ethnic groups
                    - Optimizes detection for specific skin tones (clusters 1-6)
                    - Applies conservative enhancements for dark skin + dark hair combinations
                    
                    **Fallback Strategies**: If one method fails, the algorithm automatically tries alternative approaches to ensure reliable results.
                    
                    **Quality Assurance**: Morphological operations clean up the final mask and remove noise while preserving hair details.
                    """)
                
            else:
                st.error("❌ Facer segmentation failed. Cannot perform enhanced color analysis.")
                st.markdown("**Possible solutions:**")
                st.markdown("- Ensure the image contains a clear, well-lit face")
                st.markdown("- Try uploading a higher resolution image")
                st.markdown("- Make sure the eyebrows are clearly visible")

        # Statistics Tab (unchanged)
        with tab4:
            st.header("Eyebrow Color Statistics")
            st.write("Analysis of eyebrow colors across the dataset")
            
            # Create tabs within the Statistics tab
            stat_tab1, stat_tab2, stat_tab4 = st.tabs(["Color Swatches", "3D Scatter", "Summary Stats"])
            
            # Color Swatches tab
            with stat_tab1:
                st.subheader("Dominant Eyebrow Colors")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Left Eyebrow - Dominant Colors")
                    left_swatches = eyebrow_stats.create_dominant_color_swatches('left')
                    if left_swatches:
                        st.image(left_swatches, use_container_width=True)
                        
                with col2:
                    st.write("Right Eyebrow - Dominant Colors")
                    right_swatches = eyebrow_stats.create_dominant_color_swatches('right')
                    if right_swatches:
                        st.image(right_swatches, use_container_width=True)
                
                st.subheader("All Colors Distribution")
                color_dist_fig = eyebrow_stats.create_color_distribution_plot()
                if color_dist_fig:
                    st.plotly_chart(color_dist_fig, use_container_width=True)
            
            # 3D Scatter tab
            with stat_tab2:
                st.subheader("3D LAB Color Space Visualization")
                col1, col2 = st.columns(2)
                
                # Left eyebrow plots
                with col1:
                    st.write("Left Eyebrow")
                    for i in range(1, 4):
                        fig = eyebrow_stats.create_3d_color_scatter('left', i)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    hist_fig = eyebrow_stats.create_percentage_histogram('left')
                    if hist_fig:
                        st.plotly_chart(hist_fig, use_container_width=True)
                
                # Right eyebrow plots
                with col2:
                    st.write("Right Eyebrow")
                    for i in range(1, 4):
                        fig = eyebrow_stats.create_3d_color_scatter('right', i)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    hist_fig = eyebrow_stats.create_percentage_histogram('right')
                    if hist_fig:
                        st.plotly_chart(hist_fig, use_container_width=True)
            
            # Summary Statistics tab
            with stat_tab4:
                st.subheader("Summary Statistics")
                stats_df = eyebrow_stats.get_summary_statistics()
                if stats_df is not None:
                    # Display color swatches in the dataframe
                    st.write("Average LAB values and percentages for each dominant color")
                    
                    # Display the dataframe with hex colors
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Create a visual representation of the colors
                    st.subheader("Color Swatches")
                    for i, row in stats_df.iterrows():
                        st.markdown(f"**{row['Side']} Eyebrow - Color {row['Color']}**")
                        st.markdown(f"<div style='background-color: {row['Hex Color']}; width: 100px; height: 50px; border: 1px solid #000;'></div>", unsafe_allow_html=True)
                        st.write(f"L: {row['Avg L']:.2f}, a: {row['Avg a']:.2f}, b: {row['Avg b']:.2f}, %: {row['Avg %']:.2f}")
                        st.write("---")
                    
                    # Create a bar chart of percentages
                    st.write("Average color percentages")
                    fig = px.bar(stats_df, 
                                x='Color', 
                                y='Avg %', 
                                color='Side', 
                                barmode='group',
                                title="Average Color Percentages by Side and Dominance",
                                labels={'Avg %': 'Average Percentage', 'Color': 'Dominant Color Number'})
                    st.plotly_chart(fig, use_container_width=True)


        with tab5:
            st.header("Debugging Insights")
            st.markdown("""
            This tab provides detailed debugging information for the eyebrow hair detection process, including:
            - Intermediate masks and filters
            - Detected hair pixels
            - Comparison of different methods (e.g., Gabor filters, LBP, etc.)
            """)

            # Debugging for Left Eyebrow
            st.subheader("👈 Left Eyebrow Debugging")
            left_debug_images = results.get('left_debug_images', {})
            if left_debug_images:
                for key, image in left_debug_images.items():
                    st.write(f"**{key}**")  # Display the key as the title
                    if isinstance(image, np.ndarray):  # Check if it's a NumPy array
                        st.image(image, use_container_width=True)
                    elif isinstance(image, str):  # If it's a string (e.g., a note), display it as text
                        st.write(image)
                    st.markdown("---")
            else:
                st.warning("No debugging information available for the left eyebrow.")

            # Debugging for Right Eyebrow
            st.subheader("👉 Right Eyebrow Debugging")
            right_debug_images = results.get('right_debug_images', {})
            if right_debug_images:
                for key, image in right_debug_images.items():
                    st.write(f"**{key}**")  # Display the key as the title
                    if isinstance(image, np.ndarray):  # Check if it's a NumPy array
                        st.image(image, use_container_width=True)
                    elif isinstance(image, str):  # If it's a string (e.g., a note), display it as text
                        st.write(image)
                    st.markdown("---")
            else:
                st.warning("No debugging information available for the right eyebrow.")

            # Comparison of Methods
            st.subheader("🔬 Comparison of Methods")
            st.markdown("""
            This section compares different approaches for eyebrow hair detection:
            - **HSV Thresholding**
            - **Edge Detection**
            - **Gabor Filters**
            - **LBP (Local Binary Patterns)**
            - **LAB Color Space**
            """)

            # Display comparison images (if available)
            comparison_images = {
                "HSV Thresholding": left_debug_images.get("3_hsv_value_mask"),
                "Edge Detection": left_debug_images.get("6_edge_detection"),
                "Gabor Filters": left_debug_images.get("gabor_response"),
                "LBP Analysis": left_debug_images.get("lbp_response"),
                "LAB Filtering": left_debug_images.get("9_lab_lightness_mask"),
            }

            for method, image in comparison_images.items():
                if image is not None:
                    st.write(f"**{method}**")
                    st.image(image, use_container_width=True)
                else:
                    st.warning(f"No debug image available for {method}.")


# Add information about the app (unchanged)
st.sidebar.title("About")
st.sidebar.info("""
This app analyzes eyebrows in facial images using computer vision techniques.
It extracts information about:
- Eyebrow color and dominant shades
- Eyebrow shape characteristics
- Detailed visualization of eyebrow features
- Metadata-optimized analysis based on ethnicity and skin tone

Upload a high-resolution facial image to get started.
""")

# Add instructions (updated)
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. Upload a high-resolution facial image (preferably named with ID from dataset)
2. The app will automatically detect the face and eyebrows
3. If metadata is found, analysis will be optimized for ethnicity and skin tone
4. View the analysis results in the different tabs
5. The "Overview" tab shows the basic detection results
6. The "Color Analysis" tab shows enhanced hair color detection with metadata optimization
7. The "Shape Analysis" tab shows the shape characteristics
8. The "Statistics" tab provides detailed statistics and visualizations of eyebrow colors
""")

# Add technical notes (updated)
st.sidebar.title("Technical Notes")
st.sidebar.markdown("""
- Face detection and landmark extraction using MediaPipe
- Eyebrow segmentation using Facer neural network + fallback methods
- Enhanced hair color analysis with metadata optimization
- Ethnicity and skin cluster-aware thresholding
- Shape analysis using contour analysis and geometric calculations
- Multi-method hair detection (HSV, LAB, edges, texture)
""")

if __name__ == "__main__":
    # This will be executed when the script is run directly
    pass