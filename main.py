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
    
    # Create eyebrow masks
    left_mask, left_bbox = eyebrow_segmentation.create_eyebrow_mask(face_crop, left_eyebrow)
    right_mask, right_bbox = eyebrow_segmentation.create_eyebrow_mask(face_crop, right_eyebrow)
    
    # Refine masks
    left_refined_mask = eyebrow_segmentation.refine_eyebrow_mask(face_crop, left_mask)
    right_refined_mask = eyebrow_segmentation.refine_eyebrow_mask(face_crop, right_mask)
    
    # Get cropped masks
    left_cropped_mask = eyebrow_segmentation.get_cropped_mask(left_refined_mask, left_bbox)
    right_cropped_mask = eyebrow_segmentation.get_cropped_mask(right_refined_mask, right_bbox)
    
    # Extract eyebrow regions
    left_eyebrow_region, left_roi_mask = eyebrow_segmentation.extract_eyebrow_region(face_crop, left_refined_mask, left_bbox)
    right_eyebrow_region, right_roi_mask = eyebrow_segmentation.extract_eyebrow_region(face_crop, right_refined_mask, right_bbox)
    
    # Create alpha matting with cropping
    left_alpha, left_alpha_mask = eyebrow_segmentation.alpha_matting(face_crop, left_refined_mask, left_bbox)
    right_alpha, right_alpha_mask = eyebrow_segmentation.alpha_matting(face_crop, right_refined_mask, right_bbox)
    
    # Extract dominant colors
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
    return {
        'original_image': image,
        'face_crop': face_crop,
        'landmarks_image': landmarks_image,
        'cropped_landmarks_image': cropped_landmarks_image,
        'eyebrow_landmarks_image': eyebrow_landmarks_image,
        'left_eyebrow': left_eyebrow,
        'right_eyebrow': right_eyebrow,
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
        'right_shape_desc': right_shape_desc
    }

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
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Color Analysis", "Shape Analysis", "Detailed View"])
        
        with tab1:
            # Overview tab
            st.header("Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(cv2_to_pil(results['original_image']), use_column_width=True) # type: ignore
            
            with col2:
                st.subheader("Face Detection")
                st.image(cv2_to_pil(results['face_crop']), use_column_width=True) # type: ignore
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Facial Landmarks (Full Image)")
                st.image(cv2_to_pil(results['landmarks_image']), use_column_width=True) # type: ignore
                
                st.subheader("Facial Landmarks (Cropped Face)")
                st.image(cv2_to_pil(results['cropped_landmarks_image']), use_column_width=True) # type: ignore
            
            with col4:
                st.subheader("Eyebrow Landmarks")
                st.image(cv2_to_pil(results['eyebrow_landmarks_image']), use_column_width=True) # type: ignore
        
        with tab2:
            # Color Analysis tab
            st.header("Color Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Left Eyebrow")
                st.image(cv2_to_pil(results['left_eyebrow_region']), use_column_width=True) # type: ignore
                
                if results['left_palette'] is not None:
                    st.subheader("Dominant Colors")
                    st.image(cv2_to_pil(results['left_palette']), use_column_width=True) # type: ignore
                    
                    st.subheader("Color Information")
                    for i, color_info in enumerate(results['left_color_info']):
                        st.markdown(f"**Color {i+1}**: {color_info['percentage']}")
                        st.markdown(f"RGB: {color_info['rgb']}, HEX: `{color_info['hex']}`")
                        st.markdown(f"LAB: {color_info['lab']}")
                        st.markdown(f"LCH: {color_info['lch']}")
                        st.markdown(f"HSV: {color_info['hsv']}")
                        st.markdown(f"<div style='background-color: {color_info['hex']}; width: 100%; height: 20px;'></div>", unsafe_allow_html=True)
                    
                    st.subheader("Color Properties")
                    for i, properties in enumerate(results['left_color_properties']):
                        st.markdown(f"**Color {i+1}**:")
                        st.markdown(f"- Brightness: {properties['brightness']}")
                        st.markdown(f"- Saturation: {properties['saturation']}")
                        st.markdown(f"- Intensity: {properties['intensity']}")
                        st.markdown(f"- Tone: {properties['tone']}")
            
            with col2:
                st.subheader("Right Eyebrow")
                st.image(cv2_to_pil(results['right_eyebrow_region']), use_column_width=True) # type: ignore
                
                if results['right_palette'] is not None:
                    st.subheader("Dominant Colors")
                    st.image(cv2_to_pil(results['right_palette']), use_column_width=True) # type: ignore
                    
                    st.subheader("Color Information")
                    for i, color_info in enumerate(results['right_color_info']):
                        st.markdown(f"**Color {i+1}**: {color_info['percentage']}")
                        st.markdown(f"RGB: {color_info['rgb']}, HEX: `{color_info['hex']}`")
                        st.markdown(f"LAB: {color_info['lab']}")
                        st.markdown(f"LCH: {color_info['lch']}")
                        st.markdown(f"HSV: {color_info['hsv']}")
                        st.markdown(f"<div style='background-color: {color_info['hex']}; width: 100%; height: 20px;'></div>", unsafe_allow_html=True)
                    
                    st.subheader("Color Properties")
                    for i, properties in enumerate(results['right_color_properties']):
                        st.markdown(f"**Color {i+1}**:")
                        st.markdown(f"- Brightness: {properties['brightness']}")
                        st.markdown(f"- Saturation: {properties['saturation']}")
                        st.markdown(f"- Intensity: {properties['intensity']}")
                        st.markdown(f"- Tone: {properties['tone']}")
        
        with tab3:
            # Shape Analysis tab
            st.header("Shape Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Left Eyebrow Shape")
                st.image(cv2_to_pil(results['left_shape_vis']), use_column_width=True) # type: ignore
                
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
                st.image(cv2_to_pil(results['right_shape_vis']), use_column_width=True) # type: ignore
                
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
                    st.image(cv2_to_pil(results['left_alpha']), use_column_width=True) # type: ignore
            
            with col4:
                st.markdown("Right Eyebrow Alpha Matte")
                if results['right_alpha'] is not None:
                    st.image(cv2_to_pil(results['right_alpha']), use_column_width=True) # type: ignore

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
