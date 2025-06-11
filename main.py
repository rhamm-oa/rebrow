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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
st.title("🔬 Advanced Eyebrow Analysis App")
st.markdown("""
This app analyzes eyebrows in facial images using **multiple robust detection methods** to extract insights about:
- 🎨 Eyebrow color (using 11 different detection methods)
- 📐 Eyebrow shape and characteristics  
- 🔬 Detailed debugging and method comparison
- 📊 Comprehensive visualization of results
""")

# Initialize modules with metadata support
@st.cache_resource
def initialize_modules():
    """Initialize all modules once and cache them"""
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
    
    return {
        'face_detector': face_detector,
        'eyebrow_segmentation': eyebrow_segmentation,
        'color_analyzer': color_analyzer,
        'shape_analyzer': shape_analyzer,
        'facer_segmenter': facer_segmenter,
        'eyebrow_stats': eyebrow_stats
    }

# Initialize modules
modules = initialize_modules()
face_detector = modules['face_detector']
eyebrow_segmentation = modules['eyebrow_segmentation']
color_analyzer = modules['color_analyzer']
shape_analyzer = modules['shape_analyzer']
facer_segmenter = modules['facer_segmenter']
eyebrow_stats = modules['eyebrow_stats']

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

def create_method_comparison_grid(left_robust_results, right_robust_results, face_crop):
    """
    Create a comprehensive debugging grid similar to the debugging script
    """
    if not left_robust_results or not right_robust_results:
        return None
    
    left_methods = left_robust_results['methods_results']
    right_methods = right_robust_results['methods_results']
    left_color_results = left_robust_results['color_results']
    right_color_results = right_robust_results['color_results']
    
    # Get all method names
    all_methods = set(left_methods.keys()) | set(right_methods.keys())
    method_list = sorted(list(all_methods))
    
    # Create the comparison data
    comparison_data = []
    
    for method_name in method_list:
        left_method = left_methods.get(method_name, {})
        right_method = right_methods.get(method_name, {})
        left_color = left_color_results.get(method_name, {})
        right_color = right_color_results.get(method_name, {})
        
        method_info = {
            'method_name': method_name,
            'display_name': left_method.get('name', method_name.replace('_', ' ').title()),
            'left_success': left_method.get('success', False),
            'right_success': right_method.get('success', False),
            'left_pixels': left_method.get('pixel_count', 0),
            'right_pixels': right_method.get('pixel_count', 0),
            'left_quality': left_method.get('quality_score', 0),
            'right_quality': right_method.get('quality_score', 0),
            'description': left_method.get('description', 'N/A'),
            'left_color_status': left_color.get('status', 'Failed'),
            'right_color_status': right_color.get('status', 'Failed'),
            'left_mask': left_method.get('mask'),
            'right_mask': right_method.get('mask'),
            'left_colors': left_color.get('colors'),
            'right_colors': right_color.get('colors'),
            'left_percentages': left_color.get('percentages'),
            'right_percentages': right_color.get('percentages'),
            'left_palette': left_color.get('palette'),
            'right_palette': right_color.get('palette')
        }
        comparison_data.append(method_info)
    
    return comparison_data

def display_robust_analysis_results(robust_results, side_name):
    """
    Display comprehensive results from robust analysis
    """
    if not robust_results:
        st.error(f"No robust analysis results available for {side_name}")
        return
    
    st.subheader(f"🔬 {side_name} Eyebrow - Robust Analysis Results")
    
    # Display summary
    summary = robust_results.get('summary', {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Methods", summary.get('total_methods', 0))
    with col2:
        st.metric("Successful Methods", summary.get('successful_methods', 0))
    with col3:
        st.metric("Color Extractions", summary.get('successful_color_extractions', 0))
    
    # Display primary results (best method)
    if robust_results.get('primary_colors') is not None:
        st.subheader(f"🏆 Primary Results (Best Method: {robust_results.get('best_method', 'Unknown')})")
        
        # Show color palette
        debug_images = robust_results.get('debug_images', {})
        palette = debug_images.get('color_palette')
        if palette is not None:
            st.image(palette, caption="Extracted Color Palette", width=400)
        
        # Show color information
        primary_colors = robust_results['primary_colors']
        primary_percentages = robust_results['primary_percentages']
        
        color_info = color_analyzer.get_color_info(primary_colors, primary_percentages)
        
        # Display color table
        for i, info in enumerate(color_info):
            with st.expander(f"Color {i+1}: {info['hex']} ({info['percentage']})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**RGB:** {info['rgb']}")
                    st.write(f"**HEX:** {info['hex']}")
                    st.write(f"**LAB:** L:{info['lab'][0]} a:{info['lab'][1]} b:{info['lab'][2]}")
                with col2:
                    st.write(f"**LCH:** L:{info['lch'][0]} C:{info['lch'][1]} H:{info['lch'][2]}")
                    st.write(f"**HSV:** H:{info['hsv'][0]} S:{info['hsv'][1]} V:{info['hsv'][2]}")
                    st.write(f"**Percentage:** {info['percentage']}")

def display_method_selector_and_results(left_robust_results, right_robust_results):
    """
    Display method selector and results for each method - Shows ALL methods with their masks (even failed ones)
    """
    if not left_robust_results or not right_robust_results:
        st.error("Robust analysis results not available")
        return
    
    # Get all available methods (including failed ones)
    left_methods = left_robust_results.get('methods_results', {})
    right_methods = right_robust_results.get('methods_results', {})
    all_methods = set(left_methods.keys()) | set(right_methods.keys())
    
    # 🆕 Expected methods list to ensure we show all 11
    expected_methods = [
        'hsv_method',
        'lab_method', 
        'edge_method',
        'gabor_method',
        'texture_method',
        'tophat_method',
        'outlier_method',
        'percentile_method',
        'erosion_method',
        'minimal_method',
        'intelligent_combination'
    ]
    
    # Ensure all expected methods are included (add missing ones as failed)
    for method in expected_methods:
        if method not in all_methods:
            # Add missing method as failed
            left_methods[method] = {'success': False, 'reason': 'Method not executed', 'name': method.replace('_', ' ').title()}
            right_methods[method] = {'success': False, 'reason': 'Method not executed', 'name': method.replace('_', ' ').title()}
            all_methods.add(method)
    
    # Create method display names with success indicators
    method_display_names = {}
    method_success_counts = {}
    
    for method_name in sorted(all_methods):  # Sort for consistent order
        left_method = left_methods.get(method_name, {})
        right_method = right_methods.get(method_name, {})
        
        # Get display name
        display_name = left_method.get('name') or right_method.get('name') or method_name.replace('_', ' ').title()
        
        # Count successes
        left_success = left_method.get('success', False)
        right_success = right_method.get('success', False)
        success_count = sum([left_success, right_success])
        
        # Add success indicator to display name
        if success_count == 2:
            display_name_with_status = f"✅✅ {display_name} (Both Successful)"
        elif success_count == 1:
            side = "Left" if left_success else "Right"
            display_name_with_status = f"⚠️ {display_name} ({side} Only)"
        else:
            display_name_with_status = f"❌ {display_name} (Both Failed)"
        
        method_display_names[display_name_with_status] = method_name
        method_success_counts[method_name] = success_count
    
    # Method selector with enhanced info
    st.subheader("🔧 Method Selection - All 11 Methods")
    
    # Show summary statistics
    total_methods = len(all_methods)
    successful_both = sum(1 for count in method_success_counts.values() if count == 2)
    successful_one = sum(1 for count in method_success_counts.values() if count == 1)
    failed_both = sum(1 for count in method_success_counts.values() if count == 0)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Methods", total_methods)
    with col2:
        st.metric("Both Successful", successful_both, delta=f"{successful_both/total_methods*100:.0f}%")
    with col3:
        st.metric("Partial Success", successful_one, delta=f"{successful_one/total_methods*100:.0f}%")
    with col4:
        st.metric("Both Failed", failed_both, delta=f"{failed_both/total_methods*100:.0f}%")
    
    st.info(f"📊 **All {total_methods} detection methods shown** - Select any method to view its results, masks, and failure reasons")
    
    selected_display_name = st.selectbox(
        "Choose a detection method to analyze:",
        options=list(method_display_names.keys()),
        help="All 11 methods shown with success indicators. Failed methods will show empty/black masks and diagnostic info.",
        key="method_selector"
    )
    
    selected_method = method_display_names[selected_display_name]
    
    # Display results for selected method
    st.subheader(f"📊 Detailed Results for: {selected_method.replace('_', ' ').title()}")
    
    col1, col2 = st.columns(2)
    
    # Left eyebrow results - ALWAYS SHOW MASK (even if failed)
    with col1:
        st.subheader("👈 Left Eyebrow")
        left_method_data = left_methods.get(selected_method, {})
        left_color_data = left_robust_results.get('color_results', {}).get(selected_method, {})
        
        # 🆕 ALWAYS show mask regardless of success/failure
        if 'mask' in left_method_data and left_method_data['mask'] is not None:
            mask = left_method_data['mask']
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            
            # Show mask statistics
            total_pixels = np.sum(mask > 0)
            mask_shape = mask.shape
            percentage_detected = (total_pixels / (mask_shape[0] * mask_shape[1])) * 100
            
            st.image(mask_rgb, caption=f"Detection Mask ({total_pixels} pixels, {percentage_detected:.1f}% of region)", width=400)
            
            # Show mask analysis
            st.write(f"**Mask Analysis:**")
            st.write(f"- Detected pixels: {total_pixels}")
            st.write(f"- Mask dimensions: {mask_shape[0]}x{mask_shape[1]}")
            st.write(f"- Coverage: {percentage_detected:.1f}% of eyebrow region")
            
            if total_pixels == 0:
                st.error("🔍 **MASK IS COMPLETELY BLACK** - This is why the method failed!")
                st.write("**Possible reasons for black mask:**")
                st.write("- Thresholds too strict for this image")
                st.write("- Method not suitable for this hair/skin combination")
                st.write("- Lighting conditions incompatible with method")
            elif total_pixels < 10:
                st.warning(f"🔍 **VERY FEW PIXELS DETECTED** ({total_pixels}) - Likely insufficient for reliable color analysis")
        else:
            st.error("🔍 **NO MASK GENERATED** - Method failed to create any detection mask")
        
        # Show success/failure status
        if left_method_data.get('success', False):
            st.success(f"✅ Success - {left_method_data.get('pixel_count', 0)} pixels detected")
            st.write(f"**Description:** {left_method_data.get('description', 'N/A')}")
            st.write(f"**Quality Score:** {left_method_data.get('quality_score', 0):.1f}/100")
            
            # Show colors if available
            if left_color_data.get('status') == 'Success':
                st.success("🎨 Colors extracted successfully")
                
                if left_color_data.get('palette') is not None:
                    st.image(left_color_data['palette'], caption="Color Palette", width=300)
                
                # Show color details with LAB values
                if left_color_data.get('colors') is not None:
                    colors = left_color_data['colors']
                    percentages = left_color_data['percentages']
                    
                    st.write("**Detected Colors:**")
                    for i, (color, pct) in enumerate(zip(colors, percentages)):
                        # Convert to LAB for display
                        color_bgr = color[::-1]  # RGB to BGR
                        color_lab = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2LAB)[0][0] # type: ignore
                        
                        col_a, col_b = st.columns([1, 3])
                        with col_a:
                            hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
                            st.markdown(f"<div style='background-color: {hex_color}; width: 100%; height: 50px; border: 1px solid #000;'></div>", unsafe_allow_html=True)
                        with col_b:
                            st.write(f"**Color {i+1}:** RGB{tuple(color)} ({pct:.1f}%)")
                            st.write(f"LAB: L{color_lab[0]} a{color_lab[1]} b{color_lab[2]}")
                            st.write(f"HEX: {hex_color}")
            else:
                st.error(f"❌ Color extraction failed: {left_color_data.get('status', 'Unknown error')}")
        else:
            st.error(f"❌ Detection failed")
            failure_reason = left_method_data.get('reason', 'Unknown error')
            st.write(f"**Failure reason:** {failure_reason}")
            
            # 🆕 Enhanced diagnostic information for failed methods
            with st.expander("🔍 **Detailed Failure Analysis**", expanded=True):
                st.markdown(f"""
                **Why did this method fail?**
                
                **Method:** {selected_method.replace('_', ' ').title()}
                **Reason:** {failure_reason}
                
                **Diagnostic Steps:**
                1. **Check the mask above** - Is it completely black?
                2. **If black:** Method's thresholds are too strict for this image
                3. **If few pixels:** Method detected something but not enough
                4. **If no mask:** Method crashed during execution
                
                **Method-Specific Troubleshooting:**
                """)
                
                # Method-specific diagnostics
                if selected_method == 'erosion_method':
                    st.write("- **Erosion Method**: May fail if initial mask is too small")
                    st.write("- **Solution**: Try methods with larger initial detection like 'hsv_method'")
                elif selected_method == 'minimal_method':
                    st.write("- **Minimal Method**: Should rarely fail - usually detects bottom 5% of pixels")
                    st.write("- **If failed**: The eyebrow region might be extremely uniform in color")
                elif selected_method == 'gabor_method':
                    st.write("- **Gabor Method**: Detects hair texture patterns")
                    st.write("- **May fail with**: Very smooth eyebrows or low resolution images")
                elif selected_method == 'outlier_method':
                    st.write("- **Outlier Method**: Needs sufficient pixel variation for statistics")
                    st.write("- **May fail with**: Very uniform eyebrow regions")
                else:
                    st.write(f"- Check if this method is suitable for your image's lighting/hair type")
                
                st.write(f"\n**Alternative methods to try:**")
                st.write(f"- Look for methods marked with ✅✅ (successful on both sides)")
                st.write(f"- 'Intelligent Combination' usually works by combining successful methods")
    
    # Right eyebrow results - ALWAYS SHOW MASK (even if failed)
    with col2:
        st.subheader("👉 Right Eyebrow")
        right_method_data = right_methods.get(selected_method, {})
        right_color_data = right_robust_results.get('color_results', {}).get(selected_method, {})
        
        # 🆕 ALWAYS show mask regardless of success/failure
        if 'mask' in right_method_data and right_method_data['mask'] is not None:
            mask = right_method_data['mask']
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            
            # Show mask statistics
            total_pixels = np.sum(mask > 0)
            mask_shape = mask.shape
            percentage_detected = (total_pixels / (mask_shape[0] * mask_shape[1])) * 100
            
            st.image(mask_rgb, caption=f"Detection Mask ({total_pixels} pixels, {percentage_detected:.1f}% of region)", width=400)
            
            # Show mask analysis
            st.write(f"**Mask Analysis:**")
            st.write(f"- Detected pixels: {total_pixels}")
            st.write(f"- Mask dimensions: {mask_shape[0]}x{mask_shape[1]}")
            st.write(f"- Coverage: {percentage_detected:.1f}% of eyebrow region")
            
            if total_pixels == 0:
                st.error("🔍 **MASK IS COMPLETELY BLACK** - This is why the method failed!")
                st.write("**Possible reasons for black mask:**")
                st.write("- Thresholds too strict for this image")
                st.write("- Method not suitable for this hair/skin combination")
                st.write("- Lighting conditions incompatible with method")
            elif total_pixels < 10:
                st.warning(f"🔍 **VERY FEW PIXELS DETECTED** ({total_pixels}) - Likely insufficient for reliable color analysis")
        else:
            st.error("🔍 **NO MASK GENERATED** - Method failed to create any detection mask")
        
        # Show success/failure status
        if right_method_data.get('success', False):
            st.success(f"✅ Success - {right_method_data.get('pixel_count', 0)} pixels detected")
            st.write(f"**Description:** {right_method_data.get('description', 'N/A')}")
            st.write(f"**Quality Score:** {right_method_data.get('quality_score', 0):.1f}/100")
            
            # Show colors if available
            if right_color_data.get('status') == 'Success':
                st.success("🎨 Colors extracted successfully")
                
                if right_color_data.get('palette') is not None:
                    st.image(right_color_data['palette'], caption="Color Palette", width=300)
                
                # Show color details with LAB values
                if right_color_data.get('colors') is not None:
                    colors = right_color_data['colors']
                    percentages = right_color_data['percentages']
                    
                    st.write("**Detected Colors:**")
                    for i, (color, pct) in enumerate(zip(colors, percentages)):
                        # Convert to LAB for display
                        color_bgr = color[::-1]  # RGB to BGR
                        color_lab = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2LAB)[0][0] # type: ignore
                        
                        col_a, col_b = st.columns([1, 3])
                        with col_a:
                            hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
                            st.markdown(f"<div style='background-color: {hex_color}; width: 100%; height: 50px; border: 1px solid #000;'></div>", unsafe_allow_html=True)
                        with col_b:
                            st.write(f"**Color {i+1}:** RGB{tuple(color)} ({pct:.1f}%)")
                            st.write(f"LAB: L{color_lab[0]} a{color_lab[1]} b{color_lab[2]}")
                            st.write(f"HEX: {hex_color}")
            else:
                st.error(f"❌ Color extraction failed: {right_color_data.get('status', 'Unknown error')}")
        else:
            st.error(f"❌ Detection failed")
            failure_reason = right_method_data.get('reason', 'Unknown error')
            st.write(f"**Failure reason:** {failure_reason}")
            
            # Enhanced diagnostic information for failed methods
            with st.expander("🔍 **Detailed Failure Analysis**", expanded=True):
                st.markdown(f"""
                **Why did this method fail?**
                
                **Method:** {selected_method.replace('_', ' ').title()}
                **Reason:** {failure_reason}
                
                **Check the mask above to see what happened!**
                """)
    
    # 🆕 Method comparison summary
    st.markdown("---")
    st.subheader("📈 Quick Method Performance Overview")
    
    # Create a simple performance table
    performance_data = []
    for method_name in sorted(all_methods):
        left_method = left_methods.get(method_name, {})
        right_method = right_methods.get(method_name, {})
        
        display_name = left_method.get('name') or right_method.get('name') or method_name.replace('_', ' ').title()
        left_success = "✅" if left_method.get('success', False) else "❌"
        right_success = "✅" if right_method.get('success', False) else "❌"
        left_pixels = left_method.get('pixel_count', 0)
        right_pixels = right_method.get('pixel_count', 0)
        left_quality = left_method.get('quality_score', 0)
        right_quality = right_method.get('quality_score', 0)
        
        performance_data.append({
            'Method': display_name,
            'Left': left_success,
            'Right': right_success,
            'Left Pixels': left_pixels,
            'Right Pixels': right_pixels,
            'Left Quality': f"{left_quality:.1f}",
            'Right Quality': f"{right_quality:.1f}"
        })
    
    import pandas as pd
    df = pd.DataFrame(performance_data)
    st.dataframe(df, use_container_width=True)


def display_debugging_grid(left_robust_results, right_robust_results, face_crop):
    """
    Display debugging grid with original eyebrow visualization and zoom capabilities
    """
    st.subheader("🔬 Method Comparison Grid")
    
    comparison_data = create_method_comparison_grid(left_robust_results, right_robust_results, face_crop)
    
    if not comparison_data:
        st.error("No comparison data available")
        return
    
    # Show face crop
    st.subheader("Original Face Crop")
    if face_crop is not None:
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        st.image(face_rgb, caption="Face Crop", width=400)
    
    # Show analysis summary
    left_successful = sum(1 for method in comparison_data if method['left_success'])
    right_successful = sum(1 for method in comparison_data if method['right_success'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Methods", len(comparison_data))
    with col2:
        st.metric("Left Successful", left_successful)
    with col3:
        st.metric("Right Successful", right_successful)
    
    # 🆕 Zoom Controls
    st.subheader("🔍 Magnification Controls")
    zoom_col1, zoom_col2 = st.columns(2)
    with zoom_col1:
        mask_zoom = st.slider("🔍 Mask Image Size", min_value=300, max_value=800, value=600, step=50, 
                             help="Adjust the size of mask images for detailed inspection")
    with zoom_col2:
        show_overlay = st.checkbox("📊 Show Overlay Comparison", value=True, 
                                  help="Show original eyebrow with mask overlay for better comparison")
    
    # Get original eyebrow regions from the results
    left_initial_mask = left_robust_results.get('debug_images', {}).get('detected_hair_overlay')
    right_initial_mask = right_robust_results.get('debug_images', {}).get('detected_hair_overlay')
    
    # If not available from debug images, try to get from session state or recreate
    if 'cached_results' in st.session_state:
        cached_results = st.session_state.cached_results.get('results', {})
        left_eyebrow_mask = cached_results.get('left_refined_mask')
        right_eyebrow_mask = cached_results.get('right_refined_mask')
    else:
        left_eyebrow_mask = None
        right_eyebrow_mask = None
    
    # 🆕 Quality Score Explanation
    with st.expander("📊 Understanding Quality Scores", expanded=False):
        st.markdown("""
        **Quality Score Calculation (0-100 points):**
        
        The quality score evaluates how well each detection method performs based on three factors:
        
        🎯 **Pixel Count Score (40% weight):**
        - **100 points**: Detects 5-40% of the total eyebrow region (optimal range)
        - **Lower points**: Too few pixels (<5%) or too many pixels (>40%)
        - **Rationale**: Good methods should detect substantial hair without over-detecting
        
        🌑 **Darkness Score (40% weight):**
        - **100 points**: Detected pixels are significantly darker than the average eyebrow region
        - **Lower points**: Detected pixels are similar to or lighter than the average
        - **Rationale**: Hair pixels should be darker than skin pixels
        
        🔗 **Spatial Coherence Score (20% weight):**
        - **100 points**: 1-3 connected components (coherent detection)
        - **Lower points**: Many scattered components (fragmented detection)
        - **Rationale**: Hair should form coherent regions, not scattered dots
        
        **Interpretation:**
        - **80-100**: Excellent detection quality
        - **60-79**: Good detection with minor issues  
        - **40-59**: Moderate detection, may have artifacts
        - **20-39**: Poor detection with significant issues
        - **0-19**: Very poor or failed detection
        """)
    
    # Method-by-method analysis with enhanced visualization
    st.subheader("📊 Method-by-Method Analysis")
    
    for method_info in comparison_data:
        with st.expander(f"{method_info['display_name']} - {method_info['description']}", expanded=False):
            
            # Method information
            st.write(f"**Method:** {method_info['display_name']}")
            st.write(f"**Description:** {method_info['description']}")
            
            # Results summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if method_info['left_success']:
                    st.success(f"Left: ✅ {method_info['left_pixels']} pixels")
                else:
                    st.error("Left: ❌ Failed")
            
            with col2:
                if method_info['right_success']:
                    st.success(f"Right: ✅ {method_info['right_pixels']} pixels")
                else:
                    st.error("Right: ❌ Failed")
            
            with col3:
                # Color-coded quality score
                left_quality = method_info['left_quality']
                if left_quality >= 80:
                    st.success(f"Left Quality: {left_quality:.1f}")
                elif left_quality >= 60:
                    st.warning(f"Left Quality: {left_quality:.1f}")
                else:
                    st.error(f"Left Quality: {left_quality:.1f}")
            
            with col4:
                # Color-coded quality score
                right_quality = method_info['right_quality']
                if right_quality >= 80:
                    st.success(f"Right Quality: {right_quality:.1f}")
                elif right_quality >= 60:
                    st.warning(f"Right Quality: {right_quality:.1f}")
                else:
                    st.error(f"Right Quality: {right_quality:.1f}")
            
            # 🔍 ENHANCED Visual results with original eyebrow comparison
            st.markdown("---")
            st.markdown("### 🔍 Detailed Mask Analysis (Enlarged View with Original Comparison)")
            
            visual_col1, visual_col2 = st.columns(2)
            
            # Left side: Enhanced analysis
            with visual_col1:
                st.markdown("#### 👈 Left Eyebrow Analysis")
                
                # Create original eyebrow region for comparison
                if face_crop is not None and left_eyebrow_mask is not None:
                    # Extract original eyebrow region
                    original_left_region = extract_eyebrow_region(face_crop, left_eyebrow_mask)
                    if original_left_region is not None:
                        original_left_rgb = cv2.cvtColor(original_left_region, cv2.COLOR_BGR2RGB)
                        
                        # Show original eyebrow region
                        st.write("**📷 Original Eyebrow Region:**")
                        st.image(original_left_rgb, caption="Original Left Eyebrow", width=mask_zoom//2)
                
                # Show detection mask
                if method_info['left_mask'] is not None and method_info['left_success']:
                    mask_rgb = cv2.cvtColor(method_info['left_mask'], cv2.COLOR_GRAY2RGB)
                    
                    # Extract detected region for better visualization
                    detected_region = extract_eyebrow_region_with_mask(face_crop, method_info['left_mask'])
                    
                    st.write(f"**🎯 Detection Mask ({method_info['display_name']}):**")
                    st.image(mask_rgb, caption=f"Left Detection Mask", width=mask_zoom)
                    
                    # Show overlay if requested
                    if show_overlay and face_crop is not None:
                        overlay_img = create_mask_overlay(face_crop, method_info['left_mask'], left_eyebrow_mask)
                        if overlay_img is not None:
                            st.write("**🔍 Overlay: Original + Detection:**")
                            overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
                            st.image(overlay_rgb, caption="Original (blue) + Detection (green)", width=mask_zoom)
                    
                    # Show statistics
                    mask_stats = analyze_mask_statistics(method_info['left_mask'], left_eyebrow_mask if left_eyebrow_mask is not None else method_info['left_mask'])
                    st.write("**📊 Detection Statistics:**")
                    for key, value in mask_stats.items():
                        st.write(f"- {key}: {value}")
                        
                else:
                    st.error("❌ No detection mask available")
                    if face_crop is not None and left_eyebrow_mask is not None:
                        # Still show original region for reference
                        original_left_region = extract_eyebrow_region(face_crop, left_eyebrow_mask)
                        if original_left_region is not None:
                            original_left_rgb = cv2.cvtColor(original_left_region, cv2.COLOR_BGR2RGB)
                            st.write("**📷 Original Eyebrow Region (for reference):**")
                            st.image(original_left_rgb, caption="Original Left Eyebrow", width=mask_zoom//2)
                
                # Left colors
                st.markdown("#### 🎨 Left Colors")
                if method_info['left_color_status'] == 'Success' and method_info['left_palette'] is not None:
                    st.image(method_info['left_palette'], caption="Left Colors", width=400)
                    
                    # Show LAB values
                    if method_info['left_colors'] is not None:
                        for i, (color, pct) in enumerate(zip(method_info['left_colors'], method_info['left_percentages'])):
                            color_bgr = color[::-1]
                            color_lab = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2LAB)[0][0] # type: ignore
                            st.text(f"C{i+1}({pct:.0f}%): L{color_lab[0]} a{color_lab[1]} b{color_lab[2]}")
                else:
                    st.write(f"Colors: {method_info['left_color_status']}")
            
            # Right side: Enhanced analysis
            with visual_col2:
                st.markdown("#### 👉 Right Eyebrow Analysis")
                
                # Create original eyebrow region for comparison
                if face_crop is not None and right_eyebrow_mask is not None:
                    # Extract original eyebrow region
                    original_right_region = extract_eyebrow_region(face_crop, right_eyebrow_mask)
                    if original_right_region is not None:
                        original_right_rgb = cv2.cvtColor(original_right_region, cv2.COLOR_BGR2RGB)
                        
                        # Show original eyebrow region
                        st.write("**📷 Original Eyebrow Region:**")
                        st.image(original_right_rgb, caption="Original Right Eyebrow", width=mask_zoom//2)
                
                # Show detection mask
                if method_info['right_mask'] is not None and method_info['right_success']:
                    mask_rgb = cv2.cvtColor(method_info['right_mask'], cv2.COLOR_GRAY2RGB)
                    
                    st.write(f"**🎯 Detection Mask ({method_info['display_name']}):**")
                    st.image(mask_rgb, caption=f"Right Detection Mask", width=mask_zoom)
                    
                    # Show overlay if requested
                    if show_overlay and face_crop is not None:
                        overlay_img = create_mask_overlay(face_crop, method_info['right_mask'], right_eyebrow_mask)
                        if overlay_img is not None:
                            st.write("**🔍 Overlay: Original + Detection:**")
                            overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
                            st.image(overlay_rgb, caption="Original (blue) + Detection (green)", width=mask_zoom)
                    
                    # Show statistics
                    mask_stats = analyze_mask_statistics(method_info['right_mask'], right_eyebrow_mask if right_eyebrow_mask is not None else method_info['right_mask'])
                    st.write("**📊 Detection Statistics:**")
                    for key, value in mask_stats.items():
                        st.write(f"- {key}: {value}")
                        
                else:
                    st.error("❌ No detection mask available")
                    if face_crop is not None and right_eyebrow_mask is not None:
                        # Still show original region for reference
                        original_right_region = extract_eyebrow_region(face_crop, right_eyebrow_mask)
                        if original_right_region is not None:
                            original_right_rgb = cv2.cvtColor(original_right_region, cv2.COLOR_BGR2RGB)
                            st.write("**📷 Original Eyebrow Region (for reference):**")
                            st.image(original_right_rgb, caption="Original Right Eyebrow", width=mask_zoom//2)
                
                # Right colors
                st.markdown("#### 🎨 Right Colors")
                if method_info['right_color_status'] == 'Success' and method_info['right_palette'] is not None:
                    st.image(method_info['right_palette'], caption="Right Colors", width=400)
                    
                    # Show LAB values
                    if method_info['right_colors'] is not None:
                        for i, (color, pct) in enumerate(zip(method_info['right_colors'], method_info['right_percentages'])):
                            color_bgr = color[::-1]
                            color_lab = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2LAB)[0][0] # type: ignore
                            st.text(f"C{i+1}({pct:.0f}%): L{color_lab[0]} a{color_lab[1]} b{color_lab[2]}")
                else:
                    st.write(f"Colors: {method_info['right_color_status']}")

# 🆕 Helper functions for enhanced visualization
def extract_eyebrow_region(face_crop, eyebrow_mask, padding=10):
    """Extract the eyebrow region from face crop using the mask"""
    if face_crop is None or eyebrow_mask is None:
        return None
    
    try:
        # Find bounding box of the eyebrow region
        y_indices, x_indices = np.where(eyebrow_mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return None
        
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Add padding
        h, w = face_crop.shape[:2]
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        
        # Extract region
        eyebrow_region = face_crop[y_min:y_max, x_min:x_max]
        return eyebrow_region
        
    except Exception as e:
        print(f"Error extracting eyebrow region: {e}")
        return None

def extract_eyebrow_region_with_mask(face_crop, detection_mask, padding=10):
    """Extract detected eyebrow pixels highlighted on original region"""
    if face_crop is None or detection_mask is None:
        return None
    
    try:
        # Create a copy of face crop
        result = face_crop.copy()
        
        # Highlight detected pixels in green
        result[detection_mask > 0] = [0, 255, 0]  # Green for detected hair
        
        # Find bounding box and extract region
        y_indices, x_indices = np.where(detection_mask > 0)
        if len(y_indices) == 0:
            return None
        
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Add padding
        h, w = face_crop.shape[:2]
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        
        # Extract region
        detected_region = result[y_min:y_max, x_min:x_max]
        return detected_region
        
    except Exception as e:
        print(f"Error extracting detected region: {e}")
        return None

def create_mask_overlay(face_crop, detection_mask, original_mask):
    """Create overlay showing original eyebrow (blue) and detection (green)"""
    if face_crop is None or detection_mask is None:
        return None
    
    try:
        # Create overlay image
        overlay = face_crop.copy()
        
        # Show original eyebrow region in blue
        if original_mask is not None:
            overlay[original_mask > 0] = overlay[original_mask > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
        
        # Show detected pixels in green
        overlay[detection_mask > 0] = overlay[detection_mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
        
        return overlay
        
    except Exception as e:
        print(f"Error creating overlay: {e}")
        return None

def analyze_mask_statistics(detection_mask, reference_mask):
    """Analyze mask statistics for detailed comparison"""
    if detection_mask is None:
        return {"Error": "No detection mask"}
    
    stats = {}
    
    # Basic statistics
    detected_pixels = np.sum(detection_mask > 0)
    total_pixels = detection_mask.size
    
    stats["Detected pixels"] = f"{detected_pixels:,}"
    stats["Total mask pixels"] = f"{total_pixels:,}"
    stats["Detection percentage"] = f"{(detected_pixels/total_pixels)*100:.2f}%"
    
    # Compare with reference if available
    if reference_mask is not None:
        reference_pixels = np.sum(reference_mask > 0)
        if reference_pixels > 0:
            coverage = (detected_pixels / reference_pixels) * 100
            stats["Coverage of eyebrow region"] = f"{coverage:.1f}%"
        
        # Overlap analysis
        if detected_pixels > 0:
            overlap = np.sum((detection_mask > 0) & (reference_mask > 0))
            overlap_percentage = (overlap / detected_pixels) * 100
            stats["Overlap with eyebrow region"] = f"{overlap_percentage:.1f}%"
    
    # Connected components
    num_components, _ = cv2.connectedComponents(detection_mask)
    stats["Connected regions"] = f"{num_components - 1}"  # Subtract background
    
    return stats

# Modified process_image function with proper caching that considers n_colors
def process_image_cached(image, filename=None, n_colors=3):
    """
    Process image with proper session state caching to avoid reprocessing
    Also triggers reprocessing when n_colors changes
    """
    # Create a cache key based on image and parameters
    image_hash = hash(image.tobytes())
    cache_key = f"{image_hash}_{filename}_{n_colors}"
    
    # Check if we already have results for this image and parameters
    if 'cached_results' in st.session_state and st.session_state.cached_results.get('cache_key') == cache_key:
        print("🚀 Using cached results - no reprocessing needed!")
        return st.session_state.cached_results['results']
    
    print(f"🔄 Processing image (n_colors={n_colors})...")
    
    # Detect face landmarks
    face_detector_local = FaceDetector(use_gpu=True)
    results = face_detector_local.detect_face(image)
    
    if not results.multi_face_landmarks:
        st.error("No face detected in the image. Please upload another image.")
        return None
    
    # Crop face
    face_crop, crop_coords = face_detector_local.crop_face(image, results)
    
    if face_crop is None or crop_coords is None:
        st.error("Could not crop face properly. Please upload another image.")
        return None
        
    # Get face dimensions
    x_min, y_min, x_max, y_max = crop_coords
    
    # Run landmark detection on the cropped face for better accuracy
    face_crop_results = face_detector_local.detect_face(face_crop)
    
    if not face_crop_results.multi_face_landmarks:
        st.error("Could not detect facial landmarks on the cropped face. Please upload another image.")
        return None
    
    # Get eyebrow landmarks (still needed for shape analysis)
    left_eyebrow, right_eyebrow = face_detector_local.get_eyebrow_landmarks(image, results, crop_coords)
    
    if left_eyebrow is None or right_eyebrow is None:
        left_eyebrow, right_eyebrow = face_detector_local.get_eyebrow_landmarks(image, results)
        
        if left_eyebrow is None or right_eyebrow is None:
            st.error("Could not detect eyebrows. Please upload another image.")
            return None
    
    # *** PRIMARY METHOD: Use Facer segmentation for masks ***
    try:
        facer_segmenter_local = FacerSegmentation(use_gpu=True)
        facer_result = facer_segmenter_local.segment_eyebrows(face_crop, visualize=True)
        
        if facer_result.get('success', False):
            # Use Facer masks as primary masks
            left_refined_mask = facer_result.get('left_eyebrow_mask')
            right_refined_mask = facer_result.get('right_eyebrow_mask')
            using_facer_masks = True
            st.session_state['facer_available'] = True
        else:
            # Fallback to traditional method only if Facer fails completely
            eyebrow_segmentation_local = EyebrowSegmentation()
            left_mask, left_bbox = eyebrow_segmentation_local.create_eyebrow_mask(face_crop, left_eyebrow)
            right_mask, right_bbox = eyebrow_segmentation_local.create_eyebrow_mask(face_crop, right_eyebrow)
            left_refined_mask = eyebrow_segmentation_local.refine_eyebrow_mask(face_crop, left_mask)
            right_refined_mask = eyebrow_segmentation_local.refine_eyebrow_mask(face_crop, right_mask)
            using_facer_masks = False
            st.session_state['facer_available'] = False
            st.warning("Facer segmentation failed, using traditional method as fallback")
            
    except Exception as e:
        st.warning(f"Facer segmentation failed: {e}. Using traditional method as fallback.")
        # Fallback to traditional method
        eyebrow_segmentation_local = EyebrowSegmentation()
        left_mask, left_bbox = eyebrow_segmentation_local.create_eyebrow_mask(face_crop, left_eyebrow)
        right_mask, right_bbox = eyebrow_segmentation_local.create_eyebrow_mask(face_crop, right_eyebrow)
        left_refined_mask = eyebrow_segmentation_local.refine_eyebrow_mask(face_crop, left_mask)
        right_refined_mask = eyebrow_segmentation_local.refine_eyebrow_mask(face_crop, right_mask)
        using_facer_masks = False
        st.session_state['facer_available'] = False
    
    # *** 🆕 ROBUST COLOR ANALYSIS with all methods ***
    print("🚀 Running robust color analysis...")
    
    # Run robust analysis on both eyebrows
    left_robust_results = color_analyzer.extract_robust_eyebrow_colors(
        face_crop, left_refined_mask, n_colors, filename=filename)
    
    right_robust_results = color_analyzer.extract_robust_eyebrow_colors(
        face_crop, right_refined_mask, n_colors, filename=filename)
    
    # Keep legacy color analysis for backward compatibility
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
    shape_analyzer_local = ShapeAnalysis()
    left_shape_info = shape_analyzer_local.analyze_eyebrow_shape(left_eyebrow)
    right_shape_info = shape_analyzer_local.analyze_eyebrow_shape(right_eyebrow)
    left_shape_vis = shape_analyzer_local.visualize_shape(face_crop, left_eyebrow, left_shape_info)
    right_shape_vis = shape_analyzer_local.visualize_shape(face_crop, right_eyebrow, right_shape_info)
    left_shape_desc = shape_analyzer_local.get_shape_description(left_shape_info)
    right_shape_desc = shape_analyzer_local.get_shape_description(right_shape_info)
    
    # Draw landmarks on original image for visualization
    landmarks_image = face_detector_local.draw_landmarks(image, results)
    cropped_landmarks_image = face_detector_local.draw_landmarks(face_crop, face_crop_results)
    eyebrow_landmarks_image = face_detector_local.draw_eyebrow_landmarks(face_crop, left_eyebrow, right_eyebrow)
    
    # Return all processed data including robust results
    processed_results = {
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
        
        # Legacy color analysis (for backward compatibility)
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
        
        # 🆕 Robust analysis results
        'left_robust_results': left_robust_results,
        'right_robust_results': right_robust_results,
        
        # Shape analysis
        'left_shape_info': left_shape_info,
        'right_shape_info': right_shape_info,
        'left_shape_vis': left_shape_vis,
        'right_shape_vis': right_shape_vis,
        'left_shape_desc': left_shape_desc,
        'right_shape_desc': right_shape_desc
    }
    
    # Cache the results
    st.session_state.cached_results = {
        'cache_key': cache_key,
        'results': processed_results
    }
    
    print("✅ Results cached successfully!")
    
    return processed_results

# Sidebar settings
st.sidebar.title("🔧 Analysis Settings")
n_colors = st.sidebar.slider("Number of colors to extract per method", min_value=2, max_value=5, value=3, 
                            help="Number of dominant colors to extract for each detection method. Changing this will trigger reprocessing.")

show_debug = st.sidebar.checkbox("Show detailed debugging information", value=True, 
                                help="Display detailed debugging and intermediate results")

st.sidebar.markdown("---")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")

# Reset session state when a new image is uploaded
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

# Check if the uploaded file has changed
if uploaded_file is not None and (st.session_state.last_uploaded_file is None or 
                                 uploaded_file.name != st.session_state.last_uploaded_file):
    # Clear cache when new image is uploaded
    if 'cached_results' in st.session_state:
        del st.session_state.cached_results
    
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
    
    # 🆕 Process the image with caching (considers n_colors changes)
    with st.spinner('Processing image with robust analysis...'):
        results = process_image_cached(image, filename=uploaded_file.name, n_colors=n_colors)
    
    if results:
        # Create tabs for different analyses - 🆕 RESTORED Statistics tab
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Overview", 
            "Shape Analysis", 
            "🆕 Robust Color Analysis", 
            "Legacy Color Analysis",
            "Statistics",  # 🆕 RESTORED
            "🔬 Advanced Debugging"
        ])
        
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
            # Shape Analysis tab (unchanged)
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
            # 🆕 NEW: Robust Color Analysis Tab
            st.header("🔬 Robust Color Analysis with Multiple Methods")
            
            st.markdown("""
            **🚀 Enhanced Analysis Features:**
            - 🎯 **11 Detection Methods**: HSV, LAB, Edge Detection, Gabor Filters, Texture Analysis, Top-hat, Statistical Outliers, Percentile Thresholding, Erosion-based, Minimal Detection, and Intelligent Combination
            - 🧠 **Intelligent Fallbacks**: If one method fails, automatically tries alternatives
            - 📊 **Quality Scoring**: Each method gets a quality score based on pixel count, darkness, and coherence
            - 🏆 **Best Method Selection**: Automatically selects the best performing method
            - 🔍 **Method Comparison**: Compare results across all detection methods
            - 📈 **Smart Caching**: Method selection changes display only, no reprocessing (unless n_colors changes)
            """)
            
            # Check if we have robust results
            left_robust_results = results.get('left_robust_results')
            right_robust_results = results.get('right_robust_results')
            
            if left_robust_results and right_robust_results:
                
                # Display primary results for both eyebrows
                col1, col2 = st.columns(2)
                with col1:
                    display_robust_analysis_results(left_robust_results, "Left")
                with col2:
                    display_robust_analysis_results(right_robust_results, "Right")
                
                st.markdown("---")
                
                # Method selector and individual results
                display_method_selector_and_results(left_robust_results, right_robust_results)
                
            else:
                st.error("❌ Robust analysis results not available")

        with tab4:
            # Legacy Color Analysis tab (keep existing functionality)
            st.header("Legacy Color Analysis (Single Method)")
            
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
                
                # Enhanced Color Analysis Section with metadata awareness
                st.subheader("🎨 Legacy Hair-Only Color Analysis")
                
                color_col1, color_col2 = st.columns(2)
                
                # Left Eyebrow Enhanced Analysis
                with color_col1:
                    st.subheader("👈 Left Eyebrow Analysis")
                    
                    # Get the analysis results
                    left_colors = results.get('left_colors')
                    left_percentages = results.get('left_percentages') 
                    
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
                    else:
                        st.warning("⚠️ Could not extract hair colors from left eyebrow")
                
                # Right Eyebrow Enhanced Analysis  
                with color_col2:
                    st.subheader("👉 Right Eyebrow Analysis")
                    
                    # Get the analysis results
                    right_colors = results.get('right_colors')
                    right_percentages = results.get('right_percentages')
                    
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
                    else:
                        st.warning("⚠️ Could not extract hair colors from right eyebrow")
            else:
                st.error("❌ Facer segmentation failed. Cannot perform color analysis.")

        # 🆕 RESTORED Statistics Tab
        with tab5:
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

        with tab6:
            # 🆕 Advanced Debugging Tab (REMOVED legacy debugging section)
            st.header("🔬 Advanced Debugging & Method Comparison")
            
            st.markdown("""
            **This advanced debugging section provides:**
            - 📊 **Method Comparison Grid**: See all 11 methods side-by-side like in the debugging script
            - 🎯 **Success/Failure Analysis**: Understand why methods succeed or fail
            - 🔍 **Enlarged Mask View**: "Hand lens" detailed view of detection masks
            - 📈 **Quality Scores**: See which methods perform best for this image (with explanation)
            - 🧠 **LAB Color Values**: Detailed color information for each method
            """)
            
            # Get robust results
            left_robust_results = results.get('left_robust_results')
            right_robust_results = results.get('right_robust_results')
            face_crop = results.get('face_crop')
            
            if left_robust_results and right_robust_results:
                # Display debugging grid with enlarged masks and quality explanation
                display_debugging_grid(left_robust_results, right_robust_results, face_crop)
                
                # 🗑️ REMOVED: Legacy Single-Method Debugging section
                # (This was the section you wanted removed)
                
            else:
                st.error("❌ Advanced debugging not available - robust analysis failed")


# Add information about the app (updated)
st.sidebar.title("About")
st.sidebar.info("""
**🔬 Advanced Eyebrow Analysis App**

This app analyzes eyebrows using **11 robust detection methods**:

**🎯 Detection Methods:**
1. HSV Color Space Analysis
2. LAB Lightness Detection  
3. Edge Detection (Canny)
4. Gabor Filter Banks
5. Texture Variance Analysis
6. Morphological Top-hat
7. Statistical Outlier Detection
8. Percentile Thresholding
9. Erosion-based Detection
10. Minimal Detection
11. Intelligent Combination

**📊 Features:**
- Smart caching (no reprocessing on method selection)
- Quality scoring with detailed explanation
- Method comparison and debugging
- Enlarged mask visualization
- Metadata-optimized analysis
- Interactive visualizations

Upload a high-resolution facial image to get started.
""")

# Add instructions (updated)
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. **Upload Image**: Choose a high-resolution facial image
2. **Set Parameters**: Adjust number of colors in the sidebar (triggers reprocessing)
3. **View Results**: 
   - **Overview**: Basic face detection
   - **Shape Analysis**: Geometric measurements
   - **🆕 Robust Analysis**: 11 detection methods with method selector
   - **Legacy Analysis**: Single-method approach
   - **Statistics**: Dataset color statistics
   - **🔬 Advanced Debugging**: Method comparison with enlarged masks

4. **Method Selection**: Choose different detection methods to compare (no reprocessing!)
5. **Quality Scores**: Understand method performance with detailed explanations
""")

# Add technical notes (updated)
st.sidebar.title("Technical Notes")
st.sidebar.markdown("""
**🔬 Robust Detection Pipeline:**
- 11 independent detection methods
- Intelligent fallback strategies
- Quality scoring (pixel count, darkness, coherence)
- Automatic best method selection
- Smart caching (reprocesses only when n_colors changes)

**📊 Analysis Features:**
- LAB color space optimization
- Metadata-aware thresholding
- Edge-preserving filtering
- Morphological operations
- Statistical validation

**🔍 Debugging Features:**
- Enlarged mask visualization (600px width)
- Quality score breakdown and explanation
- Color-coded performance indicators
- Comprehensive method comparison
""")

if __name__ == "__main__":
    # This will be executed when the script is run directly
    pass