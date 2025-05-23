# Eyebrow Analysis App

A comprehensive application for analyzing eyebrows in facial images. This app extracts detailed insights about eyebrow color, shape, and features using computer vision techniques.

## Features

- **Face Detection**: Automatically detects faces in uploaded images
- **Eyebrow Segmentation**: Isolates eyebrow regions using facial landmarks
- **Color Analysis**: Extracts dominant colors using K-means clustering
- **Shape Analysis**: Analyzes eyebrow shape characteristics (arch type, thickness, etc.)
- **Alpha Matting**: Visualizes eyebrow details with alpha matting
- **Interactive UI**: User-friendly Streamlit interface with multiple analysis tabs

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- Streamlit

## Installation

1. Clone this repository or download the files
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run main.py
```

2. Upload a high-resolution facial image through the web interface
3. View the analysis results in the different tabs:
   - **Overview**: Basic detection results
   - **Color Analysis**: Dominant colors in the eyebrows
   - **Shape Analysis**: Shape characteristics and metrics
   - **Detailed View**: Masks and alpha matting visualizations

## Project Structure

- `main.py`: Main Streamlit application
- `face_detection.py`: Face and landmark detection module
- `eyebrow_segmentation.py`: Eyebrow segmentation and masking module
- `color_analysis.py`: Color extraction and analysis module
- `shape_analysis.py`: Shape analysis and visualization module
- `requirements.txt`: Required Python packages

## Technical Details

- Face detection and landmark extraction using MediaPipe
- Eyebrow segmentation using landmark-based masks
- Color analysis using K-means clustering
- Shape analysis using contour analysis and geometric calculations
- Alpha matting simulation for detailed visualization

## Example Results

The app provides:
- Visualization of facial landmarks
- Eyebrow region extraction
- Dominant color palettes
- Shape analysis with metrics
- Detailed eyebrow masks and alpha mattes

## Notes

- For best results, use high-resolution images with clear, well-lit faces
- The app works best with frontal face images
- Multiple faces in a single image may affect results
