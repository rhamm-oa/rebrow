"""
GPU-Optimized Batch Robust Color Analysis Script
- Sequential GPU processing to avoid memory conflicts
- Real-time CSV writing with progress tracking
- Resumable processing with checkpoint support
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse
import time
import gc
import torch
from datetime import datetime
from tqdm import tqdm
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
import psutil

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from rebrow_modules.face_detection import FaceDetector
from rebrow_modules.color_analysis import ColorAnalysis
from rebrow_modules.facer_segmentation import FacerSegmentation
from rebrow_modules.eyebrow_segmentation import EyebrowSegmentation

def convert_rgb_to_lab_proper(rgb_color):
    """Convert RGB to proper LAB values with correct scaling"""
    r, g, b = rgb_color
    rgb_color_obj = sRGBColor(r/255.0, g/255.0, b/255.0)
    lab_color_obj = convert_color(rgb_color_obj, LabColor)
    
    return {
        'L': round(lab_color_obj.lab_l, 1), # type: ignore
        'a': round(lab_color_obj.lab_a, 1), # type: ignore
        'b': round(lab_color_obj.lab_b, 1) # type: ignore
    }

class GPUColorAnalyzer:
    """GPU-optimized color analyzer with model reuse"""
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        self.models_initialized = False
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models once"""
        print("🔧 Initializing GPU models...")
        try:
            self.face_detector = FaceDetector(use_gpu=self.use_gpu)
            self.color_analyzer = ColorAnalysis()
            self.facer_segmenter = FacerSegmentation(use_gpu=self.use_gpu)
            self.eyebrow_segmentation = EyebrowSegmentation()
            self.models_initialized = True
            print("✅ GPU models initialized successfully")
            
            if self.use_gpu and torch.cuda.is_available():
                print(f"🚀 GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated")
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            raise
    
    def clear_gpu_cache(self):
        """Clear GPU cache to prevent memory issues"""
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_memory_info(self):
        """Get current memory usage info"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        info = f"CPU: {cpu_percent:.1f}%, RAM: {memory.percent:.1f}%"
        
        if self.use_gpu and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_max_memory = torch.cuda.max_memory_allocated() / 1024**3
            info += f", GPU: {gpu_memory:.1f}GB (max: {gpu_max_memory:.1f}GB)"
        
        return info
    
    def process_single_image(self, image_path, k_clusters=2, resize_max=800):
        """Process a single image with GPU optimization"""
        try:
            # Read and resize image
            image = cv2.imread(image_path)
            if image is None:
                return None, f"Could not read image {image_path}"
            
            if max(image.shape[:2]) > resize_max:
                scale = resize_max / max(image.shape[:2])
                image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            
            # Detect face landmarks
            results = self.face_detector.detect_face(image)
            if not results.multi_face_landmarks:
                return None, "No face detected"
            
            # Crop face
            face_crop, crop_coords = self.face_detector.crop_face(image, results)
            if face_crop is None or crop_coords is None:
                return None, "Could not crop face properly"
            
            # Get eyebrow landmarks
            left_eyebrow, right_eyebrow = self.face_detector.get_eyebrow_landmarks(image, results, crop_coords)
            if left_eyebrow is None or right_eyebrow is None:
                left_eyebrow, right_eyebrow = self.face_detector.get_eyebrow_landmarks(image, results)
                if left_eyebrow is None or right_eyebrow is None:
                    return None, "Could not detect eyebrows"
            
            # Get eyebrow masks using Facer segmentation
            using_facer_masks = False
            try:
                facer_result = self.facer_segmenter.segment_eyebrows(face_crop, visualize=False)
                if facer_result.get('success', False):
                    left_refined_mask = facer_result.get('left_eyebrow_mask')
                    right_refined_mask = facer_result.get('right_eyebrow_mask')
                    using_facer_masks = True
                else:
                    # Fallback to traditional method
                    left_mask, _ = self.eyebrow_segmentation.create_eyebrow_mask(face_crop, left_eyebrow)
                    right_mask, _ = self.eyebrow_segmentation.create_eyebrow_mask(face_crop, right_eyebrow)
                    left_refined_mask = self.eyebrow_segmentation.refine_eyebrow_mask(face_crop, left_mask)
                    right_refined_mask = self.eyebrow_segmentation.refine_eyebrow_mask(face_crop, right_mask)
            except Exception as e:
                # Fallback to traditional method
                left_mask, _ = self.eyebrow_segmentation.create_eyebrow_mask(face_crop, left_eyebrow)
                right_mask, _ = self.eyebrow_segmentation.create_eyebrow_mask(face_crop, right_eyebrow)
                left_refined_mask = self.eyebrow_segmentation.refine_eyebrow_mask(face_crop, left_mask)
                right_refined_mask = self.eyebrow_segmentation.refine_eyebrow_mask(face_crop, right_mask)
            
            # Robust color analysis
            left_robust_results = self.color_analyzer.extract_robust_eyebrow_colors(
                face_crop, left_refined_mask, k_clusters, filename=os.path.basename(image_path))
            
            right_robust_results = self.color_analyzer.extract_robust_eyebrow_colors(
                face_crop, right_refined_mask, k_clusters, filename=os.path.basename(image_path))
            
            if not left_robust_results or not right_robust_results:
                return None, "Robust color analysis failed"
            
            # Extract results
            result = self._extract_color_results(
                os.path.basename(image_path), 
                left_robust_results, 
                right_robust_results, 
                using_facer_masks
            )
            
            return result, None
            
        except Exception as e:
            return None, f"Error processing {image_path}: {str(e)}"
    
    def _extract_color_results(self, filename, left_results, right_results, using_facer_masks):
        """Extract and format color results"""
        left_color_results = left_results.get('color_results', {})
        right_color_results = right_results.get('color_results', {})
        left_methods_results = left_results.get('methods_results', {})
        right_methods_results = right_results.get('methods_results', {})
        
        # Get top 3 methods by quality score
        left_top_methods = sorted(left_methods_results.items(), 
                                key=lambda x: x[1].get('quality_score', 0), reverse=True)[:3]
        right_top_methods = sorted(right_methods_results.items(), 
                                 key=lambda x: x[1].get('quality_score', 0), reverse=True)[:3]
        
        # Build result dictionary
        result = {
            'image_filename': filename,
            'using_facer_masks': using_facer_masks,
            'processed_at': datetime.now().isoformat()
        }
        
        # Add left eyebrow results
        for i, (method, method_result) in enumerate(left_top_methods):
            prefix = f'left_top_method{i+1}'
            result[f'{prefix}'] = method
            result[f'{prefix}_quality_score'] = method_result.get('quality_score')
            
            # Add color data
            dominant_colors = left_color_results.get(method, {})
            colors_rgb = dominant_colors.get('colors', [])
            percentages = dominant_colors.get('percentages', [])
            
            for j, rgb_color in enumerate(colors_rgb):
                lab_values = convert_rgb_to_lab_proper(rgb_color)
                result[f'{prefix}_color{j+1}_L'] = lab_values['L']
                result[f'{prefix}_color{j+1}_a'] = lab_values['a']
                result[f'{prefix}_color{j+1}_b'] = lab_values['b']
                result[f'{prefix}_color{j+1}_percentage'] = float(percentages[j]) if j < len(percentages) else 0.0
        
        # Add right eyebrow results
        for i, (method, method_result) in enumerate(right_top_methods):
            prefix = f'right_top_method{i+1}'
            result[f'{prefix}'] = method
            result[f'{prefix}_quality_score'] = method_result.get('quality_score')
            
            # Add color data
            dominant_colors = right_color_results.get(method, {})
            colors_rgb = dominant_colors.get('colors', [])
            percentages = dominant_colors.get('percentages', [])
            
            for j, rgb_color in enumerate(colors_rgb):
                lab_values = convert_rgb_to_lab_proper(rgb_color)
                result[f'{prefix}_color{j+1}_L'] = lab_values['L']
                result[f'{prefix}_color{j+1}_a'] = lab_values['a']
                result[f'{prefix}_color{j+1}_b'] = lab_values['b']
                result[f'{prefix}_color{j+1}_percentage'] = float(percentages[j]) if j < len(percentages) else 0.0
        
        return result

class IncrementalCSVWriter:
    """Write results to CSV incrementally"""
    
    def __init__(self, output_path, resume=True):
        self.output_path = output_path
        self.temp_path = output_path.replace('.csv', '_temp.csv')
        self.processed_files = set()
        self.header_written = False
        
        # Load existing progress if resuming
        if resume and os.path.exists(self.output_path):
            self._load_existing_progress()
    
    def _load_existing_progress(self):
        """Load existing processed files"""
        try:
            df = pd.read_csv(self.output_path)
            self.processed_files = set(df['image_filename'].tolist())
            self.header_written = True
            print(f"📁 Resuming: Found {len(self.processed_files)} previously processed images")
        except Exception as e:
            print(f"⚠️ Could not load existing progress: {e}")
            self.processed_files = set()
    
    def is_processed(self, filename):
        """Check if file was already processed"""
        return filename in self.processed_files
    
    def write_result(self, result_data):
        """Write single result to CSV"""
        try:
            df = pd.DataFrame([result_data])
            
            # Write header if first time
            mode = 'w' if not self.header_written else 'a'
            header = not self.header_written
            
            df.to_csv(self.output_path, mode=mode, header=header, index=False)
            
            self.header_written = True
            self.processed_files.add(result_data['image_filename'])
            
        except Exception as e:
            print(f"❌ Error writing to CSV: {e}")
    
    def get_processed_count(self):
        """Get number of processed files"""
        return len(self.processed_files)

def main():
    parser = argparse.ArgumentParser(description='GPU-optimized batch eyebrow color analysis')
    parser.add_argument('input_folder', help='Path to folder containing images')
    parser.add_argument('output_csv', help='Path to output CSV file')
    parser.add_argument('--k_clusters', type=int, default=2, help='Number of clusters (default: 2)')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process')
    parser.add_argument('--resize_max', type=int, default=800, help='Max image dimension (default: 800)')
    parser.add_argument('--cache_clear_interval', type=int, default=10, help='Clear GPU cache every N images (default: 10)')
    parser.add_argument('--no_resume', action='store_true', help='Start fresh (ignore existing progress)')
    parser.add_argument('--use_cpu', action='store_true', help='Force CPU processing')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input_folder):
        print(f"❌ Input folder '{args.input_folder}' does not exist")
        return
    
    # Create output directory
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize GPU analyzer
    use_gpu = torch.cuda.is_available() and not args.use_cpu
    if use_gpu:
        print(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name()}")
    else:
        print("💻 Using CPU processing")
    
    try:
        analyzer = GPUColorAnalyzer(use_gpu=use_gpu)
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return
    
    # Get image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = []
    
    for root, dirs, files in os.walk(args.input_folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"❌ No image files found in '{args.input_folder}'")
        return
    
    # Limit images if specified
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"📁 Found {len(image_files)} images to process")
    
    # Initialize CSV writer
    csv_writer = IncrementalCSVWriter(args.output_csv, resume=not args.no_resume)
    
    # Filter out already processed images
    remaining_files = [f for f in image_files if not csv_writer.is_processed(os.path.basename(f))]
    
    if csv_writer.get_processed_count() > 0:
        print(f"📋 Skipping {len(image_files) - len(remaining_files)} already processed images")
    
    if not remaining_files:
        print("✅ All images already processed!")
        return
    
    print(f"🔄 Processing {len(remaining_files)} remaining images")
    print(f"🎯 K-means clusters: {args.k_clusters}")
    print(f"📊 GPU cache clearing interval: {args.cache_clear_interval} images")
    
    # Process images with progress bar
    successful = 0
    failed = 0
    start_time = time.time()
    
    with tqdm(total=len(remaining_files), desc="Processing images", 
              unit="img", dynamic_ncols=True) as pbar:
        
        for i, image_path in enumerate(remaining_files):
            try:
                # Update progress bar description with current file
                pbar.set_description(f"Processing {os.path.basename(image_path)[:20]}...")
                
                # Process image
                result, error = analyzer.process_single_image(
                    image_path, args.k_clusters, args.resize_max)
                
                if result:
                    # Write to CSV immediately
                    csv_writer.write_result(result)
                    successful += 1
                    
                    # Update progress bar with success info
                    pbar.set_postfix({
                        'Success': successful,
                        'Failed': failed,
                        'Memory': analyzer.get_memory_info().split(',')[0]  # Just CPU%
                    })
                else:
                    failed += 1
                    tqdm.write(f"❌ {os.path.basename(image_path)}: {error}")
                
                # Clear GPU cache periodically
                if (i + 1) % args.cache_clear_interval == 0:
                    analyzer.clear_gpu_cache()
                    tqdm.write(f"🧹 Cleared GPU cache - {analyzer.get_memory_info()}")
                
                pbar.update(1)
                
            except KeyboardInterrupt:
                tqdm.write("\n⏹️ Interrupted by user")
                break
            except Exception as e:
                failed += 1
                tqdm.write(f"💥 Unexpected error with {os.path.basename(image_path)}: {e}")
                pbar.update(1)
    
    # Final statistics
    elapsed_time = time.time() - start_time
    total_processed = successful + failed
    rate = total_processed / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n📊 Final Results:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⏱️ Total time: {elapsed_time/60:.1f} minutes")
    print(f"   🚀 Processing rate: {rate:.1f} images/second")
    print(f"   📁 CSV file: {args.output_csv}")
    
    # Show CSV info
    if successful > 0:
        try:
            df = pd.read_csv(args.output_csv)
            print(f"   📋 Total CSV rows: {len(df)}")
            print(f"   📋 Total CSV columns: {len(df.columns)}")
        except:
            pass
    
    # Final GPU cleanup
    analyzer.clear_gpu_cache()
    print(f"🧹 Final cleanup complete")

if __name__ == "__main__":
    main()