"""
Comprehensive Eyebrow Color Analysis and Visualization Script
Analyzes CSV data from batch_robust_color_analysis.py and creates stunning visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import colorsys
import argparse
import os
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EyebrowColorAnalyzer:
    def __init__(self, csv_path, image_folder=None):
        """Initialize with CSV data and optional image folder for interactive viewing"""
        self.df = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.setup_data()
        
    def setup_data(self):
        """Prepare data for analysis"""
        print(f"Loaded {len(self.df)} images for analysis")
        print(f"Dataset columns: {len(self.df.columns)}")
        
        # Extract method performance data
        self.method_performance = self.extract_method_performance()
        
        # Extract color data
        self.color_data = self.extract_color_data()
        
    def extract_method_performance(self):
        """Extract method performance statistics"""
        methods_data = []
        
        for side in ['left', 'right']:
            for rank in [1, 2, 3]:
                method_col = f'{side}_top_method{rank}'
                score_col = f'{side}_top_method{rank}_quality_score'
                
                if method_col in self.df.columns and score_col in self.df.columns:
                    for idx, row in self.df.iterrows():
                        methods_data.append({
                            'image': row['image_filename'],
                            'side': side,
                            'rank': rank,
                            'method': row[method_col],
                            'quality_score': row[score_col]
                        })
        
        return pd.DataFrame(methods_data)
    
    def extract_color_data(self):
        """Extract all color data from the dataset"""
        colors_data = []
        
        for side in ['left', 'right']:
            for rank in [1, 2, 3]:
                for color_idx in [1, 2]:
                    l_col = f'{side}_top_method{rank}_color{color_idx}_L'
                    a_col = f'{side}_top_method{rank}_color{color_idx}_a'
                    b_col = f'{side}_top_method{rank}_color{color_idx}_b'
                    pct_col = f'{side}_top_method{rank}_color{color_idx}_percentage'
                    method_col = f'{side}_top_method{rank}'
                    score_col = f'{side}_top_method{rank}_quality_score'
                    
                    if all(col in self.df.columns for col in [l_col, a_col, b_col, pct_col]):
                        for idx, row in self.df.iterrows():
                            colors_data.append({
                                'image': row['image_filename'],
                                'side': side,
                                'rank': rank,
                                'color_idx': color_idx,
                                'method': row[method_col],
                                'L': row[l_col],
                                'a': row[a_col],
                                'b': row[b_col],
                                'percentage': row[pct_col],
                                'quality_score': row[score_col]
                            })
        
        return pd.DataFrame(colors_data)
    
    def lab_to_rgb(self, l, a, b):
        """Convert LAB to RGB for visualization"""
        from colormath.color_objects import LabColor, sRGBColor
        from colormath.color_conversions import convert_color
        
        try:
            lab = LabColor(l, a, b)
            rgb = convert_color(lab, sRGBColor)
            return (int(rgb.rgb_r * 255), int(rgb.rgb_g * 255), int(rgb.rgb_b * 255)) # type: ignore
        except:
            return (128, 128, 128)  # Gray fallback
    
    def create_method_performance_dashboard(self):
        """Create comprehensive method performance visualizations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Method Frequency (Rank 1)', 'Quality Score Distribution', 
                          'Method Performance by Side', 'Quality Score vs Rank'),
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Method frequency for rank 1 (best methods)
        rank1_methods = self.method_performance[self.method_performance['rank'] == 1]
        method_counts = rank1_methods['method'].value_counts()
        
        fig.add_trace(
            go.Bar(x=method_counts.index, y=method_counts.values, 
                   name="Method Frequency", showlegend=False),
            row=1, col=1
        )
        
        # 2. Quality score distribution by method
        methods = self.method_performance['method'].unique()
        for method in methods[:5]:  # Top 5 methods
            method_data = self.method_performance[self.method_performance['method'] == method]
            fig.add_trace(
                go.Box(y=method_data['quality_score'], name=method, showlegend=False),
                row=1, col=2
            )
        
        # 3. Method performance by side
        side_method = self.method_performance.groupby(['side', 'method'])['quality_score'].mean().reset_index()
        for side in ['left', 'right']:
            side_data = side_method[side_method['side'] == side]
            fig.add_trace(
                go.Bar(x=side_data['method'], y=side_data['quality_score'], 
                       name=f"{side.title()} Eyebrow"),
                row=2, col=1
            )
        
        # 4. Quality score vs rank
        fig.add_trace(
            go.Scatter(x=self.method_performance['rank'], y=self.method_performance['quality_score'],
                      mode='markers', opacity=0.6, showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Method Performance Dashboard")
        fig.write_html("method_performance_dashboard.html")
        print("✅ Method performance dashboard saved as 'method_performance_dashboard.html'")
        
        return fig
    
    def create_color_space_visualization(self):
        """Create 3D LAB color space visualization"""
        # Focus on rank 1 colors (best methods)
        rank1_colors = self.color_data[self.color_data['rank'] == 1].copy()
        
        # Add RGB colors for visualization
        rank1_colors['rgb'] = rank1_colors.apply(
            lambda row: self.lab_to_rgb(row['L'], row['a'], row['b']), axis=1
        )
        rank1_colors['rgb_str'] = rank1_colors['rgb'].apply(
            lambda rgb: f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'
        )
        
        fig = go.Figure()
        
        # Add points for each side
        for side in ['left', 'right']:
            side_data = rank1_colors[rank1_colors['side'] == side]
            
            fig.add_trace(go.Scatter3d(
                x=side_data['L'],
                y=side_data['a'],
                z=side_data['b'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=side_data['rgb_str'],
                    opacity=0.8,
                    line=dict(width=1, color='black')
                ),
                name=f'{side.title()} Eyebrow',
                text=side_data.apply(lambda row: 
                    f"Image: {row['image']}<br>"
                    f"Method: {row['method']}<br>"
                    f"LAB: ({row['L']:.1f}, {row['a']:.1f}, {row['b']:.1f})<br>"
                    f"Percentage: {row['percentage']:.1f}%", axis=1),
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title="3D LAB Color Space Distribution (Best Methods)",
            scene=dict(
                xaxis_title="L* (Lightness)",
                yaxis_title="a* (Green-Red)",
                zaxis_title="b* (Blue-Yellow)"
            ),
            height=700
        )
        
        fig.write_html("lab_color_space_3d.html")
        print("✅ 3D LAB color space visualization saved as 'lab_color_space_3d.html'")
        
        return fig
    
    def create_color_palette_grid(self):
        """Create a grid showing actual colors for each image"""
        rank1_colors = self.color_data[self.color_data['rank'] == 1].copy()
        
        # Group by image and side
        images = rank1_colors['image'].unique()[:20]  # First 20 images for visibility
        
        fig, axes = plt.subplots(len(images), 4, figsize=(16, len(images) * 2))
        if len(images) == 1:
            axes = axes.reshape(1, -1)
        
        for i, image in enumerate(images):
            image_data = rank1_colors[rank1_colors['image'] == image]
            
            for j, (side, color_idx) in enumerate([('left', 1), ('left', 2), ('right', 1), ('right', 2)]):
                color_data = image_data[
                    (image_data['side'] == side) & (image_data['color_idx'] == color_idx)
                ]
                
                if not color_data.empty:
                    row = color_data.iloc[0]
                    rgb = self.lab_to_rgb(row['L'], row['a'], row['b'])
                    rgb_norm = tuple(c/255.0 for c in rgb)
                    
                    axes[i, j].add_patch(Rectangle((0, 0), 1, 1, color=rgb_norm))
                    axes[i, j].text(0.5, 0.5, f"{row['percentage']:.1f}%", 
                                   ha='center', va='center', fontweight='bold',
                                   color='white' if sum(rgb_norm) < 1.5 else 'black')
                    axes[i, j].set_title(f"{side.title()} C{color_idx}")
                else:
                    axes[i, j].set_facecolor('lightgray')
                    axes[i, j].text(0.5, 0.5, 'N/A', ha='center', va='center')
                
                axes[i, j].set_xlim(0, 1)
                axes[i, j].set_ylim(0, 1)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            
            # Add image name
            fig.text(0.02, 1 - (i + 0.5) / len(images), image, 
                    rotation=90, va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('color_palette_grid.png', dpi=300, bbox_inches='tight')
        print("✅ Color palette grid saved as 'color_palette_grid.png'")
        plt.close()
    
    def perform_color_clustering(self, n_clusters=8):
        """Perform K-means clustering on all colors to find main color groups"""
        # Use all colors from rank 1 methods
        rank1_colors = self.color_data[self.color_data['rank'] == 1].copy()
        
        # Prepare data for clustering
        color_features = rank1_colors[['L', 'a', 'b']].values
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(color_features)
        
        # Add cluster labels
        rank1_colors['cluster'] = clusters
        
        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cluster Distribution', 'Cluster Centers (LAB)', 
                          'Cluster by Side', 'Cluster Colors'),
            specs=[[{"type": "bar"}, {"type": "scatter3d"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Cluster distribution
        cluster_counts = rank1_colors['cluster'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=cluster_counts.index, y=cluster_counts.values, 
                   name="Cluster Size", showlegend=False),
            row=1, col=1
        )
        
        # 2. 3D scatter of clusters
        colors = px.colors.qualitative.Set3[:n_clusters]
        for cluster_id in range(n_clusters):
            cluster_data = rank1_colors[rank1_colors['cluster'] == cluster_id]
            fig.add_trace(
                go.Scatter3d(
                    x=cluster_data['L'], y=cluster_data['a'], z=cluster_data['b'],
                    mode='markers',
                    marker=dict(size=5, color=colors[cluster_id]),
                    name=f'Cluster {cluster_id}',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Add cluster centers
        fig.add_trace(
            go.Scatter3d(
                x=cluster_centers[:, 0], y=cluster_centers[:, 1], z=cluster_centers[:, 2],
                mode='markers',
                marker=dict(size=15, color='black', symbol='diamond'),
                name='Centers',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Cluster distribution by side
        cluster_side = rank1_colors.groupby(['cluster', 'side']).size().reset_index(name='count')
        for side in ['left', 'right']:
            side_data = cluster_side[cluster_side['side'] == side]
            fig.add_trace(
                go.Bar(x=side_data['cluster'], y=side_data['count'], 
                       name=f"{side.title()} Eyebrow"),
                row=2, col=1
            )
        
        # 4. Show actual cluster colors
        cluster_rgb_colors = []
        for center in cluster_centers:
            rgb = self.lab_to_rgb(center[0], center[1], center[2])
            cluster_rgb_colors.append(f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})')
        
        fig.add_trace(
            go.Bar(x=list(range(n_clusters)), y=[1]*n_clusters,
                   marker_color=cluster_rgb_colors,
                   name="Cluster Colors", showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text=f"Color Clustering Analysis ({n_clusters} clusters)")
        fig.write_html("color_clustering_analysis.html")
        print("✅ Color clustering analysis saved as 'color_clustering_analysis.html'")
        
        # Create cluster summary
        cluster_summary = []
        for cluster_id in range(n_clusters):
            cluster_data = rank1_colors[rank1_colors['cluster'] == cluster_id]
            center = cluster_centers[cluster_id]
            rgb = self.lab_to_rgb(center[0], center[1], center[2])
            
            cluster_summary.append({
                'cluster_id': cluster_id,
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(rank1_colors) * 100,
                'center_L': center[0],
                'center_a': center[1],
                'center_b': center[2],
                'center_rgb': rgb,
                'avg_quality_score': cluster_data['quality_score'].mean()
            })
        
        cluster_df = pd.DataFrame(cluster_summary)
        cluster_df.to_csv('color_clusters_summary.csv', index=False)
        print("✅ Cluster summary saved as 'color_clusters_summary.csv'")
        
        return fig, cluster_df
    
    def create_statistical_summary(self):
        """Create comprehensive statistical summary"""
        stats = {}
        
        # Method statistics
        rank1_methods = self.method_performance[self.method_performance['rank'] == 1]
        stats['most_successful_method'] = rank1_methods['method'].mode()[0]
        stats['avg_quality_score'] = rank1_methods['quality_score'].mean()
        stats['method_consistency'] = rank1_methods['method'].value_counts().iloc[0] / len(rank1_methods)
        
        # Color statistics
        rank1_colors = self.color_data[self.color_data['rank'] == 1]
        stats['avg_lightness'] = rank1_colors['L'].mean()
        stats['lightness_std'] = rank1_colors['L'].std()
        stats['avg_a_value'] = rank1_colors['a'].mean()
        stats['avg_b_value'] = rank1_colors['b'].mean()
        
        # Side comparison
        left_colors = rank1_colors[rank1_colors['side'] == 'left']
        right_colors = rank1_colors[rank1_colors['side'] == 'right']
        
        stats['left_avg_lightness'] = left_colors['L'].mean()
        stats['right_avg_lightness'] = right_colors['L'].mean()
        stats['lightness_difference'] = abs(stats['left_avg_lightness'] - stats['right_avg_lightness'])
        
        # Save statistics
        with open('eyebrow_analysis_summary.txt', 'w') as f:
            f.write("EYEBROW COLOR ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {len(self.df)} images analyzed\n")
            f.write(f"Total color extractions: {len(self.color_data)}\n\n")
            
            f.write("METHOD PERFORMANCE:\n")
            f.write(f"Most successful method: {stats['most_successful_method']}\n")
            f.write(f"Average quality score: {stats['avg_quality_score']:.2f}\n")
            f.write(f"Method consistency: {stats['method_consistency']:.2%}\n\n")
            
            f.write("COLOR CHARACTERISTICS:\n")
            f.write(f"Average lightness (L*): {stats['avg_lightness']:.1f} ± {stats['lightness_std']:.1f}\n")
            f.write(f"Average a* value: {stats['avg_a_value']:.1f}\n")
            f.write(f"Average b* value: {stats['avg_b_value']:.1f}\n\n")
            
            f.write("SIDE COMPARISON:\n")
            f.write(f"Left eyebrow avg lightness: {stats['left_avg_lightness']:.1f}\n")
            f.write(f"Right eyebrow avg lightness: {stats['right_avg_lightness']:.1f}\n")
            f.write(f"Lightness difference: {stats['lightness_difference']:.1f}\n")
        
        print("✅ Statistical summary saved as 'eyebrow_analysis_summary.txt'")
        return stats
    
    def create_interactive_image_dashboard(self):
        """Create an interactive dashboard where clicking on points shows the actual images"""
        if not self.image_folder:
            print("⚠️  Image folder not provided. Skipping interactive image dashboard.")
            return None
            
        # Focus on rank 1 colors (best methods)
        rank1_colors = self.color_data[self.color_data['rank'] == 1].copy()
        
        # Add RGB colors for visualization
        rank1_colors['rgb'] = rank1_colors.apply(
            lambda row: self.lab_to_rgb(row['L'], row['a'], row['b']), axis=1
        )
        rank1_colors['rgb_str'] = rank1_colors['rgb'].apply(
            lambda rgb: f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'
        )
        
        # Create HTML template for interactive dashboard
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Eyebrow Color Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { display: flex; gap: 20px; }
        .plot-container { flex: 1; }
        .image-container { 
            flex: 0 0 400px; 
            border: 2px solid #ddd; 
            border-radius: 10px; 
            padding: 20px;
            background-color: #f9f9f9;
        }
        .image-info {
            margin-bottom: 15px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .image-display img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .color-info {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .color-swatch {
            width: 40px;
            height: 40px;
            border-radius: 5px;
            border: 2px solid #333;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: bold;
            color: white;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
        }
        h1 { color: #333; text-align: center; }
        h3 { color: #555; margin-top: 0; }
    </style>
</head>
<body>
    <h1>🎨 Interactive Eyebrow Color Dashboard</h1>
    <p style="text-align: center; color: #666;">Click on any point in the 3D plot to see the corresponding image and color details!</p>
    
    <div class="container">
        <div class="plot-container">
            <div id="plot3d"></div>
        </div>
        <div class="image-container">
            <div class="image-info">
                <h3>📸 Image Details</h3>
                <div id="image-details">Click on a point in the 3D plot to see image details here!</div>
            </div>
            <div class="image-display">
                <div id="image-holder">
                    <div style="text-align: center; color: #999; padding: 50px;">
                        🖼️<br>Image will appear here when you click a point
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Plot data will be inserted here
        var plotData = PLOT_DATA_PLACEHOLDER;
        var imageData = IMAGE_DATA_PLACEHOLDER;
        
        var layout = {
            title: '3D LAB Color Space (Click points to see images!)',
            scene: {
                xaxis: { title: 'L* (Lightness)' },
                yaxis: { title: 'a* (Green-Red)' },
                zaxis: { title: 'b* (Blue-Yellow)' }
            },
            height: 600
        };
        
        Plotly.newPlot('plot3d', plotData, layout);
        
        // Handle click events
        document.getElementById('plot3d').on('plotly_click', function(data) {
            var point = data.points[0];
            var imageInfo = imageData[point.pointIndex];
            
            // Update image details
            document.getElementById('image-details').innerHTML = `
                <strong>📁 File:</strong> ${imageInfo.filename}<br>
                <strong>👁️ Side:</strong> ${imageInfo.side} eyebrow<br>
                <strong>🔬 Method:</strong> ${imageInfo.method}<br>
                <strong>🎯 Quality Score:</strong> ${imageInfo.quality_score.toFixed(2)}<br>
                <strong>🎨 LAB:</strong> (${imageInfo.L.toFixed(1)}, ${imageInfo.a.toFixed(1)}, ${imageInfo.b.toFixed(1)})<br>
                <strong>📊 Percentage:</strong> ${imageInfo.percentage.toFixed(1)}%
            `;
            
            // Update image display
            document.getElementById('image-holder').innerHTML = `
                <img src="${imageInfo.image_path}" alt="${imageInfo.filename}" 
                     onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4='; this.alt='Image not found';">
                <div class="color-info">
                    <div class="color-swatch" style="background-color: ${imageInfo.rgb_str};">
                        ${imageInfo.percentage.toFixed(0)}%
                    </div>
                </div>
            `;
        });
    </script>
</body>
</html>
        """
        
        # Prepare plot data
        plot_data = []
        image_data = []
        
        for side in ['left', 'right']:
            side_data = rank1_colors[rank1_colors['side'] == side]
            
            plot_data.append({
                'x': side_data['L'].tolist(),
                'y': side_data['a'].tolist(), 
                'z': side_data['b'].tolist(),
                'mode': 'markers',
                'type': 'scatter3d',
                'name': f'{side.title()} Eyebrow',
                'marker': {
                    'size': 8,
                    'color': side_data['rgb_str'].tolist(),
                    'opacity': 0.8,
                    'line': {'width': 1, 'color': 'black'}
                },
                'text': side_data['image'].tolist(),
                'hovertemplate': '%{text}<extra></extra>'
            })
            
            # Prepare image data for JavaScript
            for _, row in side_data.iterrows():
                image_path = os.path.join(self.image_folder, row['image']) if self.image_folder else row['image']
                image_data.append({
                    'filename': row['image'],
                    'side': row['side'],
                    'method': row['method'],
                    'L': row['L'],
                    'a': row['a'],
                    'b': row['b'],
                    'percentage': row['percentage'],
                    'quality_score': row['quality_score'],
                    'rgb_str': row['rgb_str'],
                    'image_path': image_path
                })
        
        # Replace placeholders in HTML
        html_content = html_template.replace('PLOT_DATA_PLACEHOLDER', str(plot_data).replace("'", '"'))
        html_content = html_content.replace('IMAGE_DATA_PLACEHOLDER', str(image_data).replace("'", '"'))
        
        # Save interactive dashboard
        with open('interactive_eyebrow_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("✅ Interactive image dashboard saved as 'interactive_eyebrow_dashboard.html'")
        print("   🖱️  Click on any point to see the corresponding image!")
        
        return html_content
    
    def run_complete_analysis(self):
        """Run all analysis and create all visualizations"""
        print("🎨 Starting comprehensive eyebrow color analysis...")
        print("=" * 60)
        
        # 1. Method performance analysis
        print("📊 Creating method performance dashboard...")
        self.create_method_performance_dashboard()
        
        # 2. Color space visualization
        print("🌈 Creating 3D LAB color space visualization...")
        self.create_color_space_visualization()
        
        # 3. Color palette grid
        print("🎨 Creating color palette grid...")
        self.create_color_palette_grid()
        
        # 4. Color clustering
        print("🔍 Performing color clustering analysis...")
        self.perform_color_clustering()
        
        # 5. Interactive image dashboard
        print("📸 Creating interactive image dashboard...")
        self.create_interactive_image_dashboard()
        
        # 6. Statistical summary
        print("📈 Generating statistical summary...")
        self.create_statistical_summary()
        
        print("\n✅ Analysis complete! Generated files:")
        print("   📄 method_performance_dashboard.html")
        print("   📄 lab_color_space_3d.html") 
        print("   📄 color_clustering_analysis.html")
        print("   🖼️  color_palette_grid.png")
        print("   📊 color_clusters_summary.csv")
        print("   📝 eyebrow_analysis_summary.txt")
        print("   📄 interactive_eyebrow_dashboard.html")
        print("\n🎉 Open the HTML files in your browser for interactive visualizations!")

def main():
    parser = argparse.ArgumentParser(description='Analyze eyebrow color data and create interactive visualizations with clickable images')
    parser.add_argument('csv_file', help='Path to the CSV file from batch_robust_color_analysis.py')
    parser.add_argument('--clusters', type=int, default=8, help='Number of clusters for color analysis')
    parser.add_argument('--image_folder', help='Path to the folder containing images for interactive viewing (enables click-to-view images)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"❌ Error: CSV file '{args.csv_file}' not found!")
        return
    
    if args.image_folder and not os.path.exists(args.image_folder):
        print(f"⚠️  Warning: Image folder '{args.image_folder}' not found! Interactive images will not work.")
    
    # Create analyzer and run analysis
    analyzer = EyebrowColorAnalyzer(args.csv_file, image_folder=args.image_folder)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
