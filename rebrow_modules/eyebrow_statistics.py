import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from skimage import color
import io
import base64
from PIL import Image
import streamlit as st

class EyebrowStatistics:
    def __init__(self):
        self.data = None
        
    def lab_to_rgb(self, L, a, b):
        """Convert LAB color to RGB color."""
        # Convert LAB to RGB
        lab = np.array([L, a, b]).reshape(1, 1, 3)
        # Convert to RGB (0-1 scale)
        rgb = color.lab2rgb(lab)[0, 0]
        # Convert to RGB (0-255 scale)
        rgb_255 = (rgb * 255).astype(int)
        return rgb_255
    
    def get_color_hex(self, L, a, b):
        """Convert LAB color to hex color code."""
        rgb = self.lab_to_rgb(L, a, b)
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
        
    def create_color_swatch(self, L, a, b, size=(100, 100)):
        """Create a color swatch image from LAB values."""
        rgb = self.lab_to_rgb(L, a, b)
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        img[:, :] = rgb
        return img
        
    def load_data(self, csv_path):
        self.data = pd.read_csv(csv_path)
        
    def create_3d_color_scatter(self, side='left', color_number=1):
        """Create a 3D scatter plot for a specific dominant color."""
        if self.data is None:
            return None
            
        prefix = f"{side}_eyebrow_color{color_number}"
        
        # Get the data and handle NaN values
        L_vals = self.data[f'{prefix}_L'].fillna(0)
        a_vals = self.data[f'{prefix}_a'].fillna(0)
        b_vals = self.data[f'{prefix}_b'].fillna(0)
        percentages = self.data[f'{prefix}_percentage'].fillna(0)
        
        # Filter out rows where all values are 0 (originally NaN)
        mask = ~((L_vals == 0) & (a_vals == 0) & (b_vals == 0))
        L_vals = L_vals[mask]
        a_vals = a_vals[mask]
        b_vals = b_vals[mask]
        percentages = percentages[mask]
        image_filenames = self.data['image_filename'][mask]
        
        # Create a list of RGB colors from LAB values
        rgb_colors = []
        hex_colors = []
        for l, a_val, b_val in zip(L_vals, a_vals, b_vals):
            hex_color = self.get_color_hex(l, a_val, b_val)
            hex_colors.append(hex_color)
            rgb = self.lab_to_rgb(l, a_val, b_val)
            rgb_colors.append(f'rgb({rgb[0]},{rgb[1]},{rgb[2]})')
        
        fig = go.Figure(data=[go.Scatter3d(
            x=L_vals,
            y=a_vals,
            z=b_vals,
            mode='markers',
            marker=dict(
                size=percentages/2,  # Size proportional to percentage
                color=rgb_colors,    # Using actual RGB colors
                opacity=0.8
            ),
            text=[f"Image: {img}<br>L: {l:.1f}<br>a: {a_val:.1f}<br>b: {b_val:.1f}<br>%: {p:.1f}<br>Color: {color}" 
                  for img, l, a_val, b_val, p, color in zip(image_filenames, L_vals, a_vals, b_vals, percentages, hex_colors)],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title=f"{side.capitalize()} Eyebrow - Dominant Color {color_number}",
            scene=dict(
                xaxis_title='L',
                yaxis_title='a',
                zaxis_title='b'
            ),
            width=600,
            height=600
        )
        
        return fig
        
    def create_percentage_histogram(self, side='left'):
        """Create histograms of color percentages."""
        if self.data is None:
            return None
            
        fig = go.Figure()
        colors = ['#6495ED', '#4682B4', '#1E90FF']  # Blue colors for better visibility
        
        for i in range(1, 4):
            col = f"{side}_eyebrow_color{i}_percentage"
            # Get the average LAB values for this color to show in the legend
            prefix = f"{side}_eyebrow_color{i}"
            avg_L = self.data[f'{prefix}_L'].mean()
            avg_a = self.data[f'{prefix}_a'].mean()
            avg_b = self.data[f'{prefix}_b'].mean()
            hex_color = self.get_color_hex(avg_L, avg_a, avg_b)
            
            # Filter out NaN values
            valid_data = self.data[col].dropna()
            
            fig.add_trace(go.Histogram(
                x=valid_data,
                name=f'Color {i} (avg: {valid_data.mean():.1f}%)',
                marker_color=hex_color,
                opacity=0.7,
                nbinsx=20,
                hovertemplate='Percentage: %{x:.1f}%<br>Count: %{y}<br>'
            ))
            
        fig.update_layout(
            title=f"{side.capitalize()} Eyebrow - Color Distribution Percentages",
            xaxis_title="Percentage of Eyebrow Area",
            yaxis_title="Number of Images",
            barmode='overlay',
            width=600,
            height=400,
            legend_title="Dominant Colors",
            hovermode='closest'
        )
        
        # Add annotation explaining the chart
        fig.add_annotation(
            text="This histogram shows how frequently each percentage occurs in the dataset.<br>For example, a peak at 40% for Color 1 means many images have Color 1 covering 40% of the eyebrow.",
            xref="paper", yref="paper",
            x=0.5, y=1.15,
            showarrow=False,
            font=dict(size=10),
            align="center"
        )
        
        return fig
        
    def get_summary_statistics(self):
        """Get summary statistics for all colors."""
        if self.data is None:
            return None
            
        stats = []
        for side in ['left', 'right']:
            for i in range(1, 4):
                prefix = f"{side}_eyebrow_color{i}"
                avg_L = self.data[f'{prefix}_L'].mean()
                avg_a = self.data[f'{prefix}_a'].mean()
                avg_b = self.data[f'{prefix}_b'].mean()
                avg_pct = self.data[f'{prefix}_percentage'].mean()
                
                # Get the hex color for the average LAB values
                hex_color = self.get_color_hex(avg_L, avg_a, avg_b)
                
                color_stats = {
                    'Side': side.capitalize(),
                    'Color': i,
                    'Avg L': avg_L,
                    'Avg a': avg_a,
                    'Avg b': avg_b,
                    'Avg %': avg_pct,
                    'Hex Color': hex_color
                }
                stats.append(color_stats)
                
        return pd.DataFrame(stats)
        
    def create_dominant_color_swatches(self, side='left'):
        """Create an image with the dominant colors for a side."""
        if self.data is None:
            return None
            
        # Get average LAB values for each dominant color
        avg_colors = []
        avg_percentages = []
        
        for i in range(1, 4):
            prefix = f"{side}_eyebrow_color{i}"
            avg_L = self.data[f'{prefix}_L'].mean()
            avg_a = self.data[f'{prefix}_a'].mean()
            avg_b = self.data[f'{prefix}_b'].mean()
            avg_pct = self.data[f'{prefix}_percentage'].mean()
            
            avg_colors.append((avg_L, avg_a, avg_b))
            avg_percentages.append(avg_pct)
        
        # Create figure with color swatches
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        
        for i, ((L, a, b), pct) in enumerate(zip(avg_colors, avg_percentages)):
            # Create color swatch
            rgb = self.lab_to_rgb(L, a, b)
            hex_color = self.get_color_hex(L, a, b)
            
            # Display color swatch
            ax[i].add_patch(patches.Rectangle((0, 0), 1, 1, color=hex_color))
            ax[i].set_xlim(0, 1)
            ax[i].set_ylim(0, 1)
            ax[i].set_title(f'Color {i+1}: {pct:.1f}%')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].text(0.5, 0.5, f'L: {L:.1f}\na: {a:.1f}\nb: {b:.1f}', 
                      ha='center', va='center', color='white' if sum(rgb) < 380 else 'black')
        
        plt.suptitle(f'{side.capitalize()} Eyebrow - Dominant Colors')
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        
        return Image.open(buf)
        
    def create_icicle_plot(self):
        """Create a sunburst chart showing the hierarchy of eyebrow colors."""
        try:
            # Check if data exists
            if self.data is None or len(self.data) == 0:
                return None
                
            # Create a sunburst chart for a single image or a small sample
            # Get a sample of images (up to 10) to show individual eyebrow colors
            sample_size = min(10, len(self.data))
            sample_data = self.data.sample(n=sample_size) if len(self.data) > sample_size else self.data.copy()
            
            # Create the base structure
            labels = ['All Eyebrows']
            parents = ['']
            values = [100]
            colors = ['#FFFFFF']
            
            # Add individual images
            for idx, row in sample_data.iterrows():
                # Convert to string to handle numeric values
                if 'image_filename' in row and not pd.isna(row['image_filename']):
                    image_name = str(row['image_filename'])
                else:
                    image_name = f'Image {idx}'
                    
                # Truncate long filenames
                if len(image_name) > 20:
                    image_name = image_name[:17] + '...'
                    
                # Add image node
                labels.append(image_name)
                parents.append('All Eyebrows')
                values.append(10)  # Equal weight for each image
                colors.append('#EEEEEE')
                
                # Add left and right eyebrows for this image
                for side in ['left', 'right']:
                    side_label = f"{image_name} {side.capitalize()}"
                    labels.append(side_label)
                    parents.append(image_name)
                    values.append(5)  # Equal weight for left and right
                    colors.append('#DDDDDD')
                    
                    # Add the three dominant colors for this side
                    for color_num in range(1, 4):
                        prefix = f"{side}_eyebrow_color{color_num}"
                        
                        # Check if color data exists for this image
                        if (f'{prefix}_L' in row and 
                            f'{prefix}_a' in row and 
                            f'{prefix}_b' in row and 
                            f'{prefix}_percentage' in row):
                            
                            L_val = row[f'{prefix}_L']
                            a_val = row[f'{prefix}_a']
                            b_val = row[f'{prefix}_b']
                            pct = row[f'{prefix}_percentage']
                            
                            # Skip if any values are NaN
                            if pd.isna(L_val) or pd.isna(a_val) or pd.isna(b_val) or pd.isna(pct):
                                continue
                                
                            color_id = f"{side_label} Color {color_num}"
                            hex_color = self.get_color_hex(L_val, a_val, b_val)
                            
                            labels.append(color_id)
                            parents.append(side_label)
                            values.append(pct)
                            colors.append(hex_color)
            
            # Create sunburst chart
            fig = go.Figure(go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                marker=dict(colors=colors),
                hovertemplate='<b>%{label}</b><br>Percentage: %{value:.1f}%<br>',
                textinfo="label+percent entry"
            ))
            
            fig.update_layout(
                title="Individual Eyebrow Color Visualization (Sample of Images)",
                width=700,
                height=700,
                margin=dict(t=50, l=0, r=0, b=0)
            )
            
            return fig
            
        except Exception as e:
            print(f"Error in create_icicle_plot: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        return fig
        
    def create_color_distribution_plot(self):
        """Create a plot showing the distribution of colors in LAB space."""
        if self.data is None:
            return None
            
        # Create a 3D figure
        fig = go.Figure()
        
        # Add points for each color group
        for side in ['left', 'right']:
            for color_num in range(1, 4):
                prefix = f"{side}_eyebrow_color{color_num}"
                
                # Get data and handle NaN values
                L_vals = self.data[f'{prefix}_L'].fillna(0)
                a_vals = self.data[f'{prefix}_a'].fillna(0)
                b_vals = self.data[f'{prefix}_b'].fillna(0)
                percentages = self.data[f'{prefix}_percentage'].fillna(0)
                
                # Filter out rows where all values are 0 (originally NaN)
                mask = ~((L_vals == 0) & (a_vals == 0) & (b_vals == 0))
                L_vals = L_vals[mask]
                a_vals = a_vals[mask]
                b_vals = b_vals[mask]
                percentages = percentages[mask]
                
                # Create RGB colors
                rgb_colors = []
                for l, a_val, b_val in zip(L_vals, a_vals, b_vals):
                    rgb = self.lab_to_rgb(l, a_val, b_val)
                    rgb_colors.append(f'rgb({rgb[0]},{rgb[1]},{rgb[2]})')
                
                # Add scatter trace
                fig.add_trace(go.Scatter3d(
                    x=L_vals,
                    y=a_vals,
                    z=b_vals,
                    mode='markers',
                    marker=dict(
                        size=percentages/3,
                        color=rgb_colors,
                        opacity=0.7
                    ),
                    name=f'{side.capitalize()} Color {color_num}',
                    text=[f"L: {l:.1f}, a: {a_val:.1f}, b: {b_val:.1f}, %: {p:.1f}" 
                          for l, a_val, b_val, p in zip(L_vals, a_vals, b_vals, percentages)],
                    hoverinfo='text'
                ))
        
        # Update layout
        fig.update_layout(
            title="All Eyebrow Colors in LAB Space",
            scene=dict(
                xaxis_title='L',
                yaxis_title='a',
                zaxis_title='b'
            ),
            width=800,
            height=600
        )
        
        return fig
