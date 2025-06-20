import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.cluster import KMeans
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
import numpy as np
import alphashape

def lab_to_rgb_using_colormath(l, a, b):
    """Converts LAB to RGB using the colormath library, with error handling."""
    try:
        lab_color = LabColor(lab_l=l, lab_a=a, lab_b=b)
        rgb_color = convert_color(lab_color, sRGBColor)
        r = int(max(0, min(255, rgb_color.rgb_r * 255))) # type: ignore
        g = int(max(0, min(255, rgb_color.rgb_g * 255))) # type: ignore
        b = int(max(0, min(255, rgb_color.rgb_b * 255))) # type: ignore
        return [r, g, b]
    except Exception:
        return [128, 128, 128]

def render():
    """
    Renders the advanced dataset analysis tab.
    """
    st.header("📊 Advanced Eyebrow Color Demographics Analysis")
    
    st.markdown("""
    **Comprehensive analysis of eyebrow colors across demographics** featuring:
    - 🎨 Interactive 3D LAB color space visualization with demographic filtering
    - 📈 Method performance analysis by ethnicity, age, and skin tone
    - 📊 LAB value distributions across demographics
    - 🔍 Advanced clustering insights by population groups
    - 🌈 Color pattern discovery across different demographics
    """)
    
    csv_file_path = st.text_input(
        "CSV file path:", 
        value="data/color_analysis_data/color_and_metadata.csv",  
        help="Path to your batch analysis CSV file (relative to script location)",
        placeholder="e.g., debug_and_batch/batch_analysis_results.csv"
    )
    
    if csv_file_path and os.path.exists(csv_file_path):
        st.info(f"📄 Loading: {csv_file_path}")
        
        try:
            df = pd.read_csv(csv_file_path)
            st.success(f"✅ Loaded {len(df)} images from {os.path.basename(csv_file_path)}")
            
            demographic_cols = ['ETHNI_USR', 'eval_cluster', 'SCF1_MOY', 'RESP_FINAL']
            available_demo_cols = [col for col in demographic_cols if col in df.columns]
            
            if not available_demo_cols:
                st.warning("⚠️ No demographic columns found. Analysis will be limited.")
            else:
                st.info(f"📋 Available demographic data: {', '.join(available_demo_cols)}")
            
            color_data = []
            
            for _, row in df.iterrows():
                image_name = row['image_filename']
                
                demo_info = {}
                if 'ETHNI_USR' in df.columns:
                    demo_info['ETHNI_USR'] = row['ETHNI_USR'] if pd.notna(row['ETHNI_USR']) else 'Unknown'
                if 'eval_cluster' in df.columns:
                    demo_info['eval_cluster'] = row['eval_cluster'] if pd.notna(row['eval_cluster']) else 'Unknown'
                if 'SCF1_MOY' in df.columns:
                    age = row['SCF1_MOY'] if pd.notna(row['SCF1_MOY']) else None
                    if age is not None:
                        demo_info['SCF1_MOY'] = age
                    else:
                        demo_info['SCF1_MOY'] = None
                
                for side in ['left', 'right']:
                    for method_rank in range(1, 4):
                        method_col = f'{side}_top_method{method_rank}'
                        if method_col in df.columns and pd.notna(row[method_col]):
                            for color_idx in range(1, 3):
                                l_col = f'{side}_top_method{method_rank}_color{color_idx}_L'
                                a_col = f'{side}_top_method{method_rank}_color{color_idx}_a'
                                b_col = f'{side}_top_method{method_rank}_color{color_idx}_b'
                                pct_col = f'{side}_top_method{method_rank}_color{color_idx}_percentage'
                                quality_col = f'{side}_top_method{method_rank}_quality_score'
                                
                                if all(col in df.columns for col in [l_col, a_col, b_col, pct_col, quality_col]):
                                    if pd.notna(row[l_col]):
                                        color_entry = {
                                            'image': image_name, 'side': side, 'method': row[method_col],
                                            'method_rank': method_rank, 'color_idx': color_idx,
                                            'L': row[l_col], 'a': row[a_col], 'b': row[b_col],
                                            'percentage': row[pct_col], 'quality_score': row[quality_col]
                                        }
                                        color_entry.update(demo_info)
                                        color_data.append(color_entry)
            
            if color_data:
                color_df = pd.DataFrame(color_data)
                st.success(f"✅ Extracted {len(color_df)} color data points with demographics")
                
                overview_tab, method_perf_tab, lab_dist_tab, clustering_tab, demographics_tab = st.tabs([
                    "🔍 Overview", "🏆 Method Performance", "📊 LAB Distributions", 
                    "🎯 Clustering Analysis", "👥 Demographics Deep Dive"
                ])
                
                with overview_tab:
                    st.subheader("Dataset Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("Total Images", df['image_filename'].nunique())
                    with col2: st.metric("Color Points", len(color_df))
                    with col3:
                        if 'ETHNI_USR' in color_df.columns: st.metric("Ethnicities", color_df['ETHNI_USR'].nunique())
                    with col4:
                        if 'SCF1_MOY' in color_df.columns: st.metric("Age Groups", pd.qcut(color_df['SCF1_MOY'], q=5).nunique())
                    
                    if 'ETHNI_USR' in color_df.columns or 'eval_cluster' in color_df.columns:
                        st.subheader("Demographic Distribution")
                        demo_cols = st.columns(3)
                        if 'ETHNI_USR' in color_df.columns:
                            with demo_cols[0]:
                                ethnicity_counts = color_df.groupby('image')['ETHNI_USR'].first().value_counts()
                                fig_eth = px.pie(values=ethnicity_counts.values, names=ethnicity_counts.index, title="Distribution by Ethnicity")
                                st.plotly_chart(fig_eth, use_container_width=True)
                        if 'eval_cluster' in color_df.columns:
                            with demo_cols[1]:
                                skin_counts = color_df.groupby('image')['eval_cluster'].first().value_counts()
                                fig_skin = px.pie(values=skin_counts.values, names=skin_counts.index, title="Distribution by Skin Tone")
                                st.plotly_chart(fig_skin, use_container_width=True)
                        if 'SCF1_MOY' in color_df.columns:
                            with demo_cols[2]:
                                age_bins = pd.qcut(color_df.groupby('image')['SCF1_MOY'].first(), q=5)
                                age_counts = age_bins.value_counts()
                                fig_age = px.pie(values=age_counts.values, names=age_counts.index.astype(str), title="Distribution by Age Group")
                                st.plotly_chart(fig_age, use_container_width=True)
                
                with method_perf_tab:
                    st.subheader("🏆 Method Performance Analysis")
                    st.markdown("""
                    Analyze how different detection methods perform across demographics. 
                    Select demographics to compare and use the interactive legend to show/hide specific groups.
                    Quality scores range from 0-100, combining pixel count, darkness, and spatial coherence.
                    """)
                    
                    # Add demographic filters
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_skin_tones = st.multiselect(
                            "Select Skin Tones",
                            options=sorted(color_df['eval_cluster'].unique()),
                            default=sorted(color_df['eval_cluster'].unique())
                        )
                    with col2:
                        age_range = st.slider(
                            "Age Range",
                            min_value=int(color_df['SCF1_MOY'].min()),
                            max_value=int(color_df['SCF1_MOY'].max()),
                            value=(25, 55)
                        )
                    
                    # Filter data based on selections
                    filtered_df = color_df[
                        (color_df['eval_cluster'].isin(selected_skin_tones)) &
                        (color_df['SCF1_MOY'].between(age_range[0], age_range[1]))
                    ]
                    
                    # Create interactive performance plots
                    if not filtered_df.empty:
                        # Method performance by skin tone
                        st.subheader("📊 Method Quality Scores by Skin Tone")
                        method_quality = filtered_df.groupby(['eval_cluster', 'method', 'side'])['quality_score'].agg(['mean', 'count']).reset_index()
                        method_quality.columns = ['skin_tone', 'method', 'side', 'avg_quality', 'count']
                        
                        # Create interactive line plot
                        fig = go.Figure()
                        for tone in selected_skin_tones:
                            for side in ['left', 'right']:
                                tone_data = method_quality[
                                    (method_quality['skin_tone'] == tone) &
                                    (method_quality['side'] == side)
                                ]
                                fig.add_trace(go.Scatter(
                                    x=tone_data['method'],
                                    y=tone_data['avg_quality'],
                                    mode='lines+markers',
                                    name=f'Skin Tone {tone} - {side}',
                                    hovertemplate='Method: %{x}<br>Quality: %{y:.2f}<br>Count: %{text}<extra></extra>',
                                    text=tone_data['count']
                                ))
                        
                        fig.update_layout(
                            title="Method Quality by Skin Tone and Side",
                            xaxis_title="Method",
                            yaxis_title="Average Quality Score",
                            showlegend=True
                        )
                        st.plotly_chart(fig, key="method_quality_plot", use_container_width=True)
                        
                        # Add detailed statistics table
                        st.markdown("### 📋 Detailed Performance Statistics")
                        stats_df = method_quality.pivot_table(
                            index=['method'],
                            columns=['skin_tone', 'side'],
                            values='avg_quality',
                            aggfunc='mean'
                        ).round(2)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Best methods analysis
                        st.markdown("### 🏆 Best Performing Methods")
                        best_methods = method_quality.loc[method_quality.groupby(['skin_tone', 'side'])['avg_quality'].idxmax()]
                        best_methods = best_methods.sort_values(['skin_tone', 'side'])
                        st.dataframe(best_methods[['skin_tone', 'side', 'method', 'avg_quality', 'count']], use_container_width=True)
                        
                        # Age correlation analysis
                        st.subheader("📈 Method Performance vs Age")
                        age_corr = filtered_df.groupby('method').apply(
                            lambda x: x['quality_score'].corr(x['SCF1_MOY'])
                        ).reset_index()
                        age_corr.columns = ['method', 'age_correlation']
                        
                        fig_age = px.bar(
                            age_corr,
                            x='method',
                            y='age_correlation',
                            title="Correlation between Method Performance and Age",
                            labels={'age_correlation': 'Correlation Coefficient'}
                        )
                        st.plotly_chart(fig_age, key="age_correlation_plot", use_container_width=True)
                        
                        # Age group analysis
                        st.subheader("Method Quality by Age Group")
                        filtered_df['age_group'] = pd.qcut(filtered_df['SCF1_MOY'], q=5, labels=['18-29', '30-39', '40-49', '50-59', '60+'])
                        method_quality_age = filtered_df.groupby(['age_group', 'method', 'side'])['quality_score'].agg(['mean', 'count']).reset_index()
                        method_quality_age.columns = ['age_group', 'method', 'side', 'avg_quality', 'count']
                        fig_qual_age = px.bar(method_quality_age, x='method', y='avg_quality', color='age_group', facet_col='side', title="Average Quality Score by Method, Age Group, and Side")
                        st.plotly_chart(fig_qual_age, key="age_group_quality_plot", use_container_width=True)
                    else:
                        st.warning("No data available for the selected filters. Please adjust your selection.")

                with lab_dist_tab:
                    st.subheader("🎨 LAB Color Analysis")
                    st.markdown("""
                    Analysis of LAB color values from the best performing detection method for each image (highest quality score).
                    LAB color space represents:
                    - **L**: Lightness (0 = black, 100 = white)
                    - **a**: Green (-) to Red (+)
                    - **b**: Blue (-) to Yellow (+)
                    """)
                    
                    # Add demographic filters
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_skin_tones = st.multiselect(
                            "Filter by Skin Tone",
                            options=sorted(color_df['eval_cluster'].unique()),
                            default=sorted(color_df['eval_cluster'].unique())
                        )
                    with col2:
                        age_range = st.slider(
                            "Filter by Age",
                            min_value=int(color_df['SCF1_MOY'].min()),
                            max_value=int(color_df['SCF1_MOY'].max()),
                            value=(25, 55)
                        )
                    
                    # Filter data
                    filtered_df = color_df[
                        (color_df['eval_cluster'].isin(selected_skin_tones)) &
                        (color_df['SCF1_MOY'].between(age_range[0], age_range[1]))
                    ]
                    
                    if not filtered_df.empty:
                        # Get best method results only
                        best_methods_df = filtered_df.loc[filtered_df.groupby(['image', 'side'])['quality_score'].idxmax()]
                        
                        # 3D LAB scatter plots (separate for left and right)
                        st.subheader("📈 3D LAB Color Space Distribution")
                        
                        # Create two columns for side-by-side plots
                        col1, col2 = st.columns(2)
                        
                        for side, col in zip(['left', 'right'], [col1, col2]):
                            with col:
                                fig_3d = go.Figure()
                                
                                for tone in selected_skin_tones:
                                    tone_data = best_methods_df[
                                        (best_methods_df['eval_cluster'] == tone) &
                                        (best_methods_df['side'] == side)
                                    ]
                                    
                                    fig_3d.add_trace(go.Scatter3d(
                                        x=tone_data['L'],
                                        y=tone_data['a'],
                                        z=tone_data['b'],
                                        mode='markers',
                                        name=f"Skin Tone {tone}",
                                        marker=dict(
                                            size=5,
                                            opacity=0.7
                                        ),
                                        text=[f"Method: {m}<br>Quality: {q:.1f}" 
                                              for m, q in zip(tone_data['method'], tone_data['quality_score'])]
                                    ))
                                
                                fig_3d.update_layout(
                                    scene=dict(
                                        xaxis_title='L* (Lightness)',
                                        yaxis_title='a* (Green-Red)',
                                        zaxis_title='b* (Blue-Yellow)'
                                    ),
                                    title=f"LAB Color Space Distribution - {side.capitalize()} Eyebrow",
                                    legend_title="Skin Tone"
                                )
                                
                                st.plotly_chart(fig_3d, key=f"lab_3d_plot_{side}", use_container_width=True)
                        
                        # LAB value distributions
                        st.subheader("📈 LAB Value Distributions")
                        
                        # L* values by skin tone
                        fig_l = px.box(filtered_df, x='eval_cluster', y='L', color='side',
                                      title="L* Values by Skin Tone and Side",
                                      labels={'L': 'L* Value', 'eval_cluster': 'Skin Tone Cluster'})
                        st.plotly_chart(fig_l, key="lab_l_plot", use_container_width=True)
                        
                        # a* values by skin tone
                        fig_a = px.box(filtered_df, x='eval_cluster', y='a', color='side',
                                      title="a* Values by Skin Tone and Side",
                                      labels={'a': 'a* Value', 'eval_cluster': 'Skin Tone Cluster'})
                        st.plotly_chart(fig_a, key="lab_a_plot", use_container_width=True)
                        
                        # b* values by skin tone
                        fig_b = px.box(filtered_df, x='eval_cluster', y='b', color='side',
                                      title="b* Values by Skin Tone and Side",
                                      labels={'b': 'b* Value', 'eval_cluster': 'Skin Tone Cluster'})
                        st.plotly_chart(fig_b, key="lab_b_plot", use_container_width=True)
                        
                        # Summary statistics
                        st.subheader("📃 LAB Value Statistics")
                        lab_stats = filtered_df.groupby(['eval_cluster', 'side'])[['L', 'a', 'b']].agg(['mean', 'std']).round(2)
                        st.dataframe(lab_stats, use_container_width=True)
                        
                        # Method distribution
                        st.markdown("### 🏆 Best Performing Methods")
                        method_dist = best_methods_df['method'].value_counts()
                        fig_methods = px.pie(
                            values=method_dist.values,
                            names=method_dist.index,
                            title="Distribution of Best Performing Methods"
                        )
                        st.plotly_chart(fig_methods, key="method_dist_plot", use_container_width=True)
                        st.plotly_chart(fig_methods, use_container_width=True)
                    else:
                        st.warning("No data available for the selected filters. Please adjust your selection.")


                with clustering_tab:
                    st.subheader("🎯 Advanced Clustering Analysis")
                    st.markdown("""
                    **Clustering analysis using only the best performing method per image/side combination.**
                    - Uses the method with the highest quality score for each eyebrow
                    - Shows enhanced 3D visualizations with multiple boundary options
                    - Displays actual eyebrow colors (LAB→RGB conversion)
                    """)
                    
                    # Enhanced control panel with 5 columns
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1: 
                        n_clusters = st.slider("Number of clusters", 2, 15, 8)
                    with col2: 
                        side_filter = st.selectbox("Eyebrow side:", ['Both', 'left', 'right'])
                    with col3:
                        selected_skin_tones = st.multiselect(
                            "Filter by Skin Tone:",
                            options=sorted(color_df['eval_cluster'].unique()),
                            default=sorted(color_df['eval_cluster'].unique())
                        )
                    with col4:
                        color_mode = st.selectbox(
                            "Color Mode:",
                            options=['Real Colors', 'Cluster Colors'],
                            help="Real Colors: Actual eyebrow colors\nCluster Colors: Distinct colors per cluster"
                        )
                    with col5:
                        viz_enhancement = st.selectbox(
                            "3D Enhancement:",
                            options=['Convex Hulls', 'Confidence Ellipsoids', 'Alpha Shapes', 'Density Contours', 'None'],
                            help="Different ways to visualize cluster boundaries"
                        )
                    
                    # Get only the best method per image/side combination
                    best_methods_df = color_df.loc[color_df.groupby(['image', 'side'])['quality_score'].idxmax()].copy()
                    
                    # Apply filters
                    filtered_data = best_methods_df.copy()
                    if side_filter != 'Both':
                        filtered_data = filtered_data[filtered_data['side'] == side_filter]
                    filtered_data = filtered_data[filtered_data['eval_cluster'].isin(selected_skin_tones)]
                    
                    # Display data info
                    total_images = df['image_filename'].nunique()
                    expected_points = total_images * (2 if side_filter == 'Both' else 1)
                    st.info(f"📊 Using {len(filtered_data)} data points from {len(filtered_data['image'].unique())} images (Expected ~{expected_points} for best methods only)")
                    
                    if len(filtered_data) > 0:
                        # Perform clustering
                        color_features = filtered_data[['L', 'a', 'b']].values
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(color_features)
                        
                        # Create visualization dataframe
                        filtered_data_viz = filtered_data.copy()
                        filtered_data_viz['cluster'] = clusters.astype(int)
                        
                        # Convert LAB to RGB for real colors
                        filtered_data_viz['real_rgb'] = filtered_data_viz.apply(
                            lambda row: lab_to_rgb_using_colormath(row['L'], row['a'], row['b']), axis=1
                        )
                        filtered_data_viz['real_rgb_str'] = filtered_data_viz['real_rgb'].apply(
                            lambda rgb: f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'
                        )
                        
                        # Enhanced 3D visualization with multiple boundary options
                        fig_cluster = go.Figure()
                        
                        # Check scipy availability
                        try:
                            from scipy.spatial import ConvexHull
                            from scipy import stats
                            import numpy as np
                            scipy_available = True
                            st.success("✅ Enhanced 3D visualization available")
                        except ImportError:
                            scipy_available = False
                            st.warning("⚠️ Install scipy for enhanced visualizations")
                        
                        # Cluster colors (more vibrant for better visibility)
                        cluster_colors = [
                            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                            '#DDA0DD', '#98D8C8', '#F7DC6F', '#85C1E9', '#F8C471',
                            '#82E0AA', '#F1948A', '#AED6F1', '#A9DFBF', '#FAD7A0'
                        ]
                        
                        # Add cluster boundaries based on selected enhancement
                        if scipy_available and viz_enhancement != 'None':
                            for cluster_id in sorted(filtered_data_viz['cluster'].unique()):
                                cluster_data = filtered_data_viz[filtered_data_viz['cluster'] == cluster_id]
                                if len(cluster_data) >= 4:
                                    points = cluster_data[['L', 'a', 'b']].values
                                    cluster_color = cluster_colors[cluster_id % len(cluster_colors)]
                                    
                                    try:
                                        if viz_enhancement == 'Convex Hulls':
                                            # Enhanced convex hulls (these should always capture all points)
                                            hull = ConvexHull(points)
                                            fig_cluster.add_trace(go.Mesh3d(
                                                x=points[hull.vertices, 0],
                                                y=points[hull.vertices, 1], 
                                                z=points[hull.vertices, 2],
                                                i=hull.simplices[:, 0],
                                                j=hull.simplices[:, 1],
                                                k=hull.simplices[:, 2],
                                                opacity=0.25,
                                                color=cluster_color,
                                                name=f'Hull {cluster_id}',
                                                showlegend=False,
                                                hoverinfo='skip',
                                                alphahull=0,
                                                lighting=dict(ambient=0.4, diffuse=0.8, fresnel=0.1, specular=1, roughness=0.05),
                                                lightposition=dict(x=100, y=200, z=0)
                                            ))
                                        
                                        elif viz_enhancement == 'Confidence Ellipsoids':
                                            # More inclusive 3D confidence ellipsoids (3-sigma instead of 2-sigma)
                                            mean = np.mean(points, axis=0)
                                            cov = np.cov(points.T)
                                            
                                            # Use 3-sigma for better coverage
                                            try:
                                                eigenvals, eigenvecs = np.linalg.eigh(cov)
                                                radii = 3 * np.sqrt(eigenvals)  # 3-sigma confidence (99.7% coverage)
                                                
                                                # Create more detailed ellipsoid surface
                                                u = np.linspace(0, 2 * np.pi, 30)
                                                v = np.linspace(0, np.pi, 30)
                                                
                                                # Create ellipsoid surface
                                                x_ellipse = radii[0] * np.outer(np.cos(u), np.sin(v))
                                                y_ellipse = radii[1] * np.outer(np.sin(u), np.sin(v))
                                                z_ellipse = radii[2] * np.outer(np.ones(np.size(u)), np.cos(v))
                                                
                                                # Rotate and translate
                                                ellipsoid_points = np.array([x_ellipse.flatten(), y_ellipse.flatten(), z_ellipse.flatten()])
                                                rotated_points = eigenvecs @ ellipsoid_points
                                                final_points = rotated_points + mean.reshape(-1, 1)
                                                
                                                # Create triangulation for proper surface
                                                from scipy.spatial import SphericalVoronoi
                                                # Create spherical mesh
                                                sphere_points = np.array([[np.cos(u_val) * np.sin(v_val), 
                                                                        np.sin(u_val) * np.sin(v_val), 
                                                                        np.cos(v_val)] for u_val in u for v_val in v])
                                                
                                                # Transform to ellipsoid
                                                ellipsoid_surface = (eigenvecs @ (radii[:, np.newaxis] * sphere_points.T) + mean.reshape(-1, 1)).T
                                                
                                                # Use ConvexHull to create mesh
                                                hull_ellipsoid = ConvexHull(ellipsoid_surface)
                                                
                                                fig_cluster.add_trace(go.Mesh3d(
                                                    x=ellipsoid_surface[hull_ellipsoid.vertices, 0],
                                                    y=ellipsoid_surface[hull_ellipsoid.vertices, 1],
                                                    z=ellipsoid_surface[hull_ellipsoid.vertices, 2],
                                                    i=hull_ellipsoid.simplices[:, 0],
                                                    j=hull_ellipsoid.simplices[:, 1],
                                                    k=hull_ellipsoid.simplices[:, 2],
                                                    opacity=0.2,
                                                    color=cluster_color,
                                                    name=f'Ellipsoid {cluster_id}',
                                                    showlegend=False,
                                                    hoverinfo='skip'
                                                ))
                                            except np.linalg.LinAlgError:
                                                st.warning(f"Could not compute ellipsoid for cluster {cluster_id} - using bounding box")
                                                # Fallback to bounding box
                                                min_vals = points.min(axis=0)
                                                max_vals = points.max(axis=0)
                                                # Create bounding box corners
                                                box_corners = np.array([
                                                    [min_vals[0], min_vals[1], min_vals[2]],
                                                    [max_vals[0], min_vals[1], min_vals[2]],
                                                    [max_vals[0], max_vals[1], min_vals[2]],
                                                    [min_vals[0], max_vals[1], min_vals[2]],
                                                    [min_vals[0], min_vals[1], max_vals[2]],
                                                    [max_vals[0], min_vals[1], max_vals[2]],
                                                    [max_vals[0], max_vals[1], max_vals[2]],
                                                    [min_vals[0], max_vals[1], max_vals[2]]
                                                ])
                                                box_hull = ConvexHull(box_corners)
                                                fig_cluster.add_trace(go.Mesh3d(
                                                    x=box_corners[box_hull.vertices, 0],
                                                    y=box_corners[box_hull.vertices, 1],
                                                    z=box_corners[box_hull.vertices, 2],
                                                    i=box_hull.simplices[:, 0],
                                                    j=box_hull.simplices[:, 1],
                                                    k=box_hull.simplices[:, 2],
                                                    opacity=0.15,
                                                    color=cluster_color,
                                                    name=f'BoundingBox {cluster_id}',
                                                    showlegend=False,
                                                    hoverinfo='skip'
                                                ))
                                        
                                        elif viz_enhancement == 'Alpha Shapes':
                                            # More inclusive alpha shapes with multiple alpha values
                                            try:
                                                from alphashape import alphashape
                                                
                                                # Try different alpha values to find one that captures all points
                                                alpha_values = [0.1, 0.2, 0.5, 1.0, 2.0]
                                                alpha_shape = None
                                                
                                                for alpha_val in alpha_values:
                                                    try:
                                                        alpha_shape = alphashape(points, alpha_val)
                                                        if alpha_shape is not None and hasattr(alpha_shape, 'vertices'):
                                                            break
                                                    except:
                                                        continue
                                                
                                                # If alpha shape failed, fallback to convex hull
                                                if alpha_shape is None or not hasattr(alpha_shape, 'vertices'):
                                                    st.info(f"Alpha shape failed for cluster {cluster_id}, using convex hull")
                                                    hull = ConvexHull(points)
                                                    fig_cluster.add_trace(go.Mesh3d(
                                                        x=points[hull.vertices, 0],
                                                        y=points[hull.vertices, 1], 
                                                        z=points[hull.vertices, 2],
                                                        i=hull.simplices[:, 0],
                                                        j=hull.simplices[:, 1],
                                                        k=hull.simplices[:, 2],
                                                        opacity=0.3,
                                                        color=cluster_color,
                                                        name=f'Alpha-Hull {cluster_id}',
                                                        showlegend=False,
                                                        hoverinfo='skip'
                                                    ))
                                                else:
                                                    # Extract mesh data
                                                    vertices = np.array(alpha_shape.vertices)
                                                    faces = np.array(alpha_shape.faces)
                                                    
                                                    fig_cluster.add_trace(go.Mesh3d(
                                                        x=vertices[:, 0],
                                                        y=vertices[:, 1],
                                                        z=vertices[:, 2],
                                                        i=faces[:, 0],
                                                        j=faces[:, 1],
                                                        k=faces[:, 2],
                                                        opacity=0.3,
                                                        color=cluster_color,
                                                        name=f'Alpha Shape {cluster_id}',
                                                        showlegend=False,
                                                        hoverinfo='skip'
                                                    ))
                                                    
                                            except ImportError:
                                                st.warning("Install alphashape package: pip install alphashape")
                                                # Fallback to convex hull
                                                hull = ConvexHull(points)
                                                fig_cluster.add_trace(go.Mesh3d(
                                                    x=points[hull.vertices, 0],
                                                    y=points[hull.vertices, 1], 
                                                    z=points[hull.vertices, 2],
                                                    i=hull.simplices[:, 0],
                                                    j=hull.simplices[:, 1],
                                                    k=hull.simplices[:, 2],
                                                    opacity=0.3,
                                                    color=cluster_color,
                                                    name=f'Fallback Hull {cluster_id}',
                                                    showlegend=False,
                                                    hoverinfo='skip'
                                                ))
                                        
                                        elif viz_enhancement == 'Density Contours':
                                            # More inclusive density estimation with larger margins
                                            from scipy.stats import gaussian_kde
                                            
                                            # Create a density estimator
                                            kde = gaussian_kde(points.T)
                                            
                                            # Create a larger grid around ALL points
                                            margin = np.std(points, axis=0) * 2  # Adaptive margin based on std
                                            x_min, x_max = points[:, 0].min() - margin[0], points[:, 0].max() + margin[0]
                                            y_min, y_max = points[:, 1].min() - margin[1], points[:, 1].max() + margin[1]
                                            z_min, z_max = points[:, 2].min() - margin[2], points[:, 2].max() + margin[2]
                                            
                                            # Create higher resolution grid
                                            x_grid = np.linspace(x_min, x_max, 20)
                                            y_grid = np.linspace(y_min, y_max, 20)
                                            z_grid = np.linspace(z_min, z_max, 20)
                                            
                                            xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
                                            grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
                                            
                                            # Calculate density
                                            density = kde(grid_points).reshape(xx.shape)
                                            
                                            # Use lower threshold to capture more points
                                            min_threshold = density.max() * 0.05  # Very low threshold
                                            max_threshold = density.max() * 0.6
                                            
                                            # Add isosurface
                                            fig_cluster.add_trace(go.Isosurface(
                                                x=xx.flatten(),
                                                y=yy.flatten(),
                                                z=zz.flatten(),
                                                value=density.flatten(),
                                                isomin=min_threshold,
                                                isomax=max_threshold,
                                                opacity=0.2,
                                                colorscale=[[0, cluster_color], [1, cluster_color]],
                                                showscale=False,
                                                name=f'Density {cluster_id}',
                                                showlegend=False,
                                                hoverinfo='skip',
                                                caps=dict(x_show=False, y_show=False, z_show=False)
                                            ))
                                    
                                    except Exception as e:
                                        st.warning(f"Could not create {viz_enhancement.lower()} for cluster {cluster_id}: {str(e)}")
                                        # Always fallback to convex hull as it's guaranteed to work
                                        try:
                                            hull = ConvexHull(points)
                                            fig_cluster.add_trace(go.Mesh3d(
                                                x=points[hull.vertices, 0],
                                                y=points[hull.vertices, 1], 
                                                z=points[hull.vertices, 2],
                                                i=hull.simplices[:, 0],
                                                j=hull.simplices[:, 1],
                                                k=hull.simplices[:, 2],
                                                opacity=0.25,
                                                color=cluster_color,
                                                name=f'Fallback Hull {cluster_id}',
                                                showlegend=False,
                                                hoverinfo='skip'
                                            ))
                                        except Exception as fallback_error:
                                            st.error(f"Even fallback failed for cluster {cluster_id}: {str(fallback_error)}")
                                else:
                                    st.info(f"Cluster {cluster_id} has only {len(cluster_data)} points - need at least 4 for 3D boundaries")
                        
                        # Add scatter points
                        for cluster_id in sorted(filtered_data_viz['cluster'].unique()):
                            cluster_data = filtered_data_viz[filtered_data_viz['cluster'] == cluster_id]
                            
                            # Choose color mode
                            if color_mode == 'Real Colors':
                                point_colors = cluster_data['real_rgb_str'].tolist()
                                marker_color = point_colors
                            else:
                                cluster_color = cluster_colors[cluster_id % len(cluster_colors)]
                                marker_color = cluster_color
                            
                            # Enhanced scatter points
                            fig_cluster.add_trace(go.Scatter3d(
                                x=cluster_data['L'], 
                                y=cluster_data['a'], 
                                z=cluster_data['b'],
                                mode='markers',
                                marker=dict(
                                    size=8, 
                                    color=marker_color,
                                    opacity=0.9,
                                    line=dict(width=2, color='white'),
                                    sizemode='diameter'
                                ),
                                name=f'Cluster {cluster_id} ({len(cluster_data)} points)',
                                legendgroup=f'cluster_{cluster_id}',
                                text=cluster_data.apply(lambda row: 
                                    f"<b>Cluster {row['cluster']}</b><br>" +
                                    f"Image: {row['image']}<br>" +
                                    f"Side: {row['side']}<br>" +
                                    f"Method: {row['method']}<br>" +
                                    f"Quality: {row['quality_score']:.1f}<br>" +
                                    f"LAB: L*={row['L']:.1f}, a*={row['a']:.1f}, b*={row['b']:.1f}<br>" +
                                    f"RGB: {row['real_rgb']}<br>" +
                                    f"Skin Tone: {row['eval_cluster']}", axis=1),
                                hovertemplate='%{text}<extra></extra>'
                            ))
                        
                        # Enhanced layout
                        fig_cluster.update_layout(
                            title=f'Enhanced 3D Eyebrow Color Clustering ({viz_enhancement})',
                            scene=dict(
                                xaxis_title='L* (Lightness)', 
                                yaxis_title='a* (Green-Red)', 
                                zaxis_title='b* (Blue-Yellow)',
                                camera=dict(eye=dict(x=1.8, y=1.8, z=1.8)),
                                bgcolor='rgba(240,240,240,0.1)',
                                xaxis=dict(
                                    backgroundcolor="rgba(0, 0, 0,0)",
                                    gridcolor="lightgray",
                                    showbackground=True,
                                    zerolinecolor="gray",
                                    title_font_size=14
                                ),
                                yaxis=dict(
                                    backgroundcolor="rgba(0, 0, 0,0)",
                                    gridcolor="lightgray",
                                    showbackground=True,
                                    zerolinecolor="gray",
                                    title_font_size=14
                                ),
                                zaxis=dict(
                                    backgroundcolor="rgba(0, 0, 0,0)",
                                    gridcolor="lightgray",
                                    showbackground=True,
                                    zerolinecolor="gray",
                                    title_font_size=14
                                )
                            ), 
                            height=750,
                            showlegend=True,
                            legend=dict(
                                x=1.02, y=1, 
                                traceorder="normal",
                                font=dict(size=11),
                                bgcolor="rgba(255,255,255,0.9)",
                                bordercolor="gray",
                                borderwidth=1
                            )
                        )
                        
                        st.plotly_chart(fig_cluster, use_container_width=True)
                        
                        # Add explanation of the selected enhancement
                        enhancement_explanations = {
                            'Convex Hulls': "🔷 **Convex Hulls**: Show the smallest convex shape containing all points in each cluster",
                            'Confidence Ellipsoids': "🥚 **Confidence Ellipsoids**: Show 3-sigma confidence regions (99.7% coverage) based on cluster covariance",
                            'Alpha Shapes': "🌊 **Alpha Shapes**: More flexible boundaries that can capture non-convex cluster shapes (tries multiple alpha values)",
                            'Density Contours': "📊 **Density Contours**: Show probability density surfaces around clusters with adaptive margins",
                            'None': "⚪ **Basic View**: Simple scatter plot without cluster boundaries"
                        }
                        
                        if viz_enhancement in enhancement_explanations:
                            st.info(enhancement_explanations[viz_enhancement])
                        
                        # Cluster analysis table
                        st.subheader("📊 Cluster Demographics Analysis")
                        cluster_demo_analysis = []
                        for cluster_id in sorted(filtered_data_viz['cluster'].unique()):
                            cluster_data = filtered_data_viz[filtered_data_viz['cluster'] == cluster_id]
                            
                            # Calculate average RGB color for cluster
                            avg_rgb = [
                                int(np.mean([rgb[0] for rgb in cluster_data['real_rgb']])),
                                int(np.mean([rgb[1] for rgb in cluster_data['real_rgb']])),
                                int(np.mean([rgb[2] for rgb in cluster_data['real_rgb']]))
                            ]
                            
                            analysis = {
                                'Cluster': int(cluster_id),
                                'Count': int(len(cluster_data)),
                                'Images': int(len(cluster_data['image'].unique())),
                                'Avg RGB': f"({avg_rgb[0]}, {avg_rgb[1]}, {avg_rgb[2]})"
                            }
                            
                            # Demographics
                            if 'eval_cluster' in cluster_data.columns:
                                top_skin = cluster_data['eval_cluster'].mode().iloc[0] if not cluster_data['eval_cluster'].mode().empty else 'N/A'
                                analysis['Top Skin Tone'] = f"{top_skin} ({(cluster_data['eval_cluster'] == top_skin).sum()}/{len(cluster_data)})"
                            
                            # Method analysis
                            top_method = cluster_data['method'].mode().iloc[0] if not cluster_data['method'].mode().empty else 'N/A'
                            analysis['Top Method'] = f"{top_method} ({(cluster_data['method'] == top_method).sum()}/{len(cluster_data)})"
                            
                            # Side distribution
                            left_count = (cluster_data['side'] == 'left').sum()
                            right_count = (cluster_data['side'] == 'right').sum()
                            analysis['L/R Distribution'] = f"{left_count}/{right_count}"
                            
                            # LAB averages
                            analysis.update({
                                'Avg L*': round(cluster_data['L'].mean(), 1),
                                'Avg a*': round(cluster_data['a'].mean(), 1), 
                                'Avg b*': round(cluster_data['b'].mean(), 1),
                                'Avg Quality': round(cluster_data['quality_score'].mean(), 1)
                            })
                            
                            cluster_demo_analysis.append(analysis)
                        
                        cluster_demo_df = pd.DataFrame(cluster_demo_analysis)
                        st.dataframe(cluster_demo_df, use_container_width=True)
                        
                        # Method distribution in clusters
                        st.subheader("🏆 Best Methods Distribution in Clusters")
                        method_cluster_crosstab = pd.crosstab(
                            filtered_data_viz['cluster'], 
                            filtered_data_viz['method'],
                            margins=False
                        )
                        st.dataframe(method_cluster_crosstab, use_container_width=True)
                        
                        # Add totals manually
                        st.write("**Totals:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**By Cluster:**")
                            cluster_totals = filtered_data_viz['cluster'].value_counts().sort_index()
                            st.dataframe(cluster_totals.to_frame('Count'), use_container_width=True)
                        with col2:
                            st.write("**By Method:**")
                            method_totals = filtered_data_viz['method'].value_counts()
                            st.dataframe(method_totals.to_frame('Count'), use_container_width=True)
                        
                        # Enhanced color swatches with LAB focus
                        st.subheader("🎨 Cluster Color Swatches (Real Eyebrow Colors)")
                        st.markdown("**LAB Color Space Values** - L* (Lightness), a* (Green-Red), b* (Blue-Yellow)")
                        
                        cols = st.columns(min(5, len(cluster_demo_analysis)))
                        for i, cluster_info in enumerate(cluster_demo_analysis):
                            with cols[i % len(cols)]:
                                cluster_data = filtered_data_viz[filtered_data_viz['cluster'] == cluster_info['Cluster']]
                                
                                # Calculate average LAB values
                                avg_L = cluster_data['L'].mean()
                                avg_a = cluster_data['a'].mean()
                                avg_b = cluster_data['b'].mean()
                                
                                # Calculate LAB ranges
                                L_range = f"{cluster_data['L'].min():.1f}-{cluster_data['L'].max():.1f}"
                                a_range = f"{cluster_data['a'].min():.1f}-{cluster_data['a'].max():.1f}"
                                b_range = f"{cluster_data['b'].min():.1f}-{cluster_data['b'].max():.1f}"
                                
                                # Calculate average RGB for display color only
                                avg_rgb = [
                                    int(np.mean([rgb[0] for rgb in cluster_data['real_rgb']])),
                                    int(np.mean([rgb[1] for rgb in cluster_data['real_rgb']])),
                                    int(np.mean([rgb[2] for rgb in cluster_data['real_rgb']]))
                                ]
                                rgb_str = f"rgb({avg_rgb[0]}, {avg_rgb[1]}, {avg_rgb[2]})"
                                
                                # Calculate text color for contrast
                                brightness = (avg_rgb[0] * 0.299 + avg_rgb[1] * 0.587 + avg_rgb[2] * 0.114)
                                text_color = "white" if brightness < 128 else "black"
                                text_shadow = "2px 2px 4px rgba(0,0,0,0.8)" if brightness >= 128 else "2px 2px 4px rgba(255,255,255,0.8)"
                                
                                # Very clean swatch with just cluster number
                                st.markdown(
                                    f"<div style='background-color: {rgb_str}; width: 100%; height: 60px; "
                                    f"border: 2px solid #333; border-radius: 8px; display: flex; "
                                    f"align-items: center; justify-content: center; color: {text_color}; "
                                    f"font-weight: bold; text-shadow: {text_shadow}; font-size: 1.2em;'>"
                                    f"Cluster {cluster_info['Cluster']}"
                                    f"</div>", 
                                    unsafe_allow_html=True
                                )
                                
                                # All information below the swatch
                                st.markdown(f"**{cluster_info['Images']} images** ({cluster_info['Count']} points)")
                                
                                st.markdown("**📊 Average LAB:**")
                                st.write(f"• **L***: {avg_L:.1f} `({L_range})`")
                                st.write(f"• **a***: {avg_a:.1f} `({a_range})`")
                                st.write(f"• **b***: {avg_b:.1f} `({b_range})`")
                                
                                st.markdown("**👥 Demographics:**")
                                st.caption(f"🎨 {cluster_info['Top Skin Tone']}")
                                st.caption(f"🔬 {cluster_info['Top Method']}")
                                st.caption(f"⚖️ L/R: {cluster_info['L/R Distribution']}")
                                st.caption(f"🎯 Quality: {cluster_info['Avg Quality']}")
                    else:
                        st.warning("❌ No data available for selected filters")

                with demographics_tab:
                    st.subheader("👥 Demographics Deep Dive")
                    if all(col in color_df.columns for col in ['eval_cluster', 'SCF1_MOY']):
                        st.subheader("Cross-Demographics Analysis")
                        
                        # Create age groups for visualization
                        color_df['age_group'] = pd.qcut(color_df['SCF1_MOY'], q=5, labels=['18-29', '30-39', '40-49', '50-59', '60+'])
                        
                        demo_crosstab = pd.crosstab(
                            color_df.groupby('image')['eval_cluster'].first(),
                            color_df.groupby('image')['age_group'].first()
                        )
                        st.write("Image Count by Skin Tone and Age Group:")
                        st.dataframe(demo_crosstab, use_container_width=True)
                        
                        st.subheader("Average L*a*b* Values by Demographics")
                        lab_by_demo = color_df.groupby(['eval_cluster', 'age_group'])[['L', 'a', 'b']].mean().round(1)
                        st.dataframe(lab_by_demo, use_container_width=True)
            else:
                st.error("❌ No valid color data found in the CSV file")
                
        except Exception as e:
            st.error(f"❌ Error loading or processing CSV file: {str(e)}")
    elif csv_file_path:
        st.error(f"❌ File not found at path: {csv_file_path}")
