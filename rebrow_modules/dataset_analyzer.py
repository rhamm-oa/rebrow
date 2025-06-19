import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.cluster import KMeans
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
import numpy as np

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
                    st.subheader("🎥 Advanced Clustering Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1: n_clusters = st.slider("Number of clusters", 2, 15, 8)
                    with col2: side_filter = st.selectbox("Eyebrow side:", ['Both', 'left', 'right'])
                    with col3:
                        selected_skin_tones = st.multiselect(
                            "Filter by Skin Tone:",
                            options=sorted(color_df['eval_cluster'].unique()),
                            default=sorted(color_df['eval_cluster'].unique())
                        )
                    
                    filtered_data = color_df.copy()
                    if side_filter != 'Both':
                        filtered_data = filtered_data[filtered_data['side'] == side_filter]
                    filtered_data = filtered_data[filtered_data['eval_cluster'].isin(selected_skin_tones)]
                    
                    if len(filtered_data) > 0:
                        color_features = filtered_data[['L', 'a', 'b']].values
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(color_features)
                        filtered_data['cluster'] = clusters
                        
                        filtered_data['rgb'] = filtered_data.apply(lambda row: lab_to_rgb_using_colormath(row['L'], row['a'], row['b']), axis=1)
                        filtered_data['rgb_str'] = filtered_data['rgb'].apply(lambda rgb: f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})')
                        
                        fig_cluster = go.Figure()
                        for cluster_id in sorted(filtered_data['cluster'].unique()):
                            cluster_data = filtered_data[filtered_data['cluster'] == cluster_id]
                            fig_cluster.add_trace(go.Scatter3d(
                                x=cluster_data['L'], y=cluster_data['a'], z=cluster_data['b'],
                                mode='markers',
                                marker=dict(size=6, color=cluster_data['rgb_str'], opacity=0.8),
                                name=f'Cluster {cluster_id}',
                                text=cluster_data.apply(lambda row: 
                                    f"Cluster: {row['cluster']}<br>" +
                                    f"Image: {row['image']}<br>" +
                                    f"LAB: ({row['L']:.1f}, {row['a']:.1f}, {row['b']:.1f})<br>" +
                                    (f"Ethnicity: {row['ethnicity']}<br>" if 'ethnicity' in row else "") +
                                    (f"Skin Tone: {row['skin_tone']}<br>" if 'skin_tone' in row else "") +
                                    (f"Age Group: {row['age_group']}<br>" if 'age_group' in row else ""), axis=1),
                                hovertemplate='%{text}<extra></extra>'
                            ))
                        fig_cluster.update_layout(title=f'Eyebrow Color Clustering ({n_clusters} clusters)', scene=dict(xaxis_title='L* (Lightness)', yaxis_title='a* (Green-Red)', zaxis_title='b* (Blue-Yellow)'), height=600)
                        st.plotly_chart(fig_cluster, use_container_width=True)
                        
                        st.subheader("Cluster Demographics Analysis")
                        cluster_demo_analysis = []
                        for cluster_id in sorted(filtered_data['cluster'].unique()):
                            cluster_data = filtered_data[filtered_data['cluster'] == cluster_id]
                            analysis = {'Cluster': cluster_id, 'Count': len(cluster_data)}
                            if 'ethnicity' in cluster_data.columns: top_ethnicity = cluster_data['ethnicity'].mode().iloc[0] if not cluster_data['ethnicity'].mode().empty else 'N/A'; analysis['Top Ethnicity'] = f"{top_ethnicity} ({(cluster_data['ethnicity'] == top_ethnicity).sum()}/{len(cluster_data)})"
                            if 'skin_tone' in cluster_data.columns: top_skin = cluster_data['skin_tone'].mode().iloc[0] if not cluster_data['skin_tone'].mode().empty else 'N/A'; analysis['Top Skin Tone'] = f"{top_skin} ({(cluster_data['skin_tone'] == top_skin).sum()}/{len(cluster_data)})"
                            if 'age_group' in cluster_data.columns: top_age = cluster_data['age_group'].mode().iloc[0] if not cluster_data['age_group'].mode().empty else 'N/A'; analysis['Top Age Group'] = f"{top_age} ({(cluster_data['age_group'] == top_age).sum()}/{len(cluster_data)})"
                            analysis.update({'Avg L': round(cluster_data['L'].mean(), 1), 'Avg a': round(cluster_data['a'].mean(), 1), 'Avg b': round(cluster_data['b'].mean(), 1), 'Avg Quality': round(cluster_data['quality_score'].mean(), 2)})
                            cluster_demo_analysis.append(analysis)
                        cluster_demo_df = pd.DataFrame(cluster_demo_analysis)
                        st.dataframe(cluster_demo_df, use_container_width=True)
                        
                        st.subheader("Cluster Color Swatches with Demographics")
                        cols = st.columns(min(5, len(cluster_demo_analysis)))
                        for i, cluster_info in enumerate(cluster_demo_analysis):
                            with cols[i % len(cols)]:
                                cluster_data = filtered_data[filtered_data['cluster'] == cluster_info['Cluster']]
                                avg_rgb = lab_to_rgb_using_colormath(cluster_data['L'].mean(), cluster_data['a'].mean(), cluster_data['b'].mean())
                                rgb_str = f"rgb({avg_rgb[0]}, {avg_rgb[1]}, {avg_rgb[2]})"
                                st.markdown(f"**Cluster {cluster_info['Cluster']}**")
                                st.markdown(f"<div style='background-color: {rgb_str}; width: 100%; height: 60px; border: 2px solid #333; border-radius: 5px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; text-shadow: 1px 1px 1px rgba(0,0,0,0.8);'>{cluster_info['Count']} points</div>", unsafe_allow_html=True)
                                if 'Top Ethnicity' in cluster_info: st.caption(f"👥 {cluster_info['Top Ethnicity']}")
                                if 'Top Skin Tone' in cluster_info: st.caption(f"🎨 {cluster_info['Top Skin Tone']}")
                    else:
                        st.warning("No data available for selected filters")

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
