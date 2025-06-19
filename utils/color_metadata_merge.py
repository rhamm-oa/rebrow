"""
Merge Color Analysis with Metadata Script
Only keeps images with complete metadata (no null/empty values)
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

def clean_filename_for_merge(filename):
    """Remove .jpg extension and extract base ID"""
    if pd.isna(filename):
        return None
    # Remove extension and any path
    base_name = Path(filename).stem
    return base_name

def load_and_validate_metadata(metadata_path):
    """Load and validate metadata file"""
    print(f"📂 Loading metadata from: {metadata_path}")
    
    try:
        # Try different separators
        separators = [';', ',', '\t']
        df = None
        
        for sep in separators:
            try:
                df = pd.read_csv(metadata_path, sep=sep)
                if len(df.columns) > 10:  # Should have many columns
                    print(f"✅ Successfully loaded with separator: '{sep}'")
                    break
            except:
                continue
        
        if df is None:
            raise ValueError("Could not determine file separator")
            
        print(f"📊 Metadata shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Validate required columns
        required_cols = ['RESP_FINAL', 'SCF1_MOY', 'eval_cluster', 'ETHNI_USR']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # Clean RESP_FINAL column
        df['RESP_FINAL'] = df['RESP_FINAL'].astype(str)
        
        # Show sample data
        print(f"\n📋 Sample metadata:")
        print(df[required_cols].head())
        
        # Check for complete metadata records
        complete_metadata = df[required_cols].dropna()
        print(f"\n📊 Data completeness:")
        print(f"   • Total metadata records: {len(df)}")
        print(f"   • Complete metadata records: {len(complete_metadata)}")
        print(f"   • Records with missing values: {len(df) - len(complete_metadata)}")
        
        # Show value distributions for complete records only
        if len(complete_metadata) > 0:
            print(f"\n📊 Data distributions (complete records only):")
            print(f"   • Age (SCF1_MOY): {complete_metadata['SCF1_MOY'].min():.0f}-{complete_metadata['SCF1_MOY'].max():.0f} years")
            print(f"   • Skin clusters: {sorted(complete_metadata['eval_cluster'].unique())}")
            print(f"   • Ethnicities: {sorted(complete_metadata['ETHNI_USR'].unique())}")
            print(f"   • Unique RESP_FINAL IDs: {complete_metadata['RESP_FINAL'].nunique()}")
        
        return df[required_cols].copy()
        
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return None

def load_and_validate_colors(colors_path):
    """Load and validate color analysis file"""
    print(f"\n📂 Loading color analysis from: {colors_path}")
    
    try:
        df = pd.read_csv(colors_path)
        print(f"📊 Color data shape: {df.shape}")
        
        if 'image_filename' not in df.columns:
            print(f"❌ Missing 'image_filename' column")
            print(f"Available columns: {list(df.columns)[:10]}...")
            return None
        
        # Clean image filenames for merging and create temporary column
        df['temp_id'] = df['image_filename'].apply(clean_filename_for_merge)
        
        # Show sample data
        print(f"\n📋 Sample color data:")
        print(df[['image_filename', 'temp_id']].head())
        
        # Count non-null IDs
        valid_ids = df['temp_id'].notna().sum()
        print(f"📊 Valid IDs for merge: {valid_ids}/{len(df)}")
        print(f"📊 Unique color analysis IDs: {df['temp_id'].nunique()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading color data: {e}")
        return None

def merge_datasets(metadata_df, colors_df):
    """Merge metadata with color analysis data and keep only complete records"""
    print(f"\n🔄 Merging datasets (keeping ONLY images with COMPLETE metadata)...")
    
    # First, filter metadata to only complete records
    metadata_cols = ['SCF1_MOY', 'eval_cluster', 'ETHNI_USR']
    complete_metadata_df = metadata_df.dropna(subset=metadata_cols)
    
    print(f"📊 Metadata filtering:")
    print(f"   • Original metadata records: {len(metadata_df)}")
    print(f"   • Complete metadata records: {len(complete_metadata_df)}")
    print(f"   • Filtered out (incomplete): {len(metadata_df) - len(complete_metadata_df)}")
    
    # Show merge key distributions
    print(f"\n📊 Merge key statistics:")
    print(f"   • Complete metadata unique IDs: {complete_metadata_df['RESP_FINAL'].nunique()}")
    print(f"   • Color data unique IDs: {colors_df['temp_id'].nunique()}")
    
    # Find common IDs and missing ones
    metadata_ids = set(complete_metadata_df['RESP_FINAL'].astype(str))
    color_ids = set(colors_df['temp_id'].astype(str))
    
    common_ids = metadata_ids.intersection(color_ids)
    metadata_only = metadata_ids - color_ids
    colors_only = color_ids - metadata_ids
    
    print(f"   • Common IDs (will be in final dataset): {len(common_ids)}")
    print(f"   • Complete metadata only (unused): {len(metadata_only)}")
    print(f"   • Colors only (will be EXCLUDED): {len(colors_only)}")
    
    if len(common_ids) == 0:
        print(f"❌ No common IDs found for merging!")
        print(f"Sample complete metadata IDs: {list(metadata_ids)[:5]}")
        print(f"Sample color IDs: {list(color_ids)[:5]}")
        return None, []
    
    # Perform INNER JOIN with complete metadata only
    merged_df = colors_df.merge(
        complete_metadata_df, 
        left_on='temp_id', 
        right_on='RESP_FINAL', 
        how='inner'
    )
    
    # Get list of excluded images (no match OR incomplete metadata)
    excluded_images = colors_df[~colors_df['temp_id'].isin(metadata_ids)]['image_filename'].tolist()
    
    # Remove temporary column
    merged_df = merged_df.drop('temp_id', axis=1)
    
    print(f"✅ Merge successful!")
    print(f"📊 Merged dataset shape: {merged_df.shape}")
    print(f"📊 Original color images: {len(colors_df)}")
    print(f"📊 Final dataset: {len(merged_df)} (all have COMPLETE metadata)")
    print(f"📊 Excluded images: {len(excluded_images)}")
    
    # Verify no missing values in final dataset
    missing_check = merged_df[['SCF1_MOY', 'eval_cluster', 'ETHNI_USR']].isnull().sum()
    print(f"\n🔍 Final dataset completeness check:")
    for col, missing_count in missing_check.items():
        print(f"   • {col}: {missing_count} missing values")
    
    if missing_check.sum() > 0:
        print(f"⚠️ Warning: Final dataset still has missing values!")
        # Additional filtering if needed
        merged_df = merged_df.dropna(subset=['SCF1_MOY', 'eval_cluster', 'ETHNI_USR'])
        print(f"📊 After additional filtering: {len(merged_df)} records")
    else:
        print(f"✅ Perfect! No missing values in final dataset")
    
    # Add ethnicity labels
    ethnicity_display = {
        '1': 'Caucasian',
        '2': 'Black/African American', 
        '3': 'Asian',
        '4': 'Hispanic/Latino',
        '5': 'Native American',
        '7': 'Other'
    }
    
    merged_df['ethnicity_label'] = merged_df['ETHNI_USR'].astype(str).map(ethnicity_display)
    merged_df['ethnicity_label'] = merged_df['ethnicity_label'].fillna('Other')
    
    # Show final statistics
    print(f"\n📊 Final dataset statistics:")
    print(f"   • Total images: {len(merged_df)} (100% have COMPLETE metadata)")
    print(f"   • Age range: {merged_df['SCF1_MOY'].min():.0f}-{merged_df['SCF1_MOY'].max():.0f} years")
    print(f"   • Ethnicities: {merged_df['ethnicity_label'].value_counts().to_dict()}")
    print(f"   • Skin clusters: {merged_df['eval_cluster'].value_counts().to_dict()}")
    
    return merged_df, excluded_images

def generate_excluded_report(excluded_images, output_path, original_color_count, final_count):
    """Generate report for excluded images"""
    if not excluded_images:
        print("✅ No images were excluded - all color analysis images had complete metadata!")
        return None
        
    base_name = Path(output_path).stem
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
    
    # Generate excluded images report
    excluded_txt = os.path.join(output_dir, f"{base_name}_EXCLUDED_IMAGES.txt")
    with open(excluded_txt, 'w') as f:
        f.write(f"EXCLUDED IMAGES REPORT ({len(excluded_images)} images)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reason: No matching metadata OR incomplete metadata\n")
        f.write(f"Original color images: {original_color_count}\n")
        f.write(f"Final dataset: {final_count}\n")
        f.write(f"Excluded: {len(excluded_images)}\n")
        f.write(f"Retention rate: {final_count/original_color_count*100:.1f}%\n\n")
        
        f.write("Excluded image files:\n")
        f.write("-" * 30 + "\n")
        for i, filename in enumerate(sorted(excluded_images), 1):
            f.write(f"{i:3d}. {filename}\n")
        
        # Extract IDs for analysis
        excluded_ids = []
        for filename in excluded_images:
            base_id = Path(filename).stem
            try:
                excluded_ids.append(int(base_id))
            except:
                excluded_ids.append(base_id)
        
        # Analyze numeric IDs
        numeric_ids = [id_ for id_ in excluded_ids if isinstance(id_, int)]
        if numeric_ids:
            numeric_ids.sort()
            f.write(f"\nNumeric ID analysis:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Range: {min(numeric_ids)} - {max(numeric_ids)}\n")
            f.write(f"Count: {len(numeric_ids)} numeric IDs\n")
            
            # Find ranges
            ranges = []
            start = numeric_ids[0]
            end = numeric_ids[0]
            
            for i in range(1, len(numeric_ids)):
                if numeric_ids[i] == end + 1:
                    end = numeric_ids[i]
                else:
                    ranges.append((start, end))
                    start = end = numeric_ids[i]
            ranges.append((start, end))
            
            f.write(f"Ranges: {len(ranges)} separate ranges\n")
            for start, end in ranges[:10]:  # Show first 10 ranges
                if start == end:
                    f.write(f"  {start}\n")
                else:
                    f.write(f"  {start}-{end} ({end-start+1} images)\n")
            if len(ranges) > 10:
                f.write(f"  ... and {len(ranges)-10} more ranges\n")
    
    print(f"📄 Excluded images report saved: {excluded_txt}")
    return excluded_txt

def save_merged_data(merged_df, excluded_images, output_path, original_color_count):
    """Save merged dataset and excluded images report"""
    print(f"\n💾 Saving merged dataset to: {output_path}")
    
    try:
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save merged data (only includes images with complete metadata)
        merged_df.to_csv(output_path, index=False)
        
        print(f"✅ Saved successfully!")
        print(f"📊 Final file: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        print(f"📊 All rows have COMPLETE metadata ✅")
        
        # Generate excluded images report
        if excluded_images:
            generate_excluded_report(excluded_images, output_path, original_color_count, len(merged_df))
        
        # Save summary statistics
        summary_file = output_path.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("Dataset Merge Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Merge strategy: INNER JOIN with complete metadata only\n")
            f.write(f"Original color images: {original_color_count}\n")
            f.write(f"Final dataset: {len(merged_df)} images (100% complete metadata)\n")
            f.write(f"Excluded images: {len(excluded_images)} (no match or incomplete metadata)\n")
            f.write(f"Retention rate: {len(merged_df)/original_color_count*100:.1f}%\n")
            
            # Add breakdown by ethnicity
            f.write("\nEthnicity breakdown:\n")
            ethnicity_counts = merged_df['ethnicity_label'].value_counts()
            for ethnicity, count in ethnicity_counts.items():
                f.write(f"  {ethnicity}: {count}\n")
            
            f.write(f"\nAge statistics:\n")
            f.write(f"  Min age: {merged_df['SCF1_MOY'].min():.0f}\n")
            f.write(f"  Max age: {merged_df['SCF1_MOY'].max():.0f}\n")
            f.write(f"  Mean age: {merged_df['SCF1_MOY'].mean():.1f}\n")
            
            f.write(f"\nSkin cluster breakdown:\n")
            cluster_counts = merged_df['eval_cluster'].value_counts().sort_index()
            for cluster, count in cluster_counts.items():
                f.write(f"  Cluster {cluster}: {count}\n")
        
        print(f"📊 Summary statistics saved: {summary_file}")
        
        # Show column summary
        print(f"\n📋 Column summary:")
        color_cols = [col for col in merged_df.columns if any(x in col for x in ['_L', '_a', '_b', '_percentage', '_quality'])]
        metadata_cols = ['SCF1_MOY', 'eval_cluster', 'ETHNI_USR', 'ethnicity_label', 'RESP_FINAL']
        other_cols = [col for col in merged_df.columns if col not in color_cols + metadata_cols]
        
        print(f"   • Color analysis columns: {len(color_cols)}")
        print(f"   • Metadata columns: {len(metadata_cols)}")
        print(f"   • Other columns: {len(other_cols)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Merge eyebrow color analysis with demographic metadata (only complete records)')
    parser.add_argument('metadata_file', help='Path to MCB_DATA_Merged.csv file')
    parser.add_argument('colors_file', help='Path to color analysis CSV file')
    parser.add_argument('output_file', help='Path for merged output CSV file')
    parser.add_argument('--preview_only', action='store_true', help='Preview merge without saving')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.metadata_file):
        print(f"❌ Metadata file not found: {args.metadata_file}")
        return
    
    if not os.path.exists(args.colors_file):
        print(f"❌ Color analysis file not found: {args.colors_file}")
        return
    
    print("🚀 Starting merge process...")
    print(f"📂 Metadata: {args.metadata_file}")
    print(f"📂 Colors: {args.colors_file}")
    print(f"📂 Output: {args.output_file}")
    print("ℹ️ Strategy: Only keep images with COMPLETE metadata (no null values)")
    
    # Load datasets
    metadata_df = load_and_validate_metadata(args.metadata_file)
    if metadata_df is None:
        return
    
    colors_df = load_and_validate_colors(args.colors_file)
    if colors_df is None:
        return
    
    original_color_count = len(colors_df)
    
    # Merge datasets
    merged_df, excluded_images = merge_datasets(metadata_df, colors_df)
    if merged_df is None:
        return
    
    if args.preview_only:
        print(f"\n👀 Preview mode - showing sample of merged data:")
        preview_cols = ['image_filename', 'RESP_FINAL', 'SCF1_MOY', 'eval_cluster', 'ethnicity_label']
        available_preview_cols = [col for col in preview_cols if col in merged_df.columns]
        print("\n✅ Sample of images WITH COMPLETE metadata (all rows in final dataset):")
        print(merged_df[available_preview_cols].head(10))
        
        # Check for any remaining null values
        null_check = merged_df[['SCF1_MOY', 'eval_cluster', 'ETHNI_USR']].isnull().sum()
        print(f"\n🔍 Null value check in preview:")
        for col, null_count in null_check.items():
            print(f"   • {col}: {null_count} null values")
        
        if excluded_images:
            print(f"\n❌ Images that will be EXCLUDED ({len(excluded_images)} total):")
            for i, img in enumerate(excluded_images[:5], 1):
                print(f"   {i}. {img}")
            if len(excluded_images) > 5:
                print(f"   ... and {len(excluded_images) - 5} more")
        
        print(f"\n📊 Preview complete. Use without --preview_only to save the merged file.")
    else:
        # Save merged dataset
        success = save_merged_data(merged_df, excluded_images, args.output_file, original_color_count)
        if success:
            print(f"\n🎉 Merge completed successfully!")
            print(f"📁 Clean dataset ready for analysis: {args.output_file}")
            print(f"📊 Final dataset: {len(merged_df)} images (100% complete metadata)")
            print(f"📊 Retention rate: {len(merged_df)/original_color_count*100:.1f}%")
            if excluded_images:
                print(f"📄 Excluded {len(excluded_images)} images (no match or incomplete metadata)")
                print(f"📄 See excluded images report for details")
        else:
            print(f"\n❌ Process failed!")

if __name__ == "__main__":
    main()