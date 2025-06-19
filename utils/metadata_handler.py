# metadata_handler.py
import pandas as pd
import os

class MetadataHandler:
    def __init__(self, csv_path="data/mcb_data/MCB_DATA_MERGED.csv"):
        """
        Initialize the metadata handler with the CSV file containing ethnicity and other data.
        """
        self.df = None
        self.available = False
        
        if csv_path and os.path.exists(csv_path):
            try:
                self.df = pd.read_csv(csv_path, sep=';')  # Using semicolon separator
                self.available = True
                print(f"✅ Loaded metadata for {len(self.df)} individuals")
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.available = False
        
        self.ethnicity_map = {
            1: 'caucasian',
            2: 'black_african_american', 
            3: 'asian',
            4: 'hispanic_latino',
            5: 'native_american',
            7: 'others'
        }
        
    def get_person_metadata(self, filename):
        """
        Get metadata for a person based on filename.
        
        Args:
            filename: Image filename (e.g., "1010.jpg")
            
        Returns:
            dict: Metadata including ethnicity, skin cluster, age, etc. or None if not available
        """
        if not self.available or self.df is None:
            return None
            
        # Extract identifier from filename (remove extension)
        identifier = os.path.splitext(filename)[0]
        
        try:
            identifier = int(identifier)
        except ValueError:
            return None
        
        # Find the row with matching RESP_FINAL
        row = self.df[self.df['RESP_FINAL'] == identifier]
        
        if row.empty:
            return None
        
        row = row.iloc[0]  # Get first match
        
        ethnicity_code = row.get('ETHNI_USR', 7)
        ethnicity_name = self.ethnicity_map.get(ethnicity_code, 'others')
        skin_cluster = row.get('eval_cluster', 3)
        
        metadata = {
            'identifier': identifier,
            'ethnicity_code': ethnicity_code,
            'ethnicity_name': ethnicity_name,
            'skin_cluster': min(max(int(skin_cluster), 1), 6),  # Ensure 1-6 range
            'age_interval': row.get('SCF1R', 1),
            'actual_age': row.get('SCF1_MOY', 35),
            'hair_length': row.get('HAIR_LENGTHR', None),
            'hair_thickness': row.get('HAIR_THICKNESSR', None),
            'hair_type': row.get('HAIR_TYPER', None),
            'hair_grey': row.get('HAIR_GREYR', None),
        }
        
        return metadata