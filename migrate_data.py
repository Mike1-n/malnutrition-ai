import pandas as pd
import numpy as np
import os

def load_who_standards():
    """Loads WHO standards from CSV."""
    file_path = 'who_standards.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    return pd.read_csv(file_path)

def calculate_whz(row, standards_df):
    """Calculates WHZ using WHO LMS method."""
    gender = row['gender']
    height = row['height']
    weight = row['weight']
    
    # Filter standards by gender
    df_gender = standards_df[standards_df['gender'] == gender]
    
    # Find nearest height in standards
    # We use abs difference to find closest height
    # Note: standards height is usually in cm, 45-120 range.
    
    # Optimization: Sort by height difference and take first
    nearest_row = df_gender.iloc[(df_gender['height'] - height).abs().argsort()[:1]]
    
    if nearest_row.empty:
        return None # Should not happen if height is within reasonable range
        
    L = nearest_row['L'].values[0]
    M = nearest_row['M'].values[0]
    S = nearest_row['S'].values[0]
    
    try:
        z_score = ((weight / M)**L - 1) / (L * S)
    except:
        return None
        
    return z_score

def migrate():
    print("Loading data...")
    if not os.path.exists('malnutrition_data.csv'):
        print("malnutrition_data.csv not found.")
        return

    df = pd.read_csv('malnutrition_data.csv')
    standards = load_who_standards()
    
    print("Assigning random genders...")
    # Assign gender per child_id, not per row, to be consistent
    child_ids = df['child_id'].unique()
    gender_map = {cid: np.random.choice(['Male', 'Female']) for cid in child_ids}
    df['gender'] = df['child_id'].map(gender_map)
    
    print("Calculating Z-scores...")
    # Apply calculation
    df['WHZ'] = df.apply(lambda row: calculate_whz(row, standards), axis=1)
    
    # Drop MUAC
    if 'MUAC' in df.columns:
        print("Dropping MUAC column...")
        df = df.drop(columns=['MUAC'])
        
    # Drop rows where WHZ could not be calculated (if any)
    initial_len = len(df)
    df = df.dropna(subset=['WHZ'])
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows due to calculation errors (likely height out of range).")
    
    print("Saving updated dataset...")
    df.to_csv('malnutrition_data.csv', index=False)
    print("Migration complete. 'gender' and 'WHZ' columns added. 'MUAC' removed.")

if __name__ == "__main__":
    migrate()
