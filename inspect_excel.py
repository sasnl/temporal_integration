import pandas as pd
import sys

file_path = '/Users/tongshan/Documents/TemporalIntegration/data/Temporal_integrartion_subjectlist_2026_01_16_updated_filtering.xlsx'

try:
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Try to identify subject columns and condition columns
    # Heuristic search
    print("\nPotential groupings:")
    if 'record_id' in df.columns:
        print("Found 'record_id' column.")
        
    for col in df.columns:
        print(f"Column '{col}' unique values (top 5): {df[col].unique()[:5]}")

except Exception as e:
    print(f"Error reading excel: {e}")
