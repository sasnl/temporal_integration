import pandas as pd
import os

def process_demographics(input_file, output_file):
    print(f"Reading data from {input_file}")
    df = pd.read_csv(input_file)

    # 1. Drop 'redcap_event_name' column
    if 'redcap_event_name' in df.columns:
        print("Dropping 'redcap_event_name' column")
        df = df.drop(columns=['redcap_event_name'])
    
    # 2. Merge rows with same 'record_id'
    # We group by 'record_id' and then take the first non-null value for each column.
    # Since the user stated "Every other colum should have only one data for each record_id",
    # taking the first available non-null value is the correct strategy.
    print("Merging rows by 'record_id'")
    df_merged = df.groupby('record_id').first().reset_index()
    
    # Alternatively, if we want to be safer and ensure we catch conflicts (though user said there aren't any):
    # df_merged = df.groupby('record_id').agg(lambda x: x.dropna().iloc[0] if x.count() > 0 else None).reset_index()
    # The .first() method on groupby behaves similarly for non-nulls if we sort or if the data is structured such that nans are skipped? 
    # Actually .first() returns the first non-null value! 
    # Reference: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.first.html
    # "Compute first of group values." -> "Method to compute first of group values" 
    # Wait, let's verify if first() skips NaNs. 
    # "first() : Compute the first non-null entry of each column." (Pandas documentation)
    # So .first() is exactly what we want.

    print(f"Original shape: {df.shape}")
    print(f"Merged shape: {df_merged.shape}")

    print(f"Saving processed data to {output_file}")
    df_merged.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    input_csv = "/Users/tongshan/Documents/TemporalIntegration/data/demographic/60483ASDSpeakerListe-TISubjectDemographic_DATA_2026-01-29_1113.csv"
    output_csv = "/Users/tongshan/Documents/TemporalIntegration/data/demographic/processed_demographics.csv"
    
    if not os.path.exists(input_csv):
        print(f"Error: Input file not found: {input_csv}")
    else:
        process_demographics(input_csv, output_csv)
