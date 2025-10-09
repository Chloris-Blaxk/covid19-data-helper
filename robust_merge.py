import os
import pandas as pd
import sqlite3
from datetime import datetime

def robust_merge_data_to_sqlite():
    """
    Merges all daily COVID-19 report CSVs from 2020-2023 into a single SQLite database,
    handling column name variations.
    """
    data_dir = os.path.join('COVID-19-master', 'COVID-19-master', 'csse_covid_19_data', 'csse_covid_19_daily_reports')
    db_path = 'covid_data.db'
    table_name = 'daily_reports'

    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database at {db_path}")

    conn = sqlite3.connect(db_path)
    print(f"Database created at {db_path}")

    try:
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        print(f"Found {len(all_files)} CSV files to process.")
    except FileNotFoundError:
        print(f"Error: Directory not found at {data_dir}")
        return

    # Column name variations mapping
    column_mapping = {
        'province/state': 'province_state',
        'country/region': 'country_region',
        'last update': 'last_update',
        'confirmed': 'confirmed',
        'deaths': 'deaths',
        'recovered': 'recovered'
    }

    chunk_size = 100
    file_chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]

    for i, chunk in enumerate(file_chunks):
        data_frames = []
        print(f"Processing chunk {i+1}/{len(file_chunks)}...")
        for filename in chunk:
            file_path = os.path.join(data_dir, filename)
            
            try:
                report_date_str = filename.replace('.csv', '')
                report_date = datetime.strptime(report_date_str, '%m-%d-%Y').strftime('%Y-%m-%d')
            except ValueError:
                print(f"Skipping file with unexpected name format: {filename}")
                continue

            try:
                df = pd.read_csv(file_path)
                df['report_date'] = report_date
                
                # Standardize column names
                df.columns = df.columns.str.lower().str.strip()
                df = df.rename(columns=column_mapping)

                data_frames.append(df)
            except Exception as e:
                print(f"Error reading or processing {filename}: {e}")

        if not data_frames:
            continue

        merged_df = pd.concat(data_frames, ignore_index=True)

        # Ensure all target columns exist
        expected_cols = ['province_state', 'country_region', 'last_update', 'confirmed', 'deaths', 'recovered', 'report_date']
        for col in expected_cols:
            if col not in merged_df.columns:
                merged_df[col] = None
        
        # Keep only the expected columns
        merged_df = merged_df[expected_cols]

        # Convert data types
        merged_df['confirmed'] = pd.to_numeric(merged_df['confirmed'], errors='coerce').fillna(0).astype(int)
        merged_df['deaths'] = pd.to_numeric(merged_df['deaths'], errors='coerce').fillna(0).astype(int)
        merged_df['recovered'] = pd.to_numeric(merged_df['recovered'], errors='coerce').fillna(0).astype(int)
        merged_df['last_update'] = pd.to_datetime(merged_df['last_update'], errors='coerce')
        merged_df['report_date'] = pd.to_datetime(merged_df['report_date'])

        merged_df.to_sql(table_name, conn, if_exists='append', index=False)
        print(f"Chunk {i+1} successfully appended to the database.")

    print("All data has been merged into the SQLite database.")
    conn.close()

if __name__ == '__main__':
    robust_merge_data_to_sqlite()
