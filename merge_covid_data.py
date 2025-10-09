import os
import pandas as pd
import sqlite3
from datetime import datetime

def merge_data_to_sqlite():
    """
    Merges all daily COVID-19 report CSVs into a single SQLite database.
    """
    data_dir = os.path.join('COVID-19-master', 'COVID-19-master', 'csse_covid_19_data', 'csse_covid_19_daily_reports')
    db_path = 'covid_data.db'
    table_name = 'daily_reports'

    # Connect to SQLite database (it will be created if it doesn't exist)
    conn = sqlite3.connect(db_path)
    
    print(f"Database created at {db_path}")

    # Get list of all CSV files
    try:
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        print(f"Found {len(all_files)} CSV files to process.")
    except FileNotFoundError:
        print(f"Error: Directory not found at {data_dir}")
        return

    chunk_size = 100  # Process files in chunks to manage memory
    file_chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]

    for i, chunk in enumerate(file_chunks):
        data_frames = []
        print(f"Processing chunk {i+1}/{len(file_chunks)}...")
        for filename in chunk:
            file_path = os.path.join(data_dir, filename)
            
            # Extract date from filename
            try:
                report_date_str = filename.replace('.csv', '')
                report_date = datetime.strptime(report_date_str, '%m-%d-%Y').strftime('%Y-%m-%d')
            except ValueError:
                print(f"Skipping file with unexpected name format: {filename}")
                continue

            # Read CSV and add the report date
            try:
                df = pd.read_csv(file_path)
                df['Report_Date'] = report_date
                data_frames.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

        if not data_frames:
            continue

        # Concatenate all dataframes in the chunk
        merged_df = pd.concat(data_frames, ignore_index=True)

        # Clean and standardize data
        # Lowercase and standardize column names
        merged_df.columns = merged_df.columns.str.lower().str.replace('/', '_').str.replace(' ', '_')
        
        # Ensure essential columns exist, fill with None if not
        expected_cols = ['province_state', 'country_region', 'last_update', 'confirmed', 'deaths', 'recovered', 'report_date']
        for col in expected_cols:
            if col not in merged_df.columns:
                merged_df[col] = None

        # Select and reorder columns for consistency
        merged_df = merged_df[['province_state', 'country_region', 'last_update', 'confirmed', 'deaths', 'recovered', 'report_date']]

        # Convert to appropriate data types
        merged_df['confirmed'] = pd.to_numeric(merged_df['confirmed'], errors='coerce').fillna(0)
        merged_df['deaths'] = pd.to_numeric(merged_df['deaths'], errors='coerce').fillna(0)
        merged_df['recovered'] = pd.to_numeric(merged_df['recovered'], errors='coerce').fillna(0)
        merged_df['last_update'] = pd.to_datetime(merged_df['last_update'], errors='coerce')
        merged_df['report_date'] = pd.to_datetime(merged_df['report_date'])

        # Append to SQLite table
        # Use 'if_exists='append'' to add data from each chunk
        # Use 'index=False' to not write pandas index as a column
        merged_df.to_sql(table_name, conn, if_exists='append', index=False)
        print(f"Chunk {i+1} successfully appended to the database.")

    print("All data has been merged into the SQLite database.")
    
    # Close the connection
    conn.close()

if __name__ == '__main__':
    merge_data_to_sqlite()
