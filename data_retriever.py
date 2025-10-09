import sqlite3
import pandas as pd

DB_PATH = 'covid_data.db'

def query_covid_data(query: str) -> pd.DataFrame:
    """
    Executes a SQL query against the COVID-19 database and returns the result.

    Args:
        query (str): The SQL query to execute.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the query results.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example usage:
    print("Querying total confirmed cases by country...")
    example_query = """
    SELECT 
        country_region, 
        SUM(confirmed) as total_confirmed
    FROM 
        daily_reports
    GROUP BY 
        country_region
    ORDER BY 
        total_confirmed DESC
    LIMIT 10;
    """
    top_countries = query_covid_data(example_query)
    print("Top 10 countries by confirmed cases:")
    print(top_countries)

    print("\nQuerying daily cases for a specific country (e.g., US)...")
    us_query = """
    SELECT
        report_date,
        SUM(confirmed) as daily_confirmed
    FROM
        daily_reports
    WHERE
        country_region = 'US'
    GROUP BY
        report_date
    ORDER BY
        report_date;
    """
    us_cases = query_covid_data(us_query)
    print("Daily confirmed cases in the US:")
    print(us_cases.head())
