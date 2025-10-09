import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
from data_retriever import query_covid_data
import os

# Configure Matplotlib to use a Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # Fix for displaying negative signs

def plot_top_countries_by_confirmed_cases(df: pd.DataFrame, top_n=10, output_dir='static/plots'):
    """
    Saves a bar chart of the top N countries by total confirmed cases from a DataFrame.
    Returns the path to the saved plot.
    """
    if df.empty:
        return None

    plt.figure(figsize=(12, 8))
    plt.bar(df['country_region'], df['total_confirmed'])
    plt.xlabel('国家/地区')
    plt.ylabel('累计确诊病例数')
    plt.title(f'全球累计确诊病例数排名前 {top_n} 的国家/地区')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plot_path = os.path.join(output_dir, 'top_countries_confirmed.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_daily_cases_for_country(df: pd.DataFrame, country_name: str, output_dir='static/plots'):
    """
    Saves a line chart of daily cases for a specific country from a DataFrame.
    Returns the path to the saved plot.
    """
    if df.empty:
        return None

    df['report_date'] = pd.to_datetime(df['report_date'])

    plt.figure(figsize=(12, 8))
    plt.plot(df['report_date'], df['daily_confirmed'])
    plt.xlabel('日期')
    plt.ylabel('每日确诊病例数')
    plt.title(f'{country_name} 每日COVID-19确诊病例')
    plt.grid(True)
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sanitize filename
    safe_country_name = "".join([c for c in country_name if c.isalpha() or c.isdigit()]).rstrip()
    plot_path = os.path.join(output_dir, f'daily_cases_{safe_country_name}.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_daily_cases_for_provinces(country_name: str, provinces: list, start_date: str, end_date: str, output_dir='static/plots'):
    """
    Queries daily cases for a list of provinces in a specific country within a date range and saves a line chart.
    Returns the path to the saved plot.
    """
    provinces_tuple = tuple(provinces)
    query = f"""
    SELECT
        report_date,
        province_state,
        SUM(confirmed) as daily_confirmed
    FROM
        daily_reports
    WHERE
        country_region = '{country_name}' AND
        province_state IN {provinces_tuple} AND
        report_date BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY
        report_date, province_state
    ORDER BY
        report_date;
    """
    df = query_covid_data(query)

    if df.empty:
        return f"在指定日期范围内，未找到 {country_name} 的省份 {', '.join(provinces)} 的数据。"

    df['report_date'] = pd.to_datetime(df['report_date'])
    
    # Pivot the dataframe to have provinces as columns
    pivot_df = df.pivot(index='report_date', columns='province_state', values='daily_confirmed').fillna(0)

    plt.figure(figsize=(15, 10))
    for province in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[province], label=province)
    
    plt.xlabel('日期')
    plt.ylabel('每日确诊病例数')
    plt.title(f'{country_name} 各省份每日确诊病例 ({start_date} to {end_date})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    safe_country_name = "".join([c for c in country_name if c.isalpha() or c.isdigit()]).rstrip()
    plot_path = os.path.join(output_dir, f'daily_cases_provinces_{safe_country_name}.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def calculate_mortality_rate(country_name: str):
    """
    Calculates the overall mortality rate for a specific country.
    Returns a formatted string with the result.
    """
    query = f"""
    SELECT 
        SUM(confirmed) as total_confirmed,
        SUM(deaths) as total_deaths
    FROM 
        daily_reports
    WHERE
        country_region = '{country_name}';
    """
    df = query_covid_data(query)

    if df.empty or df.loc[0, 'total_confirmed'] == 0:
        return f"无法计算 {country_name} 的病死率（无数据或无确诊病例）。"

    mortality_rate = (df.loc[0, 'total_deaths'] / df.loc[0, 'total_confirmed']) * 100
    return f"{country_name} 的整体病死率: {mortality_rate:.2f}%"

def calculate_daily_growth_rate(country_name: str):
    """
    Calculates the daily growth rate of confirmed cases for a country.
    Returns a formatted string with the last 5 days of data.
    """
    query = f"""
    SELECT
        report_date,
        SUM(confirmed) as daily_confirmed
    FROM
        daily_reports
    WHERE
        country_region = '{country_name}'
    GROUP BY
        report_date
    ORDER BY
        report_date;
    """
    df = query_covid_data(query)

    if len(df) < 2:
        return f"没有足够的数据来计算 {country_name} 的增长率。"
    
    df['growth_rate'] = df['daily_confirmed'].pct_change() * 100
    
    # Return the tail of the dataframe as a string
    return f"{country_name} 的每日增长率 (最近5天):\n{df[['report_date', 'growth_rate']].tail().to_string()}"

def get_deaths_on_date(country_name: str, month: int, day: int):
    """
    Queries the total deaths for a specific country on a given month and day for all available years.
    Returns a formatted string with the results.
    """
    month_str = f"{month:02d}"
    day_str = f"{day:02d}"
    
    query = f"""
    SELECT
        report_date,
        SUM(deaths) as total_deaths
    FROM
        daily_reports
    WHERE
        country_region = '{country_name}' AND
        STRFTIME('%m-%d', report_date) = '{month_str}-{day_str}'
    GROUP BY
        report_date
    ORDER BY
        report_date;
    """
    df = query_covid_data(query)

    if df.empty:
        return f"没有找到 {country_name} 在 {month}月{day}日 的数据。"
    
    return f"{country_name} 在 {month}月{day}日 的死亡人数:\n{df.to_string()}"

if __name__ == '__main__':
    plot_top_countries_by_confirmed_cases(top_n=15)
    plot_daily_cases_for_country('US')
    plot_daily_cases_for_country('China')
    plot_daily_cases_for_country('India')

    calculate_mortality_rate('US')
    calculate_daily_growth_rate('US')
