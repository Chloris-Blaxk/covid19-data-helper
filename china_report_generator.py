import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from data_retriever import query_covid_data

# 配置 Matplotlib 使用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_china_provinces():
    """
    获取中国所有省份的列表。
    """
    query = """
    SELECT DISTINCT
        province_state
    FROM
        daily_reports
    WHERE
        country_region = 'China' AND province_state IS NOT NULL;
    """
    df = query_covid_data(query)
    return df['province_state'].tolist()

def plot_daily_cases_for_china_provinces(provinces: list, start_date: str, end_date: str, output_dir='plots'):
    """
    查询指定日期范围内中国特定省份的每日病例数，并保存为折线图。
    返回保存的图表路径。
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
        country_region = 'China' AND
        province_state IN {provinces_tuple} AND
        report_date BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY
        report_date, province_state
    ORDER BY
        report_date;
    """
    df = query_covid_data(query)

    if df.empty:
        return f"在指定日期范围内，未找到中国省份 {', '.join(provinces)} 的数据。"

    df['report_date'] = pd.to_datetime(df['report_date'])
    
    pivot_df = df.pivot(index='report_date', columns='province_state', values='daily_confirmed').fillna(0)

    plt.figure(figsize=(15, 10))
    for province in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[province], label=province)
    
    plt.xlabel('日期')
    plt.ylabel('每日确诊病例数')
    plt.title(f'中国各省份每日确诊病例 ({start_date} to {end_date})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_path = os.path.join(output_dir, 'daily_cases_provinces_China_2021_3_6.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def get_monthly_deaths_for_china(start_date: str, end_date: str):
    """
    获取指定日期范围内中国每月的死亡人数。
    """
    query = f"""
    SELECT
        STRFTIME('%Y-%m', report_date) as month,
        SUM(deaths) as total_deaths
    FROM
        daily_reports
    WHERE
        country_region = 'China' AND
        report_date BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY
        month
    ORDER BY
        month;
    """
    df = query_covid_data(query)
    if df.empty:
        return "在指定日期范围内没有找到中国的死亡数据。"
    return df.to_string()

def generate_china_report():
    """
    生成2021年3月至6月中国疫情报告。
    """
    start_date = '2021-03-01'
    end_date = '2021-06-30'
    
    # 1. 获取省份列表并生成图表
    provinces = get_china_provinces()
    provinces_to_plot = [p for p in provinces if p not in ['Unknown', 'Diamond Princess', 'Grand Princess']]
    if provinces_to_plot:
        print(f"正在为中国的以下省份生成图表: {', '.join(provinces_to_plot)}")
        plot_path = plot_daily_cases_for_china_provinces(provinces_to_plot, start_date, end_date)
        print(f"图表已保存至: {plot_path}")
    else:
        print("未能获取用于绘图的省份列表。")

    # 2. 获取月度死亡人数
    monthly_deaths = get_monthly_deaths_for_china(start_date, end_date)
    print("\n2021年3月至6月中国月度死亡人数:")
    print(monthly_deaths)

if __name__ == '__main__':
    generate_china_report()
