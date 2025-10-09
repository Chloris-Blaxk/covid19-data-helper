from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field
import data_analyzer

# Pydantic models for tool inputs

class PlotTopCountriesInput(BaseModel):
    top_n: int = Field(10, description="The number of countries to include in the plot.")

class CountryNameInput(BaseModel):
    country_name: str = Field(..., description="The name of the country.")

from typing import List

class DeathsOnDateInput(BaseModel):
    country_name: str = Field(..., description="The name of the country.")
    month: int = Field(..., description="The month (1-12).")
    day: int = Field(..., description="The day (1-31).")

class PlotProvincesInput(BaseModel):
    country_name: str = Field(..., description="The name of the country, e.g., 'China'.")
    provinces: List[str] = Field(..., description="A list of province or state names to plot.")
    start_date: str = Field(..., description="The start date in 'YYYY-MM-DD' format.")
    end_date: str = Field(..., description="The end date in 'YYYY-MM-DD' format.")

# Tool definitions

@tool(args_schema=PlotTopCountriesInput)
def plot_top_countries_by_confirmed_cases(top_n: int = 10) -> str:
    """
    Generates and saves a bar chart of the top N countries by total confirmed COVID-19 cases.
    Returns the file path of the generated plot.
    IMPORTANT: For your final answer, you MUST embed this returned path in a Markdown image tag like '![alt text](path)'.
    """
    return data_analyzer.plot_top_countries_by_confirmed_cases(top_n=top_n)

@tool(args_schema=CountryNameInput)
def plot_daily_cases_for_country(country_name: str) -> str:
    """
    Generates and saves a line chart of the daily confirmed COVID-19 cases for a specific country.
    Returns the file path of the generated plot.
    IMPORTANT: For your final answer, you MUST embed this returned path in a Markdown image tag like '![alt text](path)'.
    """
    return data_analyzer.plot_daily_cases_for_country(country_name=country_name)

@tool(args_schema=CountryNameInput)
def calculate_mortality_rate(country_name: str) -> str:
    """
    Calculates the overall mortality rate (deaths / confirmed cases) for a specific country.
    Returns a string with the calculated rate.
    """
    return data_analyzer.calculate_mortality_rate(country_name=country_name)

@tool(args_schema=CountryNameInput)
def calculate_daily_growth_rate(country_name: str) -> str:
    """
    Calculates the daily growth rate of confirmed cases for a specific country.
    Returns a string with the growth rate for the last 5 days.
    """
    return data_analyzer.calculate_daily_growth_rate(country_name=country_name)

@tool(args_schema=DeathsOnDateInput)
def get_deaths_on_date(country_name: str, month: int, day: int) -> str:
    """
    Retrieves the total deaths for a specific country on a given month and day (e.g., month=3, day=1 for March 1st) for all available years.
    To answer questions about a date range, you MUST call this tool multiple times, once for each specific date.
    """
    return data_analyzer.get_deaths_on_date(country_name=country_name, month=month, day=day)

@tool(args_schema=PlotProvincesInput)
def plot_daily_cases_for_provinces(country_name: str, provinces: List[str], start_date: str, end_date: str) -> str:
    """
    Use this tool when a user asks for a plot of specific provinces or states within a country, especially for a specific date range.
    It generates a line chart comparing daily cases for the given provinces between the start and end dates.
    Returns the file path of the generated plot.
    IMPORTANT: For your final answer, you MUST embed this returned path in a Markdown image tag like '![alt text](path)'.
    """
    return data_analyzer.plot_daily_cases_for_provinces(country_name=country_name, provinces=provinces, start_date=start_date, end_date=end_date)

# List of all tools for the agent
all_tools = [
    plot_top_countries_by_confirmed_cases,
    plot_daily_cases_for_country,
    plot_daily_cases_for_provinces,
    calculate_mortality_rate,
    calculate_daily_growth_rate,
    get_deaths_on_date,
]
