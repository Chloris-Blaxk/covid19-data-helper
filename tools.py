import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field
import data_analyzer
from data_retriever import query_covid_data
import pandas as pd
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Pydantic models for tool inputs

class NaturalLanguageQueryInput(BaseModel):
    query: str = Field(..., description="The natural language query to convert to SQL.")

class SQLQueryInput(BaseModel):
    query: str = Field(..., description="The SQL query to execute.")

# Tool definitions

@tool(args_schema=NaturalLanguageQueryInput)
def generate_sql_query(query: str) -> str:
    """
    Converts a natural language query to a SQL query for the COVID-19 database.

    The database schema is as follows:
    - Table: daily_reports
      - id (INTEGER, PRIMARY KEY)
      - report_date (TEXT): The date of the report in 'YYYY-MM-DD HH:MM:SS' format.
      - country_region (TEXT): The country or region.
      - province_state (TEXT): The province or state.
      - confirmed (INTEGER): Total confirmed cases.
      - deaths (INTEGER): Total deaths.
      - recovered (INTEGER): Total recovered cases.

    IMPORTANT: When filtering by date, always use the DATE() function on the 'report_date' column
    to ensure accurate matching. For example, to filter for a specific day, use:
    WHERE DATE(report_date) = 'YYYY-MM-DD'
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
        base_url=base_url
    )
    
    prompt = f"""
    You are a SQL expert. Convert the following natural language query to a SQL query,
    following the instructions and schema provided in the tool's docstring.

    **Crucial Instruction:** If the user's query asks for a summary, a total, a comparison, or a plot,
    your SQL query MUST perform the necessary aggregation (e.g., SUM, COUNT, GROUP BY) to produce a concise,
    summarized result. Do NOT return large sets of raw data.

    Natural language query: "{query}"
    SQL query:
    """
    
    response = model.invoke(prompt)
    return response.content

@tool(args_schema=SQLQueryInput)
def execute_sql_query(query: str) -> str:
    """
    Executes a SQL query against the COVID-19 database and returns the result as a JSON string.
    """
    df = query_covid_data(query)
    if isinstance(df, pd.DataFrame) and not df.empty:
        # Ensure the date column is in the correct format
        if 'report_date' in df.columns:
            df['report_date'] = pd.to_datetime(df['report_date']).dt.strftime('%Y-%m-%d')
        
        # Identify potential value columns and clean them
        value_columns = ['total_confirmed', 'confirmed', 'deaths', 'recovered']
        for col in value_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with any null values after cleaning
        df.dropna(inplace=True)
        
        # Return data as a JSON string in 'records' format
        return df.to_json(orient='records')
    return json.dumps([])

# Load MCP configuration
with open('mcp_config.json', 'r') as f:
    mcp_config = json.load(f)

async def get_all_tools():
    """
    Initializes the MultiServerMCPClient and retrieves the tools from the MCP servers,
    then combines them with the local tools.
    """
    client = MultiServerMCPClient(mcp_config['mcpServers'])
    mcp_tools = await client.get_tools()
    
    local_tools = [
        generate_sql_query,
        execute_sql_query,
    ]
    
    return local_tools + mcp_tools
