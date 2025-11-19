import asyncio
import json
import subprocess
import sys
from datetime import datetime
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field
import pandas as pd
import sqlite3
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import dashscope
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain.chains import RetrievalQA
import re
import json5
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
import base64

load_dotenv()

# Pydantic models for tool inputs

class NaturalLanguageQueryInput(BaseModel):
    query: str = Field(..., description="The natural language query to convert to SQL.")

class SQLQueryInput(BaseModel):
    query: str = Field(..., description="The SQL query to execute.")

class DeepResearchInput(BaseModel):
    query: str = Field(..., description="The research question for in-depth analysis.")

class RAGQueryInput(BaseModel):
    query: str = Field(..., description="User question to be answered with literature-augmented QA.")
    k: int = Field(5, description="Top-K documents to retrieve.")

class VisitUrlInput(BaseModel):
    url: str = Field(..., description="The URL of the webpage to start the web exploration from.")

class VisitPageInput(BaseModel):
    button: str = Field(description="the button you want to click")

def _process_deep_research_responses(responses):
    """Helper function to process streaming responses from the deep research model."""
    final_content = ""
    for response in responses:
        if hasattr(response, 'status_code') and response.status_code == 200:
            if hasattr(response, 'output') and response.output:
                message = response.output.get('message', {})
                content = message.get('content', '')
                if content:
                    final_content += content
        else:
            # Handle potential errors
            error_message = f"API Error: status_code={getattr(response, 'status_code', 'N/A')}, message={getattr(response, 'message', 'N/A')}"
            print(error_message) # Log error for debugging
            return f"An error occurred during the research process: {error_message}"
    return final_content

@tool(args_schema=DeepResearchInput)
async def deep_research(query: str) -> str:
    """
    使用前需要先通过对话取得用户同意。
    Performs in-depth research on a given topic using the qwen-deep-research model.
    It internally handles a two-step process: first, generating clarifying questions,
    and second, conducting the research based on the initial query.
    The final, comprehensive report is returned as a string.
    使用中文回答。
    """
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        return "Error: DASHSCOPE_API_KEY is not set."

    # Step 1: Model asks clarifying questions
    messages = [{'role': 'user', 'content': query}]
    try:
        step1_responses = dashscope.Generation.call(
            api_key=api_key,
            model="qwen-deep-research",
            messages=messages,
            stream=True
        )
        step1_content = _process_deep_research_responses(step1_responses)

        # Step 2: In-depth research (simulating user pressing enter)
        messages.append({'role': 'assistant', 'content': step1_content})
        messages.append({'role': 'user', 'content': ""}) # Simulate empty response

        step2_responses = dashscope.Generation.call(
            api_key=api_key,
            model="qwen-deep-research",
            messages=messages,
            stream=True
        )
        final_report = _process_deep_research_responses(step2_responses)
        
        return f"Initial Clarification:\n{step1_content}\n\nFinal Research Report:\n{final_report}"

    except Exception as e:
        return f"An error occurred while calling the deep research API: {e}"

from shared import get_vector_store

@tool(args_schema=RAGQueryInput)
def rag_answer_query(query: str, k: int = 5) -> str:
    """
    Retrieves relevant literature chunks from the 'cord_idx' index in Redis based on a query.
    IMPORTANT: The user's query will be in Chinese. You MUST translate it to English before passing it to this tool.
    The underlying scientific literature database (CORD-19) is in English.
    """
    try:
        # Hardcode the index_name to prevent incorrect calls
        index_name = "cord_idx"
        store = get_vector_store(index_name)
        retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": max(1, int(k))})
        
        # Retrieve documents
        documents = retriever.get_relevant_documents(query)
        
        if documents:
            return json.dumps([doc.page_content for doc in documents])
        else:
            return json.dumps({
                "status": "not_found",
                "message": "No relevant information found in the literature database."
            })

    except Exception as e:
        return json.dumps({"status": "error", "message": f"RAG query failed: {e}"})

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
        model="gemini-2.5-pro",
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

def query_covid_data(query: str) -> pd.DataFrame:
    """
    Queries the COVID-19 SQLite database and returns the result as a pandas DataFrame.
    """
    db_path = 'covid_data.db'
    try:
        with sqlite3.connect(db_path) as con:
            df = pd.read_sql_query(query, con)
        return df
    except Exception as e:
        print(f"Database query failed: {e}")
        return pd.DataFrame()

@tool(args_schema=SQLQueryInput)
def execute_sql_query(query: str) -> str:
    """
    Executes a SQL query against the COVID-19 database and returns the result as a JSON string.
    If the result set is large (more than 20 rows), it will be automatically summarized to the top 10 confirmed cases.
    """
    df = query_covid_data(query)
    if isinstance(df, pd.DataFrame) and not df.empty:
        # Data cleaning and preparation
        if 'report_date' in df.columns:
            df['report_date'] = pd.to_datetime(df['report_date']).dt.strftime('%Y-%m-%d')
        
        value_columns = ['total_confirmed', 'confirmed', 'deaths', 'recovered']
        for col in value_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(inplace=True)

        # Summarize large datasets to prevent content filter issues
        if len(df) > 20 and 'total_confirmed' in df.columns:
            df_sorted = df.sort_values(by='total_confirmed', ascending=False)
            df_top10 = df_sorted.head(10)
            summary_message = {
                "status": "summarized",
                "message": "The result set was too large and has been automatically summarized to the top 10 countries by total confirmed cases.",
                "data": json.loads(df_top10.to_json(orient='records'))
            }
            return json.dumps(summary_message, indent=2)
            
        return df.to_json(orient='records')
    return json.dumps([])

import urllib.parse

def process_url(url, sub_url):
    """
    Args:
        url (str): url
        sub_url (str): sub_url
    
    Returns:
        str: the processed url
    """
    return urllib.parse.urljoin(url, sub_url)


def clean_markdown(res):
    """
    Args:
        res (str): markdown content
    
    Returns:
        str: cleaned markdown content
    """
    pattern = r'\[.*?\]\(.*?\)'
    try:
        result = re.sub(pattern, '', res)
        url_pattern = pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        result = re.sub(url_pattern, '', result)
        result = result.replace("* \n","")
        result = re.sub(r"\n\n+", "\n", result)
        return result
    except Exception:
        return res

async def get_info(url, screenshot = True) -> str:
    """
    Args:
        url (str): url
        screentshot (bool): whether to take a screenshot
    
    Returns:
        str: html content and cleaned markdown content
    """
    run_config = CrawlerRunConfig(
        screenshot=True,             # Grab a screenshot as base64
        screenshot_wait_for=1.0,     # Wait 1s before capturing
    )
    async with AsyncWebCrawler() as crawler:
        if screenshot:
            result = await crawler.arun(url, config=run_config)
            return result.html, clean_markdown(result.markdown), result.screenshot
        else:
            result = await crawler.arun(url, screenshot=screenshot)
            return result.html, clean_markdown(result.markdown)
    
def get_content_between_a_b(start_tag, end_tag, text):
    """
    Args:
        start_tag (str): start_tag
        end_tag (str): end_tag
        text (str): complete sentence

    Returns:
        str: the content between start_tag and end_tag
    """
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break

    return extracted_text.strip()

def extract_links_with_text(html):
    """
    Args:
        html (str): html content
    
    Returns:
        str: clickable buttons
    """
    with open("ROOT_URL.txt", "r") as f:
        ROOT_URL = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    links = []

    for a_tag in soup.find_all('a', href=True):
        url = a_tag['href']
        text = ''.join(a_tag.stripped_strings)
        
        if text and "javascript" not in url and not url.endswith(('.jpg', '.png', '.gif', '.jpeg', '.pdf')):
            if process_url(ROOT_URL, url).startswith(ROOT_URL):
                links.append({'url': process_url(ROOT_URL, url), 'text': text})

    for a_tag in soup.find_all('a', onclick=True):
        onclick_text = a_tag['onclick']
        text = ''.join(a_tag.stripped_strings)
        
        match = re.search(r"window\.location\.href='([^']*)'", onclick_text)
        if match:
            url = match.group(1)
            if url and text  and not url.endswith(('.jpg', '.png', '.gif', '.jpeg', '.pdf')):
                if process_url(ROOT_URL, url).startswith(ROOT_URL):
                    links.append({'url': process_url(ROOT_URL, url), 'text': text})

    for a_tag in soup.find_all('a', attrs={'data-url': True}):
        url = a_tag['data-url']
        text = ''.join(a_tag.stripped_strings)
        if url and text and not url.endswith(('.jpg', '.png', '.gif', '.jpeg', '.pdf')):
            if process_url(ROOT_URL, url).startswith(ROOT_URL):
                links.append({'url': process_url(ROOT_URL, url), 'text': text})

    for a_tag in soup.find_all('a', class_='herf-mask'):
        url = a_tag.get('href')
        text = a_tag.get('title') or ''.join(a_tag.stripped_strings)
        if url and text and not url.endswith(('.jpg', '.png', '.gif', '.jpeg', '.pdf')):
            if process_url(ROOT_URL, url).startswith(ROOT_URL):
                    links.append({'url': process_url(ROOT_URL, url), 'text': text})

    for button in soup.find_all('button', onclick=True):
        onclick_text = button['onclick']
        text = button.get('title') or button.get('aria-label') or ''.join(button.stripped_strings)
        match = re.search(r"window\.location\.href='([^']*)'", onclick_text)
        if match:
            url = match.group(1)
            if url and text:
                if process_url(ROOT_URL, url).startswith(ROOT_URL):
                    links.append({'url': process_url(ROOT_URL, url), 'text': text})

    unique_links = {f"{item['url']}_{item['text']}": item for item in links}  # 去重

    if not os.path.exists("BUTTON_URL_ADIC.json"):
        with open("BUTTON_URL_ADIC.json", "w") as f:
            json.dump({}, f)
    with open("BUTTON_URL_ADIC.json", "r") as f:
        BUTTON_URL_ADIC = json.load(f)
    for temp in list(unique_links.values()):
        BUTTON_URL_ADIC[temp["text"]] = temp["url"]
    with open("BUTTON_URL_ADIC.json", "w") as f:
        json.dump(BUTTON_URL_ADIC, f)
    info = ""
    for i in list(unique_links.values()):
        info += "<button>" + i["text"] + "<button>" + "\n"
    return info

@tool(args_schema=VisitUrlInput)
async def visit_url(url: str) -> str:
    """
    Visits a given URL to start the web exploration process.
    This tool should be the first step for any web-based research.
    It returns the content of the initial page and a list of clickable buttons.
    """
    try:
        # Set the root URL for the session
        with open("ROOT_URL.txt", "w") as f:
            f.write(url)

        html, markdown, screenshot = await get_info(url)

        # Save screenshot for debugging
        if screenshot:
            image_folder = "images/"
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            with open(os.path.join(image_folder, f"screenshot_initial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"), "wb") as f:
                f.write(base64.b64decode(screenshot))

        response_buttons = extract_links_with_text(html)
        response_content = markdown

        if response_content:
            response = f"Successfully visited the initial URL. The web information is:\n\n{response_content}\n\n"
        else:
            response = "The information of the initial page is not accessible.\n\n"

        response += "You can now use the `visit_page` tool to click on the following buttons:\n" + response_buttons
        return response
    except Exception as e:
        return f"An error occurred during the initial visit: {e}"

@tool(args_schema=VisitPageInput)
async def visit_page(button: str) -> str:
    """
    Clicks on a button found on the current webpage to navigate to a new page.
    Use this tool to explore the website after an initial visit with the `visit_url` tool.
    It returns the content of the new page and a new list of clickable buttons.
    """
    try:
        with open("BUTTON_URL_ADIC.json", "r") as f:
            BUTTON_URL_ADIC = json.load(f)

        button_text = button.replace("<button>", "")
        if button_text in BUTTON_URL_ADIC:
            url = BUTTON_URL_ADIC[button_text]
            html, markdown, screenshot = await get_info(url)

            # Save screenshot for debugging or future use
            if screenshot:
                image_folder = "images/"
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                with open(os.path.join(image_folder, f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"), "wb") as f:
                    f.write(base64.b64decode(screenshot))

            response_buttons = extract_links_with_text(html)
            response_content = markdown
            
            if response_content:
                response = f"The web information is:\n\n{response_content}\n\n"
            else:
                response = "The information of the current page is not accessible\n\n"
            
            response += "Clickable buttons are wrapped in <button> tag" + response_buttons
            return response
        else:
            return "The button can not be clicked, please retry a new button!"
    except Exception as e:
        return f"An error occurred in visit_page: {e}"

async def get_all_tools():
    """
    Returns a list of all available tools, including local and MCP tools.
    """
    local_tools = [
        generate_sql_query,
        execute_sql_query,
        deep_research,
        rag_answer_query,
        visit_url,
        visit_page,
    ]
    
    # Initialize MCP client and load tools
    try:
        with open('mcp_config.json', 'r') as f:
            config = json.load(f)
        
        mcp_client = MultiServerMCPClient(config.get("mcpServers", {}))
        mcp_tools = await mcp_client.get_tools()
        return local_tools + mcp_tools
    except Exception as e:
        print(f"Failed to load MCP tools: {e}")
        return local_tools
