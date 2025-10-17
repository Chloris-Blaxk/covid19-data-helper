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
import dashscope
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain.chains import RetrievalQA

load_dotenv()

# Pydantic models for tool inputs

class NaturalLanguageQueryInput(BaseModel):
    query: str = Field(..., description="The natural language query to convert to SQL.")

class SQLQueryInput(BaseModel):
    query: str = Field(..., description="The SQL query to execute.")

class DeepResearchInput(BaseModel):
    query: str = Field(..., description="The research question for in-depth analysis.")

# RAG inputs
class RAGIngestInput(BaseModel):
    csv_path: str = Field(..., description="Path to a CSV file containing columns: title, abstract, doi, url.")
    index_name: str = Field("cord_idx", description="Redis vector index name.")
    limit: int = Field(0, description="Optional limit of rows to ingest; 0 means all.")

class RAGQueryInput(BaseModel):
    query: str = Field(..., description="User question to be answered with literature-augmented QA.")
    index_name: str = Field("cord_idx", description="Redis vector index name to search.")
    k: int = Field(5, description="Top-K documents to retrieve.")

# Tool definitions

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
    Performs in-depth research on a given topic using the qwen-deep-research model.
    It internally handles a two-step process: first, generating clarifying questions,
    and second, conducting the research based on the initial query.
    The final, comprehensive report is returned as a string.
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

# ---------- RAG (Literature) Tools ----------
def _get_vector_store(index_name: str) -> RedisVectorStore:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    embedding_model = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    redis_config = RedisConfig(index_name=index_name, redis_url=redis_url)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return RedisVectorStore(embeddings=embeddings, config=redis_config)

@tool(args_schema=RAGIngestInput)
def rag_ingest_csv(csv_path: str, index_name: str = "cord_idx", limit: int = 0) -> str:
    """
    Ingests literature metadata CSV into a Redis vector index for RAG.
    Required columns: title, abstract, doi, url.
    """
    if not os.path.exists(csv_path):
        return json.dumps({"status": "error", "message": f"CSV not found: {csv_path}"})

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Failed to read CSV: {e}"})

    required_cols = {"title", "abstract", "doi", "url"}
    if not required_cols.issubset(set(df.columns)):
        return json.dumps({
            "status": "error",
            "message": f"CSV must contain columns: {sorted(list(required_cols))}"
        })

    df = df.dropna(subset=["abstract"]).reset_index(drop=True)
    if limit and limit > 0:
        df = df.head(limit)

    texts = []
    metadatas = []
    for _, row in df.iterrows():
        text = f"{row['title']}\n{row['abstract']}"
        texts.append(text)
        metadatas.append({
            "title": row.get("title", ""),
            "doi": row.get("doi", ""),
            "url": row.get("url", "")
        })

    try:
        store = _get_vector_store(index_name)
        ids = store.add_texts(texts=texts, metadatas=metadatas)
        return json.dumps({
            "status": "ok",
            "index": index_name,
            "ingested": len(ids)
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Vector ingest failed: {e}"})

@tool(args_schema=RAGQueryInput)
def rag_answer_query(query: str, index_name: str = "cord_idx", k: int = 5) -> str:
    """
    Answers theoretical questions by retrieving related literature chunks from Redis and synthesizing with LLM.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    if not api_key:
        return json.dumps({"status": "error", "message": "OPENAI_API_KEY is not set."})

    try:
        store = _get_vector_store(index_name)
        retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": max(1, int(k))})
        llm = ChatOpenAI(model="gemini-2.5-pro", temperature=0, api_key=api_key, base_url=base_url)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
        result = qa.invoke({"query": query})
        # LangChain RetrievalQA may return dict with 'result' or the formatted output
        if isinstance(result, dict) and "result" in result:
            return result["result"]
        return str(result)
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
        deep_research,
        rag_ingest_csv,
        rag_answer_query,
    ]
    
    return local_tools + mcp_tools
