import os
import json
import asyncio
import logging
from typing import TypedDict, Annotated, Sequence
import operator
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from tools import get_all_tools

# Load environment variables from .env file
load_dotenv()

# Configure a logger for this module
logger = logging.getLogger(__name__)

# 1. Define the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Get API key and optional base URL from environment variables
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")

if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable in a .env file.")

# 2. Define the nodes

# The agent node
async def call_model(state):
    logger.info("\n---CALLING MODEL---")
    messages = state['messages']
    
    # Truncate messages to keep the last 10
    if len(messages) > 10:
        messages = messages[-10:]
        
    logger.info(f"Prompt: {messages}")
    
    # Add a system message to guide the model
    system_message = HumanMessage(
        content="""
        You are a multi-disciplinary expert data analyst and researcher. Your primary goal is to provide comprehensive analysis and answers based on user queries, which may span across COVID-19 data analysis and theoretical research based on scientific literature.

        **CRITICAL INSTRUCTIONS:** You MUST follow this workflow. Your tool selection should be based on the user's query type.

        **Workflow for Theoretical/Research Questions:**
        1.  **Literature Review:** If the user's query is theoretical, conceptual, or requires background knowledge from scientific papers (e.g., "What is SARS?", "Compare the variants of COVID-19"), you MUST use the `rag_answer_query` tool to get a well-supported answer from the literature database.
        2.  **Deep Research (Optional):** If the literature search does not provide a complete answer, you may use the `deep_research` tool for a broader web-based search.
        3.  **Synthesize:** Provide a comprehensive answer based on the information gathered.

        **Workflow for Data Analysis Questions:**
        1.  **Formulate SQL:** For queries about specific data points, trends, or statistics (e.g., "confirmed cases in Germany", "top 5 countries by deaths"), use the `generate_sql_query` tool.
        2.  **Execute SQL:** Use the `execute_sql_query` tool to get the data.
        3.  **Visualize Data:** If the user asks for a plot or visualization, you MUST use an available chart generation tool (e.g., `generate_bar_chart`, `generate_line_chart`).
        4.  **Synthesize and Analyze:** Provide a final, comprehensive analysis, explaining the data and charts to answer the user's question.

        **Data Ingestion (As Needed):**
        *   If the user explicitly asks to load new literature data from a CSV file, use the `rag_ingest_csv` tool.

        **IMPORTANT:** Always choose the most appropriate workflow based on the user's query. Do not mix the workflows unnecessarily. The final output MUST be a complete answer or analysis.
        """
    )
    
    # Pass the API key and base_url to the client
    model = ChatOpenAI(
        model="gemini-2.5-pro",
        temperature=0, 
        streaming=False, 
        api_key=api_key,
        base_url=base_url,
        timeout=1200  # Set a 20-minute timeout to prevent connection drops
    )
    
    # Get tools asynchronously
    tools = await get_all_tools()
    model_with_tools = model.bind_tools(tools)
    
    response = await model_with_tools.ainvoke([system_message] + messages)
    logger.info(f"Model response: {response}")
    return {"messages": [response]}

# The tool node using the new ToolNode class
async def call_tool(state):
    """
    Wrapper for the tool node to correctly handle the output.
    """
    logger.info("\n---CALLING TOOL---")
    tool_calls = state['messages'][-1].tool_calls
    logger.info(f"Tool calls: {tool_calls}")
    
    # Get tools asynchronously
    tools = await get_all_tools()
    tool_node = ToolNode(tools)
    
    # Invoke the tool node
    tool_result = await tool_node.ainvoke(state)
    logger.info(f"Tool result: {tool_result}")

    # The result from ToolNode is a dict with a "messages" key
    if isinstance(tool_result, dict) and "messages" in tool_result:
        return {"messages": tool_result["messages"]}
    
    # Handle cases where the result might be a list of messages
    if isinstance(tool_result, list):
        return {"messages": tool_result}

    # Fallback for unexpected tool result format
    raise ValueError(f"Unexpected tool result format: {tool_result}")

# 3. Define the edges
def should_continue(state):
    last_message = state['messages'][-1]
    # If there are no tool calls, we're done
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return "end"
    # Otherwise, continue with tool execution
    else:
        return "continue"

# 4. Assemble the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")

# Compile the graph into a runnable app
app = workflow.compile()

async def run_agent(query: str):
    """
    Runs the LangGraph agent with the given query and returns a Markdown report
    and a log of tool calls.
    """
    inputs = {"messages": [HumanMessage(content=query)]}
    final_state = await app.ainvoke(inputs)
    
    tool_calls_log = []
    for message in final_state['messages']:
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls_log.append(
                    f"Tool Call: {tool_call['name']}\n"
                    f"Arguments: {json.dumps(tool_call['args'], indent=2)}"
                )
        if isinstance(message, ToolMessage):
            content = message.content
            # Check if the content is a URL and format it as a Markdown image
            if isinstance(content, str) and content.startswith('http'):
                # Add to the main content for rendering in the report
                final_state['messages'][-1].content += f"\n\n![Generated Chart]({content})"
                # Also log it correctly
                log_content = f"Image URL: {content}"
            else:
                log_content = f"Content: {content}"

            tool_calls_log.append(
                f"Tool Result (for {message.tool_call_id}):\n{log_content}"
            )

    final_content = final_state['messages'][-1].content
    return final_content, "\n\n".join(tool_calls_log)

async def main():
    # Example usage
    query1 = "Generate a plot of the top 5 countries by confirmed cases."
    result1 = await run_agent(query1)
    print(f"Query: {query1}\nResult: {result1}\n")

if __name__ == '__main__':
    asyncio.run(main())
