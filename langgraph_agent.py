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

# Global variable to cache tools
CACHED_TOOLS = None

async def get_cached_tools():
    """
    Initializes and caches the tools on the first call.
    Returns the cached tools on subsequent calls.
    """
    global CACHED_TOOLS
    if CACHED_TOOLS is None:
        CACHED_TOOLS = await get_all_tools()
    return CACHED_TOOLS

# 2. Define the nodes

# The agent node
async def call_model(state):
    logger.info("\n---CALLING MODEL---")
    messages = state['messages']
    
    # Truncate messages to preserve the initial query and the most recent context
    if len(messages) > 10:
        messages = [messages[0]] + messages[-9:]

    logger.info(f"Prompt: {messages}")

    # Add a system message to guide the model
    system_message = HumanMessage(
        content="""
        You are a multi-disciplinary expert data analyst and researcher. Your primary goal is to provide comprehensive analysis and answers based on user queries.

        **CRITICAL INSTRUCTIONS:**

        1.  **Analyze the User's Query:** First, determine the nature of the query. Is it a data analysis question, a plotting request, a literature research question, or a complex open-ended question?

        2.  **Execute the Correct Workflow:**
            -   **For Plotting/Charting:** If the user asks for a "plot", "chart", "graph", or "picture" of data, use the appropriate tool from the `mcp-server-chart` server to generate the image.
            -   **For Data Analysis:** For questions about specific numbers or statistics from the database, use the SQL tools (`generate_sql_query` and `execute_sql_query`).
            -   **For Literature Research:** For specific scientific or historical questions, use the `rag_answer_query` tool.
            -   **For Deep Research:** If a query is complex and requires in-depth analysis beyond the other tools, you MUST first ask the user for confirmation (e.g., "This query requires deep research, which may take some time. Do you want to proceed?"). Only if they agree, call the `deep_research` tool. If they decline, provide a basic answer using other tools or your own knowledge.

        3.  **Synthesize the Final Answer:**
            -   **This is the most important step.** After a tool returns information, you MUST use that information to formulate a comprehensive, human-readable answer.
            -   **DO NOT** simply state that you found information or repeat the raw tool output.
            -   If a tool returns a URL to an image, present it to the user.
            -   **Your final response should directly answer the user's original question**, using the tool's output as your source material. If a tool returns "not_found" or an empty result, inform the user that you could not find the information.
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
    
    # Get tools from cache
    tools = await get_cached_tools()
    model_with_tools = model.bind_tools(tools)
    
    try:
        response = await model_with_tools.ainvoke([system_message] + messages)
        logger.info(f"Model response: {response}")
        return {"messages": [response]}
    except IndexError:
        logger.error("IndexError caught: The model's response was likely empty.")
        error_message = HumanMessage(content="Sorry, I encountered an issue processing your request. The model returned an empty response.")
        return {"messages": [error_message]}

# The tool node using the new ToolNode class
async def call_tool(state):
    """
    Wrapper for the tool node to correctly handle the output.
    """
    logger.info("\n---CALLING TOOL---")
    tool_calls = state['messages'][-1].tool_calls
    logger.info(f"Tool calls: {tool_calls}")
    
    # Get tools from cache
    tools = await get_cached_tools()
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
    
    final_content = final_state['messages'][-1].content

    # If the last message is our specific error message, just return it.
    if final_content == "Sorry, I encountered an issue processing your request. The model returned an empty response.":
        return final_content, ""  # No tool calls to log in this case

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
                log_content = f"Image URL: {content}"
            else:
                log_content = f"Content: {content}"

            tool_calls_log.append(
                f"Tool Result (for {message.tool_call_id}):\n{log_content}"
            )

    return final_content, "\n\n".join(tool_calls_log)

async def main():
    # Example usage
    query1 = "Generate a plot of the top 5 countries by confirmed cases."
    result1 = await run_agent(query1)
    print(f"Query: {query1}\nResult: {result1}\n")

if __name__ == '__main__':
    asyncio.run(main())
