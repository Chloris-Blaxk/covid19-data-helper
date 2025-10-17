from flask import Flask, render_template, request
from langgraph_agent import run_agent
import os
import asyncio
import markdown2
import traceback
import logging
import datetime

app = Flask(__name__)

# --- Logging Setup ---
# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Generate a unique log file name with a timestamp under log directory
log_dir = os.path.join(os.path.dirname(__file__), 'log')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = os.path.join(log_dir, f"app_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Create a file handler and set the formatter
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Create a stream handler to output to the console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Add both handlers to the app's logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Configure the logger for the langgraph_agent module
agent_logger = logging.getLogger('langgraph_agent')
agent_logger.setLevel(logging.INFO)
agent_logger.addHandler(file_handler)
agent_logger.addHandler(stream_handler)
# --- End Logging Setup ---


# Ensure the static/plots directory exists so plots can be saved
if not os.path.exists('static/plots'):
    os.makedirs('static/plots')

@app.route('/', methods=['GET', 'POST'])
async def index():
    result_html = None
    tool_calls_log = None
    if request.method == 'POST':
        query = request.form['query']
        logger.info(f"Received query: {query}")
        try:
            # Await the async agent function directly
            markdown_report, tool_calls_log_str = await run_agent(query)
            
            logger.info(f"Generated Report:\n{markdown_report}")
            logger.info(f"Tool Calls Log:\n{tool_calls_log_str}")

            # Convert the Markdown report to HTML
            result_html = markdown2.markdown(markdown_report, extras=["fenced-code-blocks", "tables"])
            tool_calls_log = tool_calls_log_str
        except Exception as e:
            error_message = f"An error occurred: {e}"
            detailed_traceback = traceback.format_exc()
            logger.error(f"{error_message}\n{detailed_traceback}")
            print(error_message)
            traceback.print_exc()
            result_html = f"<p><strong>发生错误:</strong> {e}</p>"
            
    return render_template('index.html', result_html=result_html, tool_calls_log=tool_calls_log)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
