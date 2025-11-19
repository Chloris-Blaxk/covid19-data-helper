from flask import Flask, render_template, request, jsonify
from langgraph_agent import run_agent
import os
import asyncio
import traceback
import logging
import datetime
from flask_cors import CORS

app = Flask(__name__)
# Allow requests from the default Vue dev server origin
CORS(app, resources={r"/chat": {"origins": "http://localhost:8080"}})

# In-memory store for conversation history
conversation_history = []

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
agent_logger.propagate = False # Prevent duplicate logging
# --- End Logging Setup ---


# Ensure the static/plots directory exists so plots can be saved
if not os.path.exists('static/plots'):
    os.makedirs('static/plots')

@app.route('/')
def index():
    return "<h1>COVID-19 Data Helper AI Assistant API is running.</h1>"

@app.route('/chat', methods=['POST'])
async def chat():
    global conversation_history
    data = request.get_json()
    
    messages = data.get('messages', [])
    if not messages:
        return jsonify({"error": "No messages provided"}), 400
        
    logger.info(f"Received messages: {messages}")

    # Use the history sent from the client for context
    conversation_history = messages

    try:
        # Create a query string from the history for the agent.
        conversation_query = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

        markdown_report, _ = await run_agent(conversation_query)
        
        logger.info(f"Generated Report:\n{markdown_report}")

        return jsonify({"response": markdown_report})

    except Exception as e:
        error_message = f"An error occurred: {e}"
        detailed_traceback = traceback.format_exc()
        logger.error(f"{error_message}\n{detailed_traceback}")
        return jsonify({"error": "Sorry, an error occurred on the server."}), 500

if __name__ == '__main__':
    # Run on a different port to avoid conflict with EpiInsight's own backend
    app.run(debug=True, port=5001)
