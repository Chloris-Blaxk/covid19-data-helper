from flask import Flask, render_template, request
from langgraph_agent import run_agent
import os
import asyncio
import markdown2
import traceback

app = Flask(__name__)

# Ensure the static/plots directory exists so plots can be saved
if not os.path.exists('static/plots'):
    os.makedirs('static/plots')

@app.route('/', methods=['GET', 'POST'])
async def index():
    result_html = None
    tool_calls_log = None
    if request.method == 'POST':
        query = request.form['query']
        try:
            # Await the async agent function directly
            markdown_report, tool_calls_log_str = await run_agent(query)
            # Convert the Markdown report to HTML
            result_html = markdown2.markdown(markdown_report, extras=["fenced-code-blocks", "tables"])
            tool_calls_log = tool_calls_log_str
        except Exception as e:
            error_message = f"An error occurred: {e}"
            print(error_message)
            traceback.print_exc()
            result_html = f"<p><strong>发生错误:</strong> {e}</p>"
            
    return render_template('index.html', result_html=result_html, tool_calls_log=tool_calls_log)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
