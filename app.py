from flask import Flask, render_template, request
from langgraph_agent import run_agent
import os
import markdown2

app = Flask(__name__)

# Ensure the static/plots directory exists so plots can be saved
if not os.path.exists('static/plots'):
    os.makedirs('static/plots')

@app.route('/', methods=['GET', 'POST'])
def index():
    result_html = None
    if request.method == 'POST':
        query = request.form['query']
        try:
            # The agent now returns a complete Markdown report
            markdown_report = run_agent(query)
            # Convert the Markdown report to HTML
            result_html = markdown2.markdown(markdown_report, extras=["fenced-code-blocks", "tables"])
        except Exception as e:
            result_html = f"<p><strong>发生错误:</strong> {e}</p>"
            
    return render_template('index.html', result_html=result_html)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
