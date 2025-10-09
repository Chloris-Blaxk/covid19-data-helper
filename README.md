# COVID-19 Data Helper

This project provides a set of tools to analyze COVID-19 data.

## File Descriptions

*   `agent.py`: 解析自然语言查询并调用相应的数据分析功能。
*   `app.py`: Flask Web 应用程序，提供用户界面来输入查询并显示结果。
*   `covid_data.db`: 存储 COVID-19 数据的 SQLite 数据库。
*   `data_analyzer.py`: 提供用于分析 COVID-19 数据的功能，例如绘图和计算统计数据。
*   `data_retriever.py`: 从 SQLite 数据库中查询 COVID-19 数据。
*   `langgraph_agent.py`: 使用 LangGraph 和 OpenAI 构建一个代理，该代理可以利用工具来回答有关 COVID-19 数据的问题。
*   `requirements.txt`: 项目所需的依赖项。
*   `robust_merge.py`: 将 2020-2023 年的所有每日 COVID-19 报告 CSV 文件合并到一个 SQLite 数据库中，并处理列名的变化。
*   `SYSTEM_ARCHITECTURE_CN.md`: 系统的中文架构文档。
*   `tools.py`: 定义了 LangChain 代理可以使用的所有工具，这些工具调用 `data_analyzer` 中的函数。
*   `WORKFLOW_DOCUMENTATION_CN.md`: 工作流程的中文文档。
*   `templates/index.html`: Flask 应用程序的前端 HTML 模板。
