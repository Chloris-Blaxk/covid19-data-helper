# COVID-19 Data Helper

## 文件描述

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

## TODO
- * 数据挖掘/深度研究的 api/mcp
- * 处理数据异常（规则匹配 | 基于模型）
- [点到为止，实现而先不深入]文献数据库/文本向量数据库 + RAG -> 理论问题的回答（agent平台的rag工具）
- [↑]让系统具有web搜索的能力（qwen-webagent等）
- [x]让系统具有云端执行代码的能力（优先级低）
- [-]让系统可以进行文件上传和处理（前端层面、rag库）
- [难度较大，优先级较低]优化系统流程
- [可以尝试]上下文容量问题（预处理、结构化数据、中间层）


