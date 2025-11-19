# COVID-19 Data Helper

## 文件描述

*   `app.py`: Flask Web 应用程序，提供用户界面来输入查询并显示结果。
*   `covid_data.db`: 存储 COVID-19 数据的 SQLite 数据库。
*   `langgraph_agent.py`: 使用 LangGraph 和 OpenAI 构建一个代理，该代理可以利用工具来回答有关 COVID-19 数据的问题。
*   `requirements.txt`: 项目所需的依赖项。
*   `robust_merge.py`: 将 2020-2023 年的所有每日 COVID-19 报告 CSV 文件合并到一个 SQLite 数据库中，并处理列名的变化。
*   `tools.py`: 定义了 LangChain 代理可以使用的所有工具，这些工具调用 `data_analyzer` 中的函数。

## 新增：文献数据库 RAG 工具

系统新增两个工具用于“理论性问题”的回答（RAG）：
- `rag_ingest_csv(csv_path, index_name="cord_idx", limit=0)`: 将包含列`title, abstract, doi, url`的CSV入库至Redis向量索引。
- `rag_answer_query(query, index_name="cord_idx", k=5)`: 基于向量检索从文献库中召回相关内容，并用LLM生成回答。

这些工具已自动注册到主智能体（`langgraph_agent.py`），当用户提出理论性/背景性问题时，智能体会优先使用RAG工具进行回答。

### 环境变量
- `OPENAI_API_KEY`：OpenAI兼容接口Key
- `OPENAI_API_BASE`：OpenAI兼容接口Base URL（可选）
- `DASHSCOPE_API_KEY`：用于`deep_research`工具（可选）
- `REDIS_URL`：Redis地址，默认`redis://localhost:6379`
- `HF_EMBEDDING_MODEL`：HuggingFace向量模型，默认`BAAI/bge-large-zh-v1.5`
- `RAG_LLM_MODEL`：RAG回答所用LLM，默认`gpt-4o-mini`

### 论文RAG入库示例
可选方案 A：一次性预处理（推荐）

```
python scripts/pre_ingest_literature.py --csv_path metadata.csv --index_name cord_idx --limit 1000
```

可选方案 B：运行时调用工具

```
rag_ingest_csv(csv_path="metadata.csv", index_name="cord_idx", limit=1000)
```


