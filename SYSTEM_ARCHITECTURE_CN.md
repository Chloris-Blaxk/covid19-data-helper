# 系统架构与数据流详解

本文档旨在详细解释基于 LangGraph 的 COVID-19 数据分析 Agent 的系统架构、各组件职责以及核心数据处理流程。

## 1. 系统架构图

```
+----------------------+      (HTTP Request)      +---------------------+      (Python Call)      +-------------------------+
|   用户浏览器 (UI)    | <----------------------> |  Web 服务器 (Flask) | ----------------------> |   Agent 核心 (LangGraph)  |
| templates/index.html |                          |       app.py        |                         |   langgraph_agent.py    |
+----------------------+                          +---------------------+                         +------------+------------+
                                                                                                                | (LLM Decides)
                                                                                                                |
                                                                                                                v
+-------------------------+      (Function Call)     +-------------------------+      (SQL Query)      +-------------------------+
|  工具集 (LangChain Tools) | <---------------------- |    数据服务层 (Python)    | ----------------------> |     数据库 (SQLite)     |
|        tools.py         |                          | data_analyzer.py        |                         |      covid_data.db      |
+-------------------------+                          | data_retriever.py       |                         +-------------------------+
```

## 2. 组件详解

整个系统被设计为一系列松散耦合的模块，每个模块都有明确的职责。

-   **用户界面 (UI - `templates/index.html`)**
    -   **技术**: HTML, CSS
    -   **职责**: 提供一个简单的 Web 表单，作为用户与系统交互的入口。用户在此输入自然语言问题。它使用 Jinja2 模板语言来动态展示后端返回的结果（无论是文本还是图片）。

-   **Web 服务器 (`app.py`)**
    -   **技术**: Flask
    -   **职责**: 作为前端 UI 和后端 Agent 之间的桥梁。它监听 `http://127.0.0.1:5001` 上的网络请求。当用户提交查询时，它接收该请求，调用 Agent 核心进行处理，并将处理结果传回给 UI 进行渲染。

-   **Agent 核心 (`langgraph_agent.py`)**
    -   **技术**: LangGraph, LangChain, OpenAI
    -   **职责**: 这是系统的“大脑”。它不执行具体的业务逻辑，而是负责**决策**。它接收到来自 Web 服务器的查询后：
        1.  将查询和对话历史打包，发送给大语言模型（LLM）。
        2.  LLM 根据其对查询的理解和对可用工具的认知，决定是直接回答问题，还是调用一个或多个工具。
        3.  如果需要调用工具，LLM 会生成一个包含工具名称和参数的指令。
        4.  `LangGraph` 根据 LLM 的决策，将流程导向工具执行节点或结束流程。

-   **工具集 (`tools.py`)**
    -   **技术**: LangChain Tools
    -   **职责**: 将底层的业务逻辑函数（来自 `data_analyzer.py`）封装成标准化的“工具”，供 Agent 调用。每个工具都有清晰的**名称**、**描述**和**参数定义**，这是 LLM 能够理解并正确使用它们的关键。

-   **数据服务层 (`data_analyzer.py`, `data_retriever.py`)**
    -   **技术**: Pandas, Matplotlib
    -   **职责**: 封装了所有具体的业务逻辑和数据操作。
        -   `data_retriever.py`: 负责与数据库 `covid_data.db` 进行最底层的交互，执行 SQL 查询。
        -   `data_analyzer.py`: 负责更复杂的分析任务，如计算病死率、生成图表等。它调用 `data_retriever` 来获取原始数据。

-   **数据存储 (`covid_data.db`)**
    -   **技术**: SQLite
    -   **职责**: 持久化存储所有从原始 CSV 文件中清洗和整合后的 COVID-19 数据。所有的数据分析都以此数据库为基础。

## 3. 核心数据流说明

下面我们以一个具体的用户查询为例，来说明整个系统的数据流：“**为我绘制一张美国每日新增病例的图表**”。

1.  **用户输入**: 用户在 `index.html` 的输入框中键入问题，并点击“提问”。浏览器向 `app.py` 发送一个包含该问题的 HTTP POST 请求。

2.  **请求传递**: `app.py` 接收到请求，从请求体中提取出问题字符串 `"为我绘制一张美国每日新增病例的图表"`，然后调用 `langgraph_agent.run_agent()` 函数。

3.  **Agent 决策 (LangGraph 流程开始)**:
    a.  `langgraph_agent` 将问题包装成一个 `HumanMessage`，作为图（Graph）的初始输入。
    b.  流程进入第一个节点 `call_model`。该节点调用 OpenAI LLM，并告知 LLM 它可以使用的工具列表（来自 `tools.py`）及其描述。
    c.  LLM 分析问题，理解到用户的意图是“可视化”、“每日数据”和“美国”。它匹配到 `plot_daily_cases_for_country` 这个工具最符合需求，并识别出 `country_name` 参数应为 `"US"`。
    d.  LLM 的返回值不是一个直接的答案，而是一个**工具调用指令**，内容大致为：`tool_calls=[{'name': 'plot_daily_cases_for_country', 'args': {'country_name': 'US'}}]`。

4.  **条件路由**:
    a.  流程离开 `call_model` 节点。图的条件路由 `should_continue` 检查到 LLM 的返回中包含工具调用指令。
    b.  因此，路由决定将流程导向 `action` 节点（即 `ToolNode`）。

5.  **工具执行**:
    a.  `ToolNode` 接收到工具调用指令，自动执行 `tools.py` 中的 `plot_daily_cases_for_country(country_name='US')` 函数。
    b.  该函数内部会进一步调用 `data_analyzer.py` 中的同名函数。
    c.  `data_analyzer` 函数构建相应的 SQL 查询语句，并调用 `data_retriever.py` 来从 `covid_data.db` 获取数据。
    d.  获取数据后，`data_analyzer` 使用 Matplotlib 生成图表，并将其保存到 `static/plots/daily_cases_US.png`。
    e.  最后，该函数**返回**图片的路径字符串：`'static/plots/daily_cases_US.png'`。

6.  **结果反馈与最终回答**:
    a.  工具执行的结果（图片路径）被包装成一个 `ToolMessage`，然后流程重新回到 `agent` 节点 (`call_model`)。
    b.  `call_model` 再次调用 LLM，但这次的输入包含了原始问题和工具的执行结果。LLM 看到工具已经成功执行并返回了图片路径，于是决定生成一个最终的、面向用户的回答，这个回答的内容就是工具返回的路径。
    c.  此时，LLM 的返回不再包含工具调用指令。

7.  **流程结束与响应返回**:
    a.  条件路由 `should_continue` 检查到新的返回中没有工具调用，于是将流程导向 `END`，图的执行结束。
    b.  `langgraph_agent.run_agent()` 函数接收到图的最终状态，并将最终的回答（图片路径）包装成一个字典 `{'type': 'image', 'content': 'static/plots/daily_cases_US.png'}` 返回给 `app.py`。

8.  **前端渲染**: `app.py` 将这个结果字典传递给 `index.html` 模板。模板中的 Jinja2 逻辑检测到 `type` 是 `image`，于是渲染出一个 `<img>` 标签，其 `src` 指向该图片路径。

9.  **用户看到结果**: 最终，用户的浏览器页面刷新，显示出生成的图表。
