# COVID-19 数据分析 Agent：开发工作流文档

本文档旨在详细阐述 COVID-19 数据分析 Agent 的完整思考过程与开发流程。

## 1. 项目目标

最初的需求是构建一个 Agent，通过分析全球 2020-2023 年的 COVID-19 每日报告大数据集，来辅助研究人员的工作。该 Agent 需要具备理解用户问题、检索数据、执行分析以及调用工具等能力。

---

## 2. 第一阶段：基础数据管道与 v1 版 Agent

第一阶段的目标是搭建核心的后端数据处理能力，并实现一个功能性的、基于规则的初版 Agent。

### 2.1. 数据探索与整合

- **挑战**: 原始数据分散在超过 1100 个独立的 CSV 文件中，每日一个。直接查询这些文件效率极低，且不同文件间的列名存在不一致的问题。
- **思考过程**: 最佳方案是将所有 CSV 文件整合到一个统一的、可查询的数据库中。SQLite 是一个理想的选择，因为它轻量、无服务器，并且得到了 Python 的良好支持。
- **实施步骤**:
    1.  **初版合并脚本 (`merge_covid_data.py`)**: 编写了第一个脚本，尝试遍历所有 CSV，用 Pandas 加载它们，然后追加到一个 SQLite 表中。
    2.  **问题调试**: 首次尝试失败，抛出 `ValueError: cannot assemble with duplicate keys` 错误。经过排查，问题根源在于不一致的列名（例如 `Last Update` vs. `Last_Update`）在被标准化处理后变成了同一个名称，导致 DataFrame 中出现重复列。
    3.  **优化脚本 (`robust_merge.py`)**: 根据用户的反馈和对一个示例脚本的分析，开发了一个更健壮的版本。该版本在数据处理前，就明确地处理了列名的各种变体，确保了数据的一致性。
    4.  **成果**: 成功创建了 `covid_data.db`，一个包含了全部清洗后数据的综合数据库。

### 2.2. 核心工具开发

- **思考过程**: Agent 需要一种方式来与数据库交互并执行计算。最佳实践是将这些功能点分离到独立的模块中。
- **实施步骤**:
    1.  **数据检索器 (`data_retriever.py`)**: 创建了一个模块，其中包含一个核心函数 `query_covid_data`。该函数负责执行任意 SQL 查询并返回 Pandas DataFrame，从而将数据库的连接逻辑抽象出来。
    2.  **数据分析器 (`data_analyzer.py`)**: 在检索器的基础上构建。该模块包含了一系列用于满足用户具体分析需求的高阶函数，例如：
        - `plot_top_countries_by_confirmed_cases` (绘制 top N 国家确诊病例图)
        - `plot_daily_cases_for_country` (绘制某国每日病例曲线图)
        - `calculate_mortality_rate` (计算病死率)
        - `calculate_daily_growth_rate` (计算日增长率)

### 2.3. 初版 Agent (v1) 与 Web 界面

- **思考过程**: 工具齐备后，需要一个“代理”来解析用户的查询。对于初版，一个简单的、基于规则的方法已经足够。随后，为了提升可用性，用户要求增加一个 Web 界面。
- **实施步骤**:
    1.  **基于规则的 Agent (`agent.py`)**: 创建了一个脚本，它接收字符串形式的查询，然后使用 `if/elif/else` 结构和关键词匹配（如 "plot", "top countries", "mortality rate"）来决定调用 `data_analyzer` 中的哪个函数。
    2.  **Web 用户界面 (Flask)**:
        - 安装 Flask 库。
        - 创建 `app.py` 作为 Web 服务器后端。
        - 创建 `templates/index.html` 作为前端页面，包含一个简单的表单和结果展示区。
        - **为 Web 化重构**: 为了让 Web 应用能展示结果，对 `data_analyzer` 和 `agent` 脚本进行了重构。所有函数不再向控制台 `print` 信息，而是 `return` 结果（文本字符串或图片路径）。这使得 Flask 应用可以捕获这些输出，并将其渲染到 HTML 模板中。

---

## 3. 第二阶段：使用 LangChain 和 LangGraph 进行架构重构

用户要求使用更现代的 AI 框架进行重构，以获得更好的智能性和扩展性。

### 3.1. 重构动机

- **挑战**: 基于规则的 Agent 非常脆弱。它无法理解用户问题的细微变化（例如 "show me a graph" vs. "plot a chart"），也无法处理更复杂的请求。
- **思考过程**: LangChain 和 LangGraph 正是为解决此类问题而设计的。它们允许将一个大语言模型（LLM）作为 Agent 的“大脑”。LLM 能够理解用户的真实*意图*，并智能地选择正确的工具及参数。LangGraph 则提供了一种强大的方式，将 Agent 的决策过程构建为一个状态机。

### 3.2. 环境准备与工具定义

- **实施步骤**:
    1.  **安装依赖**: 安装了 `langchain`, `langgraph`, 和 `langchain-openai`。
    2.  **工具抽象化 (`tools.py`)**:
        - 创建了新文件 `tools.py`。
        - `data_analyzer` 中的所有公开函数都被 LangChain 的 `@tool` 装饰器所封装。
        - **关键一步：为每个工具添加了详尽的文档字符串（docstring）。** LLM 正是依靠这些描述来理解每个工具的功能。
        - 使用 Pydantic 模型为每个工具的输入参数定义了清晰的类型（例如 `country_name: str`），这使得 LLM 能够构建出有效的函数调用。

### 3.3. 构建 LangGraph Agent

- **思考过程**: Agent 的工作流可以被建模为一个图：用户提问 -> Agent (LLM) 思考 -> 可能调用一个工具 -> 工具执行 -> 流程重复，直到 Agent 收集到足够信息以回答问题。
- **实施步骤 (`langgraph_agent.py`)**:
    1.  **定义状态**: 定义了一个 `AgentState` 字典，用于追踪消息列表（即对话历史）。
    2.  **定义节点**:
        - `call_model`: “代理”节点。它接收当前状态，并调用 LLM 来决定下一步行动。
        - `ToolNode`: 使用了 `langgraph.prebuilt` 中更现代的 `ToolNode`。这个节点自动处理由 `call_model` 节点选择的工具的执行过程。
    3.  **定义边**: 创建了一个名为 `should_continue` 的条件路由。在 `call_model` 节点之后，这个路由会检查 LLM 是否请求了工具调用。
        - 如果 **是**，图的状态就流转到 `ToolNode`。
        - 如果 **否** (LLM 已生成最终答案)，图的状态就流转到 `END`，流程结束。
    4.  **编译图**: 将节点和边组装成一个 `StateGraph` 并将其编译。

### 3.4. 最终集成与 API 密钥管理

- **实施步骤**:
    1.  **更新 Web 应用**: 修改 `app.py`，使其导入并调用新的 `langgraph_agent.run_agent` 函数。
    2.  **API 密钥处理**:
        - Agent 的运行需要 `OPENAI_API_KEY`。最初，代码里的检查要求用户必须在终端中设置该环境变量。
        - 根据用户的反馈（“我需要一个 .env 文件吗？”），这个流程得到了优化。安装了 `python-dotenv` 库，并添加了代码以自动从 `.env` 文件加载密钥。
        - 创建了 `.env.example` 文件来引导用户完成配置。

---

## 4. 最终成果与运行指南

最终的交付成果是一个健壮、智能且用户友好的 Web 应用，由一个 LangGraph Agent 驱动。

- **如何运行**:
    1.  将 `.env.example` 文件复制为 `.env`，并在其中填入你自己的 OpenAI API 密钥。
    2.  运行 `python app.py`。
    3.  在浏览器中打开显示的 URL (默认为 `http://127.0.0.1:5001`)。
