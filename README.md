![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)
![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-red.svg)
![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-orange.svg)
# FinGPT-RAG: 基于 DeepSeek 的万级金融财报智能分析助手

> 🔗 **GitHub Repository**  
> https://github.com/Yuliu11/FinGPT-RAG


## 🌟 项目简介

本项目是一款专为金融从业者设计的 RAG (Retrieval-Augmented Generation) 助手。通过集成 DeepSeek 大模型与 Qdrant 高性能向量数据库，实现了对海量上市公司财报的精准检索与深度分析。项目核心处理了 20 份核心财报，生成了超过 60,000 个高质量知识块。

**FinGPT-RAG** 是一个专门针对 A 股上市公司财报设计的智能问答系统。它不仅能检索信息，还能像专业分析师一样思考。

## 🌟 核心进化 (Agentic RAG)
不同于传统的线性 RAG，本项目引入了 **LangGraph** 驱动的智能体逻辑：
- **自动化语义扩写**：自动将“风险”、“赚钱能力”等口语转化为“可能面对的风险”、“净利润/毛利率”等财报专业术语，极大提升检索精度。
- **多路检索与逻辑评分**：Agent 会评估搜索结果质量，若信息不足将自动触发“查询重写”逻辑，拒绝幻觉。
- **精准处理复杂表格**：针对 PDF 财报中的非结构化数据，通过正则清洗与 Markdown 转换，实现财务数值的精准提取。

## 🚀 技术架构

- **数据层**：采用递归字符分块，针对金融报表进行语义边界优化，并结合 正则表达式（Regex） 清洗页眉、页脚及目录噪音，强化 Markdown 表格解析。
- **Agent 推理层**：引入 LangGraph 构建状态机，实现自动化任务拆解。
全量查询扩写 (Query Expansion)：利用 LLM 将口语化提问实时转化为财报专业术语（如将“风险”映射为“可能面对的风险”、“经营挑战”），解决语义鸿沟。
智能评分员 (Grader)：对检索到的文档块进行相关性评分，自动剔除无关噪音，若信息不足则自动重写 Query 重新检索。
- **检索层**：Qdrant 本地化向量存储，支持万级数据秒级召回，通过扩写关键词实现多路并行检索。
- **展示层**：Streamlit 构建的响应式前端，支持流式输出与信源溯源。

## ✨ 技术难点解决

- **复杂表格语义对齐**：通过优化 Overlap 算法，解决了财务报表中科目与数值断裂的痛点，确保检索块具备完整财务语义。
- **本地模型适配**：实现了 DeepSeek 原生接口的异步调用，并通过 Prompt Engineering 强化了模型在处理复杂财务对比时的准确度。
- **金融术语对齐**：通过在检索前强制执行“语义扩写”，解决了用户提问与财报黑话不一致的痛点。
- **复杂表格语义完整性**：优化 Overlap 算法，确保财务报表中科目与数值在分块时不被截断。
- **减少模型幻觉**：Agent 具备“自我反思”能力，当检索到的内容无法回答问题时，系统会诚实告知缺失信息，而非编造数字。

## 🛠️ 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

将 `.env.example` 更名为 `.env` 并填写你的 API Key：

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 DEEPSEEK_API_KEY
```

### 3. 初始化数据

运行数据导入脚本处理 PDF 文档：

```bash
python scripts/ingest.py
```

### 4. 启动应用

运行 Streamlit 应用：

```bash
streamlit run main.py
```

## 📁 项目结构

```
Financial_RAG_Agent/
├── app/                    # 核心应用代码
│   ├── nodes/             # RAG 工作流节点
│   ├── tools/             # 工具函数
│   ├── graph.py           # LangGraph 图定义
│   ├── main.py            # Streamlit 应用（app目录）
│   └── pdf_processor.py   # PDF 处理模块
├── data/                   # 数据目录
│   ├── raw/               # 原始 PDF 文档
│   └── vector_db/         # 向量数据库存储
├── scripts/                # 脚本目录
│   └── ingest.py          # 数据导入脚本
├── config/                 # 配置文件
├── main.py                # Streamlit 启动入口（根目录）
├── requirements.txt        # Python 依赖
├── .env.example           # 环境变量模板
└── README.md              # 项目说明文档
```

> 说明：`data/` 目录默认不进入版本控制（见 `.gitignore`），使用者需自行在本地准备财报 PDF 并生成向量数据库。

## 🔧 技术栈

- **LangChain**: LLM 应用框架，提供文档处理和检索能力
- **LangGraph**: 工作流编排框架
- **Qdrant**: 高性能向量数据库
- **Streamlit**: Web 界面框架
- **DeepSeek**: 大语言模型 API
- **HuggingFace**: 本地嵌入模型（text2vec-base-chinese）

## 📊 数据来源

本系统基于以下上市公司的公开财务报告，来源于巨潮资讯公开网站：

- **中国平安** - 年度报告、半年度报告
- **招商银行** - 年度报告、半年度报告
- **格力电器** - 年度报告、半年度报告
- **比亚迪** - 年度报告、半年度报告
- **海康威视** - 年度报告、半年度报告
- **立讯精密** - 年度报告、半年度报告
- **贵州茅台** - 年度报告、半年度报告

数据涵盖 **2023-2025年** 的财务报告，所有信息均来自上市公司官方披露的 PDF 文档。

## 💡 使用示例

### 财务指标查询
- "比亚迪2024年的营业收入是多少？"
- "贵州茅台2023年的净利润增长率是多少？"
- "招商银行2024年的总资产是多少？"

### 对比分析
- "对比一下中国平安和招商银行2024年的净利润"
- "格力电器和立讯精密2023年的营收对比"

### 趋势分析
- "海康威视近两年的营收趋势如何？"
- "立讯精密2023到2024年的业绩变化"

## 🔒 安全与隐私说明（Security & Privacy）

- 本仓库不包含任何 API Key、账号密码等敏感信息；所有密钥均通过环境变量方式加载（`.env` 文件已被 `.gitignore` 忽略）。

- 原始财报 PDF 文件与本地向量数据库默认仅存放在本地目录（如 `data/raw/`、`data/vector_db/`），不会提交到公共仓库。

- 本项目仅在 **推理（inference-only）模式** 下调用大语言模型：文档通过本地检索提供上下文，不会被用于模型训练。

- 请避免在 Issue、日志或截图中粘贴任何密钥、请求头或内部文档内容。

## 📝 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📎 项目链接

- GitHub 仓库：https://github.com/Yuliu11/FinGPT-RAG
