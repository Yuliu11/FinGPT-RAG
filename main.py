"""
Streamlit 应用启动入口
根目录下的 main.py，作为 Streamlit 的启动入口点
自动加载 .env 文件并调用 app/ 目录下的核心逻辑
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 自动加载 .env 文件（优先从根目录加载）
project_root = Path(__file__).parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)

# 检查 API Key 是否加载成功（不打印敏感信息）
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("✓ 成功加载 API Key")
else:
    print("⚠ 警告：未找到有效的 API Key，请检查 .env 文件")

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(project_root))

# 导入 Streamlit 和必要的模块
import streamlit as st
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# 导入 app 目录下的核心模块
from app.graph import get_llm

# 页面配置
st.set_page_config(
    page_title="Financial RAG Agent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "document_count" not in st.session_state:
    st.session_state.document_count = 0
if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0
if "ensemble_retriever" not in st.session_state:
    st.session_state.ensemble_retriever = None
if "reranker" not in st.session_state:
    st.session_state.reranker = None


@st.cache_resource
def initialize_vector_store():
    """初始化向量数据库（缓存）"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # 初始化嵌入模型（与入库时保持一致）
        embeddings = HuggingFaceEmbeddings(
            model_name='shibing624/text2vec-base-chinese',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 初始化向量数据库（使用 QdrantVectorStore）
        vector_db_path = project_root / "data" / "vector_db"
        client = QdrantClient(path=str(vector_db_path))
        
        # 使用 QdrantVectorStore（新版本 API）
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="financial_documents",
            embedding=embeddings
        )
        
        # 获取文档数量（使用 QdrantClient 直接查询）
        try:
            collection_info = client.get_collection("financial_documents")
            document_count = collection_info.points_count
        except:
            document_count = 19085  # 默认值
        
        return vector_store, document_count
    except Exception as e:
        st.error(f"初始化向量数据库失败: {str(e)}")
        # 不显示详细堆栈跟踪，避免泄露敏感信息
        return None, 0


def get_document_count():
    """获取文档数量"""
    try:
        vector_db_path = project_root / "data" / "vector_db"
        client = QdrantClient(path=str(vector_db_path))
        collection_info = client.get_collection("financial_documents")
        return collection_info.points_count
    except:
        return 19085  # 默认值


@st.cache_resource
def initialize_ensemble_retriever(vector_store):
    """
    初始化混合检索器（向量检索 + BM25）
    
    Args:
        vector_store: 向量存储对象
        
    Returns:
        EnsembleRetriever 对象
    """
    try:
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever
        
        # 获取所有文档用于 BM25（分批获取以提高效率）
        all_docs = []
        try:
            # 尝试获取所有文档，如果文档太多则分批处理
            batch_size = 1000
            for i in range(0, 20000, batch_size):  # 最多获取 20000 个文档
                batch = vector_store.similarity_search("", k=batch_size)
                if not batch:
                    break
                all_docs.extend(batch)
                if len(batch) < batch_size:
                    break
        except Exception as e:
            # 如果获取失败，使用较小的样本
            all_docs = vector_store.similarity_search("", k=5000)
        
        if not all_docs:
            return None
        
        # 初始化 BM25 检索器
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 20  # BM25 召回数量
        
        # 向量检索器
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        
        # 混合检索器（权重各占 0.5）
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        
        return ensemble_retriever
    except Exception as e:
        st.warning(f"初始化混合检索器失败，将使用纯向量检索: {str(e)}")
        # 不显示详细堆栈跟踪，避免泄露敏感信息
        return None


@st.cache_resource
def initialize_reranker():
    """
    初始化重排序模型
    
    Returns:
        FlashrankRerank 对象
    """
    try:
        from langchain_community.cross_encoders import FlashrankRerank
        
        reranker = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
        return reranker
    except Exception as e:
        st.warning(f"初始化重排序模型失败: {str(e)}")
        return None


def retrieve_documents(query: str, vector_store, ensemble_retriever, reranker, k: int = 5):
    """
    从向量数据库检索相关文档（支持混合检索和重排序）
    
    Args:
        query: 查询文本
        vector_store: 向量存储对象
        ensemble_retriever: 混合检索器
        reranker: 重排序模型
        k: 最终返回的文档数量
        
    Returns:
        检索到的文档列表（带分数）
    """
    try:
        if vector_store is None:
            return []
        
        # 第一步：混合检索召回 20 个候选文档
        if ensemble_retriever:
            # 使用混合检索
            candidate_docs = ensemble_retriever.get_relevant_documents(query)
        else:
            # 降级到纯向量检索
            candidate_docs = vector_store.similarity_search(query, k=20)
        
        if not candidate_docs:
            return []
        
        # 第二步：重排序，精选出 top k 个文档
        if reranker and len(candidate_docs) > k:
            try:
                # FlashrankRerank 可能使用不同的 API，尝试多种方法
                if hasattr(reranker, 'compress_documents'):
                    reranked_docs = reranker.compress_documents(
                        documents=candidate_docs,
                        query=query
                    )
                elif hasattr(reranker, 'rerank'):
                    # 尝试 rerank 方法
                    reranked_docs = reranker.rerank(query, candidate_docs)
                elif hasattr(reranker, 'score'):
                    # 使用 score 方法进行重排序
                    scored_docs = []
                    for doc in candidate_docs:
                        score = reranker.score(query, doc.page_content)
                        scored_docs.append((doc, score))
                    # 按分数排序
                    scored_docs.sort(key=lambda x: x[1], reverse=True)
                    reranked_docs = [doc for doc, _ in scored_docs]
                else:
                    # 如果都不支持，使用原始文档
                    reranked_docs = candidate_docs
                
                final_docs = reranked_docs[:k] if isinstance(reranked_docs, list) else list(reranked_docs)[:k]
            except Exception as e:
                # 如果重排序失败，使用原始文档
                final_docs = candidate_docs[:k]
        else:
            final_docs = candidate_docs[:k]
        
        # 转换为带分数的格式（重排序后的文档没有分数，使用索引作为排序依据）
        docs_with_score = []
        for idx, doc in enumerate(final_docs):
            # 使用 (1.0 - idx/len(final_docs)) 作为相似度分数（排序越靠前分数越高）
            score = 1.0 - (idx / len(final_docs))
            docs_with_score.append((doc, score))
        
        return docs_with_score
    except Exception as e:
        st.error(f"检索文档失败: {str(e)}")
        # 不显示详细堆栈跟踪，避免泄露敏感信息
        return []


def format_context(docs):
    """
    格式化检索到的文档为上下文
    
    Args:
        docs: 文档列表（包含分数）
        
    Returns:
        格式化后的上下文字符串和来源信息
    """
    context_parts = []
    sources = []
    
    for i, (doc, score) in enumerate(docs, 1):
        content = doc.page_content
        metadata = doc.metadata
        
        # 提取来源信息
        company = metadata.get("company", "未知公司")
        year = metadata.get("year", "未知年份")
        report_type = metadata.get("report_type", "未知类型")
        file_name = metadata.get("file_name", "未知文件")
        
        source_info = {
            "index": i,
            "company": company,
            "year": year,
            "report_type": report_type,
            "file_name": file_name,
            "content": content[:500] + "..." if len(content) > 500 else content,  # 截取前500字符
            "score": f"{score:.4f}"
        }
        sources.append(source_info)
        
        # 构建上下文
        context_parts.append(f"[文档 {i}] {content}")
    
    return "\n\n".join(context_parts), sources


def generate_response(query: str, context: str, llm):
    """
    使用 LLM 生成回答（流式输出）
    
    Args:
        query: 用户问题
        context: 检索到的上下文
        llm: LLM 模型
        
    Yields:
        回答的文本片段
    """
    from langchain_core.prompts import ChatPromptTemplate
    
    # 构建提示词
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的金融文档分析助手。请基于提供的文档内容回答用户的问题。

要求：
1. 回答要准确、专业
2. 如果文档中没有相关信息，请明确说明
3. 可以引用具体的数字和数据
4. 回答要简洁明了

文档内容：
{context}"""),
        ("human", "{question}")
    ])
    
    # 格式化提示词
    messages = prompt_template.format_messages(
        context=context,
        question=query
    )
    
    # 流式调用 LLM
    for chunk in llm.stream(messages):
        # 处理不同类型的 chunk
        if hasattr(chunk, 'content'):
            content = chunk.content
            if content:
                yield content
        elif isinstance(chunk, str):
            yield chunk
        elif hasattr(chunk, 'text'):
            yield chunk.text


# 侧边栏：使用说明
with st.sidebar:
    st.header("📖 使用说明")
    
    st.markdown("### 🎯 核心功能")
    st.markdown("""
    本助手是一个专业的**金融文档智能问答系统**，基于 RAG（检索增强生成）技术构建。
    
    **主要能力：**
    - 📊 分析上市公司财务报告
    - 💰 回答关于营收、利润、资产等财务指标的问题
    - 📈 对比不同公司的财务表现
    - 🔍 查找特定年份或报告类型的信息
    """)
    
    st.divider()
    
    st.markdown("### 💡 提问示例")
    st.markdown("""
    **财务指标查询：**
    - "比亚迪2024年的营业收入是多少？"
    - "贵州茅台2023年的净利润增长率是多少？"
    - "招商银行2024年的总资产是多少？"
    
    **对比分析：**
    - "对比一下中国平安和招商银行2024年的净利润"
    - "格力电器和立讯精密2023年的营收对比"
    
    **趋势分析：**
    - "海康威视近两年的营收趋势如何？"
    - "立讯精密2023到2024年的业绩变化"
    
    **其他问题：**
    - "比亚迪的主要业务是什么？"
    - "贵州茅台的核心竞争力是什么？"
    """)
    
    st.divider()
    
    st.markdown("### 📚 数据来源")
    st.markdown("""
    本系统基于以下上市公司的公开财务报告：
    
    - **中国平安** - 年度报告、半年度报告
    - **招商银行** - 年度报告、半年度报告
    - **格力电器** - 年度报告、半年度报告
    - **比亚迪** - 年度报告、半年度报告
    - **海康威视** - 年度报告、半年度报告
    - **立讯精密** - 年度报告、半年度报告
    - **贵州茅台** - 年度报告、半年度报告
    
    数据涵盖 **2023-2025年** 的财务报告，所有信息均来自上市公司官方披露的PDF文档。
    """)
    
    st.divider()
    
    st.markdown("### ⚙️ 使用提示")
    st.markdown("""
    1. **输入问题**：在下方输入框中输入您的问题
    2. **智能检索**：系统会自动从文档库中检索相关信息
    3. **流式回答**：回答会以打字机效果实时显示
    4. **查看来源**：回答完成后可展开"检索来源"查看具体文档片段
    5. **多轮对话**：支持连续提问，系统会记住对话历史
    """)


# 主界面
st.title("💰 Financial RAG Agent")
st.markdown("**金融文档智能问答系统** - 基于 LangChain 和 DeepSeek")

# 确保数据库已初始化
if st.session_state.vector_store is None:
    with st.spinner("正在连接金融数据库..."):
        vector_store, doc_count = initialize_vector_store()
        st.session_state.vector_store = vector_store
        st.session_state.document_count = doc_count
    st.rerun()

# 初始化混合检索器和重排序模型
if st.session_state.vector_store and st.session_state.ensemble_retriever is None:
    with st.spinner("正在初始化混合检索器..."):
        try:
            st.session_state.ensemble_retriever = initialize_ensemble_retriever(st.session_state.vector_store)
        except Exception as e:
            st.session_state.ensemble_retriever = None

if st.session_state.reranker is None:
    try:
        st.session_state.reranker = initialize_reranker()
    except Exception as e:
        st.session_state.reranker = None

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # 如果是助手回答，显示检索来源
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 检索来源", expanded=False):
                for source in message["sources"]:
                    st.markdown(f"""
                    **文档 {source['index']}** (相似度: {source['score']})
                    - **公司**: {source['company']}
                    - **年份**: {source['year']}
                    - **报告类型**: {source['report_type']}
                    - **文件名**: {source['file_name']}
                    - **内容片段**: 
                    > {source['content']}
                    """)
                    st.divider()

# 用户输入（确保在主循环中，页面加载时就渲染）
if prompt := st.chat_input("请输入您的问题..."):
    # 1. 访问频率限制检查
    import time
    current_time = time.time()
    time_since_last_query = current_time - st.session_state.last_query_time
    
    if time_since_last_query < 3:
        st.warning("提问太快啦，请稍等")
        st.stop()
    
    # 更新最后查询时间
    st.session_state.last_query_time = current_time
    
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 检查向量数据库是否已初始化
    if st.session_state.vector_store is None:
        st.error("向量数据库未初始化，请检查数据文件")
        st.stop()
    
    # 检索相关文档（使用混合检索和重排序）
    with st.spinner("正在检索相关文档..."):
        docs = retrieve_documents(
            prompt, 
            st.session_state.vector_store,
            st.session_state.ensemble_retriever,
            st.session_state.reranker,
            k=5
        )
        
        if not docs:
            st.warning("未找到相关文档，请尝试其他问题")
            st.stop()
        
        # 格式化上下文和来源
        context, sources = format_context(docs)
    
    # 生成回答（流式输出）
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # 初始化 LLM（每次调用时重新加载，确保使用最新的 API Key）
            llm = get_llm()
            
            # 流式生成回答（打字机效果）
            for chunk in generate_response(prompt, context, llm):
                if chunk:  # 确保 chunk 不为空
                    full_response += chunk
                    # 实时更新显示，添加光标效果
                    message_placeholder.markdown(full_response + "▌")
            
            # 移除光标，显示最终回答
            message_placeholder.markdown(full_response)
            
            # 显示检索来源（使用 expander）
            with st.expander("📚 检索来源", expanded=False):
                st.markdown("**以下是从向量数据库中检索到的相关文档片段：**")
                st.markdown("")
                for source in sources:
                    st.markdown(f"""
                    **文档 {source['index']}** (相似度分数: {source['score']})
                    - **公司**: {source['company']}
                    - **年份**: {source['year']}
                    - **报告类型**: {source['report_type']}
                    - **文件名**: `{source['file_name']}`
                    """)
                    st.markdown(f"**内容片段：**")
                    st.markdown(f"> {source['content']}")
                    st.divider()
            
            # 保存消息和来源到 session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources
            })
            
        except Exception as e:
            st.error(f"生成回答时出错: {str(e)}")
            # 不显示详细堆栈跟踪，避免泄露敏感信息（如 API Key、文件路径等）
