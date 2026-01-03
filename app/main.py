"""
Streamlit 主应用入口
实现 RAG 问答界面，支持流式输出和检索来源展示
"""

import os
import sys
from pathlib import Path
import streamlit as st
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


@st.cache_resource
def initialize_vector_store():
    """初始化向量数据库（缓存）"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # 初始化嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name='shibing624/text2vec-base-chinese',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 初始化向量数据库
        vector_db_path = project_root / "data" / "vector_db"
        client = QdrantClient(path=str(vector_db_path))
        
        vector_store = Qdrant(
            client=client,
            collection_name="financial_documents",
            embeddings=embeddings
        )
        
        # 获取文档数量
        try:
            collection_info = client.get_collection("financial_documents")
            document_count = collection_info.points_count
        except:
            document_count = 19085  # 默认值
        
        return vector_store, document_count
    except Exception as e:
        st.error(f"初始化向量数据库失败: {str(e)}")
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


def retrieve_documents(query: str, vector_store, k: int = 5):
    """
    从向量数据库检索相关文档
    
    Args:
        query: 查询文本
        vector_store: 向量存储对象
        k: 返回的文档数量
        
    Returns:
        检索到的文档列表
    """
    try:
        docs = vector_store.similarity_search_with_score(query, k=k)
        return docs
    except Exception as e:
        st.error(f"检索文档失败: {str(e)}")
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


# 侧边栏：数据库状态
with st.sidebar:
    st.header("📊 数据库状态")
    
    # 初始化向量数据库
    if st.session_state.vector_store is None:
        with st.spinner("正在加载向量数据库..."):
            vector_store, doc_count = initialize_vector_store()
            st.session_state.vector_store = vector_store
            st.session_state.document_count = doc_count
    
    # 显示文档数量
    st.metric(
        label="文档块数量",
        value=f"{st.session_state.document_count:,}",
        help="当前向量数据库中的文档块总数"
    )
    
    # 刷新按钮
    if st.button("🔄 刷新状态"):
        st.session_state.vector_store = None
        st.session_state.document_count = get_document_count()
        st.rerun()
    
    st.divider()
    
    st.markdown("### 📖 使用说明")
    st.markdown("""
    1. 在下方输入框中输入您的问题
    2. 系统会从金融文档库中检索相关信息
    3. 回答会以流式方式实时显示
    4. 回答完成后可查看检索来源
    """)


# 主界面
st.title("💰 Financial RAG Agent")
st.markdown("**金融文档智能问答系统** - 基于 LangChain 和 DeepSeek")

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

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 检查向量数据库是否已初始化
    if st.session_state.vector_store is None:
        st.error("向量数据库未初始化，请检查数据文件")
        st.stop()
    
    # 检索相关文档
    with st.spinner("正在检索相关文档..."):
        docs = retrieve_documents(prompt, st.session_state.vector_store, k=5)
        
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
            # 初始化 LLM
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
            import traceback
            with st.expander("错误详情"):
                st.code(traceback.format_exc())
