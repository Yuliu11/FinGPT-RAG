"""
LangGraph 图定义文件
定义 Agentic RAG 工作流的核心图结构
"""

import os
import json
from typing import TypedDict, List, Literal, Optional
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END

# 自动寻找当前文件所在目录的父目录下的 .env
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# 增加检查逻辑（不打印任何敏感信息）
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("警告：未找到有效的 API Key，请检查 .env 文件")
else:
    print("✓ 成功加载 API Key")

# 健壮性检查：仅打印非敏感信息
model_name = os.getenv("MODEL_NAME")
if model_name:
    print(f"使用模型: {model_name}")


# 定义图状态
class GraphState(TypedDict):
    """图状态定义"""
    question: str  # 用户问题
    generation: str  # 模型回答
    documents: List[Document]  # 搜索到的文档列表
    grade: Optional[str]  # 文档相关性评分结果 ("yes" 或 "no")
    expanded_keywords: Optional[List[str]]  # 扩展后的关键词列表


def get_llm():
    """
    初始化 ChatOpenAI 模型（支持 DeepSeek）
    
    Returns:
        ChatOpenAI 实例
    """
    api_base = os.getenv("OPENAI_API_BASE")
    model_name = os.getenv("MODEL_NAME", "deepseek-chat")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("未设置 OPENAI_API_KEY 环境变量")
    
    if not api_base:
        raise ValueError("未设置 OPENAI_API_BASE 环境变量")
    
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=api_base,
        temperature=0.7,
    )
    
    return llm


def get_vector_store():
    """
    初始化向量存储（单例模式）
    
    Returns:
        QdrantVectorStore 实例
    """
    # 使用模块级变量缓存向量存储
    if not hasattr(get_vector_store, '_vector_store'):
        project_root = Path(__file__).parent.parent
        vector_db_path = project_root / "data" / "vector_db"
        
        # 初始化嵌入模型（与入库时保持一致）
        embeddings = HuggingFaceEmbeddings(
            model_name='shibing624/text2vec-base-chinese',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 初始化 Qdrant 客户端
        client = QdrantClient(path=str(vector_db_path))
        
        # 初始化向量存储
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="financial_documents",
            embedding=embeddings
        )
        
        get_vector_store._vector_store = vector_store
    
    return get_vector_store._vector_store


def expand_query(state: GraphState) -> GraphState:
    """
    查询扩写节点：将用户原始问题转化为一组 3-5 个专业财务关键词
    
    Args:
        state: 图状态
        
    Returns:
        更新后的图状态（包含 expanded_keywords）
    """
    question = state["question"]
    
    # 使用 LLM 生成专业财务关键词
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个资深审计师。请将用户的口语化提问转化为中国 A 股财报中的专业术语。

要求：
1. 输出 3-5 个专业财务关键词
2. 关键词应该是财报中常见的专业术语
3. 必须保留原问题中的公司名称（如果存在）
4. 关键词用中文逗号分隔
5. 示例：
   - "风险" → "可能面对的风险、风险提示、经营挑战"
   - "赚钱能力" → "营业收入、净利润、毛利率"
   - "比亚迪的营收" → "比亚迪、营业收入、主营业务收入"

只输出关键词，用中文逗号分隔，不要添加其他解释。"""),
        ("human", """用户问题：{question}

请生成 3-5 个专业财务关键词（用中文逗号分隔）：""")
    ])
    
    try:
        messages = prompt.format_messages(question=question)
        response = llm.invoke(messages)
        keywords_text = response.content if hasattr(response, 'content') else str(response)
        
        # 解析关键词（用中文逗号或英文逗号分隔）
        keywords = [k.strip() for k in keywords_text.replace('，', ',').split(',') if k.strip()]
        
        # 确保关键词数量在 3-5 个之间
        if len(keywords) < 3:
            # 如果关键词太少，添加原问题作为补充
            keywords.append(question)
        elif len(keywords) > 5:
            keywords = keywords[:5]
        
        # 如果解析失败，使用原问题作为唯一关键词
        if not keywords:
            keywords = [question]
        
    except Exception as e:
        print(f"查询扩写出错: {str(e)}")
        # 出错时使用原问题作为关键词
        keywords = [question]
    
    return {
        "question": question,
        "documents": state.get("documents", []),
        "generation": state.get("generation", ""),
        "grade": state.get("grade"),
        "expanded_keywords": keywords
    }


def retrieve(state: GraphState) -> GraphState:
    """
    检索节点：基于扩展后的关键词列表从 Qdrant 向量数据库检索相关文档
    对每个关键词进行检索，然后合并去重，取相似度最高的前 5-10 个块
    
    Args:
        state: 图状态
        
    Returns:
        更新后的图状态（包含 documents）
    """
    question = state["question"]
    expanded_keywords = state.get("expanded_keywords", [question])  # 如果没有扩展关键词，使用原问题
    
    # 初始化向量存储
    try:
        vector_store = get_vector_store()
    except Exception as e:
        print(f"初始化向量存储失败: {str(e)}")
        return {
            "documents": [],
            "question": question,
            "generation": state.get("generation", ""),
            "grade": state.get("grade"),
            "expanded_keywords": expanded_keywords
        }
    
    # 对每个关键词进行检索
    all_docs_with_scores = []
    doc_ids_seen = set()  # 用于去重（基于文档内容）
    
    for keyword in expanded_keywords:
        try:
            # 使用 similarity_search_with_score 获取带分数的文档
            docs_with_scores = vector_store.similarity_search_with_score(keyword, k=10)
            
            # 添加到总列表（去重）
            for doc, score in docs_with_scores:
                # 使用文档内容的哈希作为唯一标识
                doc_hash = hash(doc.page_content)
                if doc_hash not in doc_ids_seen:
                    doc_ids_seen.add(doc_hash)
                    all_docs_with_scores.append((doc, score))
        
        except Exception as e:
            print(f"检索关键词 '{keyword}' 时出错: {str(e)}")
            continue
    
    # 按相似度分数排序（分数越低表示相似度越高，如果是余弦相似度，则分数越高越好）
    # 注意：similarity_search_with_score 返回的分数可能是距离（越小越好）或相似度（越大越好）
    # 这里假设是距离，所以按分数升序排序
    all_docs_with_scores.sort(key=lambda x: x[1])
    
    # 取前 10 个文档（或更少，如果总数不够）
    final_docs = [doc for doc, score in all_docs_with_scores[:10]]
    
    return {
        "documents": final_docs,
        "question": question,
        "generation": state.get("generation", ""),
        "grade": state.get("grade"),
        "expanded_keywords": expanded_keywords
    }


def grade_documents(state: GraphState) -> GraphState:
    """
    文档评分节点：判断检索到的文档是否与问题相关
    使用 LLM 判断文档相关性，输出严格的 JSON 格式 {"score": "yes/no"}
    
    Args:
        state: 图状态
        
    Returns:
        更新后的图状态（包含 grade 字段）
    """
    documents = state["documents"]
    question = state["question"]
    
    # 如果没有文档，直接返回 "no"
    if not documents:
        return {
            "question": question,
            "documents": documents,
            "generation": state.get("generation", ""),
            "grade": "no",
            "expanded_keywords": state.get("expanded_keywords")
        }
    
    # 格式化文档内容（取前 1000 字符用于判断）
    doc_text = "\n\n".join([doc.page_content[:1000] for doc in documents[:3]])
    
    # 使用 LLM 判断文档相关性
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的文档相关性判断助手。请判断检索到的文档是否与用户问题相关。

要求：
1. 严格输出 JSON 格式：{"score": "yes"} 或 {"score": "no"}
2. 如果文档内容能够回答用户问题，输出 "yes"
3. 如果文档内容与问题无关或信息不足，输出 "no"
4. 只输出 JSON，不要添加任何其他文字"""),
        ("human", """用户问题：{question}

检索到的文档内容：
{documents}

请判断文档是否与问题相关，严格输出 JSON 格式：{{"score": "yes"}} 或 {{"score": "no"}}""")
    ])
    
    try:
        messages = prompt.format_messages(question=question, documents=doc_text)
        response = llm.invoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # 尝试解析 JSON
        try:
            # 提取 JSON 部分（可能包含 markdown 代码块）
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                parsed = json.loads(json_str)
                score = parsed.get("score", "no").lower()
            else:
                score = "no"
        except (json.JSONDecodeError, KeyError, ValueError):
            # 如果 JSON 解析失败，尝试从文本中查找 "yes" 或 "no"
            content_lower = content.lower()
            if '"score": "yes"' in content_lower or '"score":\'yes\'' in content_lower or 'score":"yes' in content_lower:
                score = "yes"
            else:
                score = "no"
        
        # 确保 score 是 "yes" 或 "no"
        score = "yes" if score == "yes" else "no"
        
    except Exception as e:
        # 如果判断失败，默认返回 "no"
        print(f"文档评分出错: {str(e)}")
        score = "no"
    
    return {
        "question": question,
        "documents": documents,
        "generation": state.get("generation", ""),
        "grade": score,
        "expanded_keywords": state.get("expanded_keywords")
    }


def should_continue(state: GraphState) -> Literal["transform", "generate"]:
    """
    条件判断函数：根据 grade_documents 的结果决定下一步
    这个函数会被 LangGraph 的条件边调用
    
    Args:
        state: 图状态
        
    Returns:
        "transform" 如果文档不相关，需要重写问题
        "generate" 如果文档相关，可以生成答案
    """
    grade = state.get("grade", "no")
    
    # 如果评分是 "no"，需要改写问题
    if grade == "no":
        return "transform"
    
    # 如果评分是 "yes"，生成答案
    return "generate"


def transform_query(state: GraphState) -> GraphState:
    """
    问题改写节点：如果文档不相关，生成 3 个更精准的财务搜索关键词
    
    Args:
        state: 图状态
        
    Returns:
        更新后的图状态（包含改写后的 question，使用第一个关键词）
    """
    question = state["question"]
    
    # 使用 LLM 生成 3 个财务搜索关键词
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的财务搜索优化助手。请基于用户问题生成 3 个更精准的财务搜索关键词。

要求：
1. 关键词应该更具体、更专业
2. 专注于财务指标、公司名称、年份等关键信息
3. 输出格式：用中文逗号分隔的 3 个关键词
4. 只输出关键词，不要添加其他解释"""),
        ("human", """用户原始问题：{question}

请生成 3 个更精准的财务搜索关键词（用中文逗号分隔）：""")
    ])
    
    try:
        messages = prompt.format_messages(question=question)
        response = llm.invoke(messages)
        keywords_text = response.content if hasattr(response, 'content') else str(response)
        
        # 解析关键词（用中文逗号或英文逗号分隔）
        keywords = [k.strip() for k in keywords_text.replace('，', ',').split(',') if k.strip()]
        
        # 如果成功生成关键词，使用第一个作为新的查询
        if keywords:
            transformed_question = keywords[0]
        else:
            transformed_question = question
    except Exception as e:
        print(f"问题改写出错: {str(e)}")
        transformed_question = question
    
    return {
        "question": transformed_question,
        "documents": state.get("documents", []),
        "generation": state.get("generation", ""),
        "grade": state.get("grade")
    }


def generate_answer(state: GraphState) -> GraphState:
    """
    生成节点：基于检索到的文档生成答案
    必须基于上下文回答，如果涉及计算，请一步步显示计算过程
    支持工具调用（如增长率计算）
    
    Args:
        state: 图状态
        
    Returns:
        更新后的图状态（包含 generation）
    """
    question = state["question"]
    documents = state["documents"]
    
    # 格式化文档上下文
    context_parts = []
    for i, doc in enumerate(documents, 1):
        content = doc.page_content
        metadata = doc.metadata
        company = metadata.get("company", "未知公司")
        year = metadata.get("year", "未知年份")
        context_parts.append(f"[文档 {i}] 来源：{company} {year}年\n{content}")
    
    context = "\n\n".join(context_parts)
    
    # 导入工具
    from app.tools.finance_tools import growth_rate_calc
    
    # 定义工具
    growth_rate_tool = StructuredTool.from_function(
        func=growth_rate_calc,
        name="growth_rate_calc",
        description="计算增长率。参数：current_value(当前值), previous_value(上期值), period(周期，可选，默认'年度')"
    )
    
    # 使用 LLM 生成答案（绑定工具）
    llm = get_llm()
    llm_with_tools = llm.bind_tools([growth_rate_tool])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的金融文档分析助手。请基于提供的文档内容回答用户的问题。

重要约束：
1. **必须基于上下文回答**：只能使用提供的文档内容回答问题，不能编造信息
2. **计算过程**：如果涉及计算（如增长率、比例等），必须一步步显示计算过程
3. **工具使用**：如需计算增长率，可以使用 growth_rate_calc 工具
4. **准确性**：如果文档中没有相关信息，请明确说明"文档中未找到相关信息"
5. **专业性**：回答要准确、专业，可以引用具体的数字和数据

文档内容：
{context}"""),
        ("human", "{question}")
    ])
    
    try:
        messages = prompt.format_messages(context=context, question=question)
        
        # 调用 LLM（支持工具调用）
        response = llm_with_tools.invoke(messages)
        
        # 检查是否有工具调用
        tool_calls = getattr(response, 'tool_calls', None) if hasattr(response, 'tool_calls') else None
        
        if tool_calls:
            # 处理工具调用
            tool_results = []
            for tool_call in tool_calls:
                # 兼容不同的工具调用格式
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                elif hasattr(tool_call, 'name'):
                    tool_name = tool_call.name
                    tool_args = tool_call.args if hasattr(tool_call, 'args') else {}
                else:
                    continue
                
                if tool_name == "growth_rate_calc":
                    try:
                        result = growth_rate_calc(**tool_args)
                        tool_results.append(f"工具调用结果（增长率计算）：{json.dumps(result, ensure_ascii=False)}")
                    except Exception as e:
                        tool_results.append(f"工具调用失败：{str(e)}")
            
            # 如果有工具调用结果，将其添加到上下文中，再次调用 LLM
            if tool_results:
                tool_result_text = "\n".join(tool_results)
                context_with_tools = f"{context}\n\n工具计算结果：\n{tool_result_text}"
                
                prompt_with_tools = ChatPromptTemplate.from_messages([
                    ("system", """你是一个专业的金融文档分析助手。请基于提供的文档内容和工具计算结果回答用户的问题。

重要约束：
1. **必须基于上下文回答**：只能使用提供的文档内容和工具计算结果回答问题
2. **计算过程**：如果涉及计算，必须一步步显示计算过程
3. **准确性**：如果文档中没有相关信息，请明确说明

文档内容和工具结果：
{context}"""),
                    ("human", "{question}")
                ])
                
                messages = prompt_with_tools.format_messages(context=context_with_tools, question=question)
                response = llm.invoke(messages)
        
        generation = response.content if hasattr(response, 'content') else str(response)
        
    except Exception as e:
        print(f"生成答案出错: {str(e)}")
        generation = f"生成答案时出错：{str(e)}"
    
    return {
        "question": question,
        "documents": documents,
        "generation": generation,
        "grade": state.get("grade")
    }


def create_rag_graph():
    """
    创建 RAG 工作流图
    
    Returns:
        编译后的图对象
    """
    # 创建状态图
    workflow = StateGraph(GraphState)
    
    # 添加节点
    workflow.add_node("expand_query", expand_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate_answer)
    
    # 设置入口点
    workflow.set_entry_point("expand_query")
    
    # 添加边
    # expand_query -> retrieve
    workflow.add_edge("expand_query", "retrieve")
    
    # retrieve -> grade_documents
    workflow.add_edge("retrieve", "grade_documents")
    
    # grade_documents 条件分支
    workflow.add_conditional_edges(
        "grade_documents",
        should_continue,  # 使用 should_continue 函数作为条件函数
        {
            "transform": "transform_query",  # 不相关 -> 改写问题
            "generate": "generate"  # 相关 -> 生成答案
        }
    )
    
    # transform_query -> retrieve（重新检索）
    workflow.add_edge("transform_query", "retrieve")
    
    # generate -> END
    workflow.add_edge("generate", END)
    
    # 编译图
    app = workflow.compile()
    
    return app
