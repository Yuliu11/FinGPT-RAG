"""
LangGraph 图定义文件
定义 RAG 工作流的核心图结构
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI

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


# TODO: 实现 LangGraph 图结构
# 示例：
# from langgraph.graph import StateGraph
# 
# def create_rag_graph():
#     llm = get_llm()
#     # 构建图结构
#     graph = StateGraph(...)
#     return graph
