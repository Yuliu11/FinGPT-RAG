import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def test_connection():
    try:
        llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "deepseek-chat"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE")
        )
        response = llm.invoke("你好，请确认你的身份。")
        print(f"✅ 连接成功！AI回复：{response.content}")
    except Exception as e:
        print(f"❌ 连接失败：{str(e)}")

if __name__ == "__main__":
    test_connection()