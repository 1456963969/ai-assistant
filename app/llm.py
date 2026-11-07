"""答案生成：主模型 qwen-plus，异常自动降级 deepseek-chat 兜底"""
import os
from dotenv import load_dotenv

load_dotenv()


def generate(system_prompt: str, user_text: str, history: list) -> tuple[str, str]:
    """history: [(role, content), ...] 来自短期记忆
    返回 (答案文本, 实际使用的模型标识)
    """
    messages = [("system", system_prompt)] + list(history) + [("human", user_text)]
    try:
        from langchain_community.chat_models.tongyi import ChatTongyi
        llm = ChatTongyi(
            model="qwen-plus",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        return llm.invoke(messages).content, "qwen-plus"
    except Exception as primary_err:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise primary_err  # 未配置备用模型，原样抛出
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        payload = [{"role": "system" if r == "system" else r, "content": c} for r, c in messages]
        resp = client.chat.completions.create(model="deepseek-chat", messages=payload)
        return resp.choices[0].message.content, "deepseek-chat（降级）"
