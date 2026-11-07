"""LangChain Tool 化：知识库检索与联网搜索注册为标准工具
工具化后可被 Router 编排，也可无缝接入 LangGraph 等 Agent 框架扩展。
"""
import os
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


@tool
def knowledge_base_search(query: str, top_k: int = 5) -> str:
    """Search the private knowledge base. Use for domain/internal questions."""
    from app.retriever import HybridRetriever
    r = HybridRetriever()
    chunks = r.retrieve(query, top_k=top_k)
    if not chunks:
        return "（知识库中未找到相关内容）"
    return "\n\n".join(
        f"[{i}]（来源：{os.path.basename(c.metadata.get('source', ''))}）\n{c.page_content}"
        for i, c in enumerate(chunks, 1)
    )


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the public web via Tavily. Use when the knowledge base has no answer."""
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return "（未配置联网搜索）"
    from tavily import TavilyClient
    client = TavilyClient(key)
    result = client.search(query, max_results=max_results)
    return "\n\n".join(
        f"[{i}] {w['title']}：{w['content']}（{w['url']}）"
        for i, w in enumerate(result.get("results", []), 1)
    )


AVAILABLE_TOOLS = [knowledge_base_search, web_search]
