"""Router 意图路由：优先私有知识库，命中不足走联网搜索兜底"""
import os


def decide(retriever, query: str, threshold: float | None = None):
    """返回 (route, payload)
    route: direct（闲聊直答）/ knowledge（知识库命中）/ web（联网兜底）/ none（无可用来源）
    """
    if threshold is None:
        threshold = float(os.getenv("KB_THRESHOLD", "0.35"))

    greeting = ("你好", "您好", "hi", "hello", "在吗", "你是谁")
    if any(query.strip().lower().startswith(g) for g in greeting) and len(query.strip()) <= 8:
        return "direct", []

    top = retriever.top_relevance(query)
    if top >= threshold:
        return "knowledge", [top]

    key = os.getenv("TAVILY_API_KEY")
    if key:
        from tavily import TavilyClient
        client = TavilyClient(key)
        result = client.search(query, max_results=3)
        web_ctx = [
            {"title": r["title"], "content": r["content"], "url": r["url"]}
            for r in result.get("results", [])
        ]
        if web_ctx:
            return "web", web_ctx
    return "none", []
