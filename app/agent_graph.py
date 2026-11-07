"""LangGraph ReAct Agent（"慢通道"）：模型自主规划、多步调用工具后生成答案

与 Router 规则路由（"快通道"）互补：
- 简单问题走快路径，单次响应，低延迟低成本；
- 复杂/多步问题进入 ReAct 循环（Reason → Act → Observe），
  由模型自主决定调用哪些工具、调用几次，直到信息充分再作答。
"""
import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

from app.tools import AVAILABLE_TOOLS

load_dotenv()

AGENT_PROMPT = """你是私人 AI 助理。你可以调用以下工具回答问题：
1. knowledge_base_search —— 私有知识库检索，领域问题优先使用；
2. web_search —— 联网搜索，知识库信息不足或需要实时信息时补充。

回答规则：
- 关键结论后标注来源编号，如 [1]、[2]；
- 工具返回无相关内容时，如实回答"知识库中未找到相关内容"，不要编造；
- 复杂问题可组合调用多个工具（如先查知识库再联网对比），信息充分后再作答；
- 用中文简洁回答。
"""

_model = None
_graph = None


def _get_model():
    global _model
    if _model is None:
        from langchain_community.chat_models.tongyi import ChatTongyi
        _model = ChatTongyi(
            model=os.getenv("AGENT_MODEL", "qwen-plus"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
    return _model


def get_graph():
    """懒加载 ReAct 状态图（复用 tools.py 注册的两个标准工具）"""
    global _graph
    if _graph is None:
        _graph = create_react_agent(_get_model(), AVAILABLE_TOOLS, prompt=AGENT_PROMPT)
    return _graph


def run_agent(query: str, history: list) -> tuple[str, list[str]]:
    """运行 ReAct 循环
    history: [(role, content), ...] 来自短期记忆（Redis）
    返回 (答案, 工具调用轨迹)
    """
    g = get_graph()
    # 角色名对齐：assistant -> ai
    msgs = [("ai" if r == "assistant" else "human", c) for r, c in history]
    result = g.invoke({"messages": msgs + [("human", query)]})

    trace: list[str] = []
    for m in result["messages"]:
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            trace.extend(tc.get("name", "?") for tc in tool_calls)
    answer = result["messages"][-1].content
    return answer, trace
