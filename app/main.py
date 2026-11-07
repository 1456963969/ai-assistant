"""FastAPI 服务：意图路由 -> 混合检索/联网兜底 -> 双层记忆 -> 生成（模型可降级）"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from app.agent_graph import run_agent
from app.llm import generate
from app.memory import Memory
from app.retriever import HybridRetriever
from app.router import decide

load_dotenv()

app = FastAPI(title="AI Assistant — 知识库优先的智能问答服务")

SYSTEM_PROMPT = """你是私人 AI 助理。回答规则：
1. 优先依据提供的上下文作答，关键结论后标注来源编号，如 [1]、[2]；
2. 上下文来自联网搜索时，答案末尾用一行列出参考链接；
3. 上下文为空或与问题无关时，回答"知识库中未找到相关内容"，不要编造；
4. 用中文简洁回答。
{context_note}
"""

memory = None
retriever = None


def get_retriever():
    global retriever
    if retriever is None:
        retriever = HybridRetriever()
    return retriever


def get_memory():
    global memory
    if memory is None:
        memory = Memory()
    return memory


class Question(BaseModel):
    query: str
    session_id: str = "default"
    top_k: int = 5
    mode: str = "fast"   # fast：规则路由单次响应；agent：LangGraph ReAct 自主规划


class Answer(BaseModel):
    answer: str
    route: str          # direct / knowledge / web / none / agent
    model: str          # 实际使用的模型（降级时标识）
    sources: list
    steps: list = []    # agent 模式下 ReAct 的工具调用轨迹


@app.post("/chat", response_model=Answer)
def chat(q: Question):
    m = get_memory()
    history = m.short_history(q.session_id)

    # ---- 慢通道：LangGraph ReAct，模型自主规划多步工具调用 ----
    if q.mode == "agent":
        try:
            answer, trace = run_agent(q.query, history)
            m.add(q.session_id, "user", q.query)
            m.add(q.session_id, "assistant", answer)
            return Answer(answer=answer, route="agent",
                          model="qwen-plus（ReAct）", sources=[], steps=trace)
        except Exception:
            pass  # Agent 链路异常时降级回规则路由，保障可用性

    # ---- 快通道：规则路由，单次响应 ----
    r = get_retriever()
    route, payload = decide(r, q.query)

    if route == "direct":
        context_note = "用户在打招呼，请简短友好地回应。"
        sources: list = []
    elif route == "knowledge":
        chunks = r.retrieve(q.query, top_k=q.top_k)
        context_note = "参考上下文：\n" + r.format_context(chunks)
        sources = r.build_sources(chunks)
    elif route == "web":
        lines = [f"[{i}] {w['title']}：{w['content']}（{w['url']}）"
                 for i, w in enumerate(payload, 1)]
        context_note = "以下来自联网搜索：\n" + "\n\n".join(lines)
        sources = [{"ref": i, "url": w["url"]} for i, w in enumerate(payload, 1)]
    else:
        context_note = "（知识库与联网均无可用内容，请如实告知用户）"
        sources = []

    answer, model_used = generate(
        SYSTEM_PROMPT.format(context_note=context_note), q.query, history
    )
    m.add(q.session_id, "user", q.query)
    m.add(q.session_id, "assistant", answer)

    return Answer(answer=answer, route=route, model=model_used, sources=sources, steps=[])


@app.get("/history/{session_id}")
def history(session_id: str):
    """长期记忆：跨会话查询完整历史"""
    return get_memory().long_history(session_id)


@app.get("/health")
def health():
    return {"status": "ok"}
