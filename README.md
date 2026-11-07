# AI Assistant — 基于 Agent 的私人 AI 助理服务

知识库优先、联网兜底的智能问答服务：意图路由 + 混合检索 + 重排 + 双层记忆 + 模型自动降级，答案强制附带来源引用。

![运行效果](docs/images/chat-demo.png)

## 架构

```
用户提问 (session_id)
   │
   ▼
Router 路由判断：闲聊问候 → 直接回复
   │              知识库命中（相关度 ≥ 阈值）→ 知识库检索
   │              命中不足且有 Tavily Key → 联网搜索兜底
   ▼
混合检索：向量检索（FAISS / Milvus）──┐
         BM25 关键词检索 ──────────┤→ RRF 融合 → BGE-Reranker 精排 → Top-5
   │
   ▼
双层记忆：短期（Redis，近 6 轮对话上下文，不可用时降级进程内）
         长期（MySQL/PostgreSQL/SQLite，SQLAlchemy 持久化全部会话）
   │
   ▼
答案生成：qwen-plus（主模型）→ 异常自动降级 deepseek-chat
   │
   ▼
返回 JSON：answer + route（本次路由）+ sources（来源文档）
```

## 核心模块

| 模块 | 说明 |
|---|---|
| Router 意图路由 | 按检索相关度阈值判断走知识库、联网还是直接回复，优先私有知识库 |
| 混合检索 | FAISS/Milvus 向量检索 + BM25 关键词检索，RRF 倒数排名融合 |
| 重排序 | 可选接入 SiliconFlow BGE-Reranker-v2-M3，对召回片段相关性精排 |
| 模型降级 | 主模型 qwen-plus 调用异常（超时/限流）自动切换 deepseek-chat 兜底 |
| 双层记忆 | 短期记忆基于 **Redis**（List + LTRIM 保留近 6 轮，24h 过期，Redis 不可用自动降级进程内 deque）；长期记忆经 **SQLAlchemy** 落库（默认 SQLite，配 `DATABASE_URL` 即切 **MySQL/PostgreSQL**，业务代码零改动） |
| 多源文档接入 | **Unstructured** 统一解析 HTML/DOC/PPTX/XLSX/CSV/EML 等格式 + 原生 PDF / Word / Markdown 解析 + 网页 URL 抓取（`python app/ingest.py --url https://...`），Embedding 可切换 text-embedding-v2 或 BGE-M3（SiliconFlow） |
| Tool 工具化 | 知识库检索与联网搜索封装为 LangChain @tool 标准工具（app/tools.py），可无缝接入 LangGraph 等 Agent 编排扩展 |
| **LangGraph ReAct 快慢双通道** | `/chat` 默认 `mode=fast` 走规则路由单次响应（低延迟低成本）；`mode=agent` 进入 **LangGraph ReAct 循环**，由模型自主规划调用哪些工具、调用几次（Reason → Act → Observe），响应中返回 `steps` 工具调用轨迹；Agent 链路异常自动降级回规则路由 |

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env    # 至少填入 DASHSCOPE_API_KEY，其余可选

# 放入知识库文档到 data/ 目录，然后构建索引
python app/ingest.py

# 启动服务
python -m uvicorn app.main:app --reload
# 浏览器打开 http://127.0.0.1:8000/docs 调试
```

**可选配置（在 .env 中）：**
- `DEEPSEEK_API_KEY`：启用模型降级兜底
- `TAVILY_API_KEY`：启用联网搜索兜底（tavily.com 免费注册）
- `SILICONFLOW_API_KEY`：启用 BGE-Reranker 精排 / BGE-M3 Embedding（siliconflow.cn 免费注册）
- `VECTOR_STORE=milvus` + `MILVUS_URI`：向量库切换为 Milvus（默认 FAISS）
- `EMBEDDING_PROVIDER=siliconflow`：Embedding 切换为 BAAI/bge-m3
- `REDIS_URL`：短期记忆接入 Redis（默认降级为进程内记忆，零配置可跑）
- `DATABASE_URL`：长期记忆数据库，默认 SQLite；生产建议 MySQL/PostgreSQL

**一键拉起 MySQL + Redis（可选）：**

```bash
docker compose up -d      # 启动 assistant-mysql(3306) + assistant-redis(6379)
# 然后在 .env 中配置：
# DATABASE_URL=mysql+pymysql://assistant:assistant123@localhost:3306/assistant
# REDIS_URL=redis://localhost:6379/0
```

## 接口示例

```json
POST /chat
{"query": "每周应该运动多长时间？", "session_id": "u1"}

{"answer": "成年人每周应进行至少150分钟中等强度有氧运动，并配合每周至少2次力量训练 [1][2]。",
 "route": "knowledge",
 "model": "qwen-plus",
 "sources": [{"ref": 1, "doc": "健康作息指南.txt"}]}
```

**Agent 模式（ReAct 自主规划）：**

```json
POST /chat
{"query": "献血的间隔要求是什么？", "session_id": "u2", "mode": "agent"}

{"answer": "全血献血间隔不少于6个月；单采血小板间隔不少于2周 [1]...",
 "route": "agent",
 "model": "qwen-plus（ReAct）",
 "steps": ["knowledge_base_search"]}
```

`steps` 即模型本次自主决定的工具调用轨迹；闲聊类问题轨迹为空（模型判断无需调工具）。

知识库未命中且未配置联网时，回答「知识库中未找到相关内容」，不编造。

## 技术栈

Python · FastAPI · LangChain · **LangGraph(ReAct)** · FAISS/Milvus · rank_bm25 · BGE-Reranker · qwen-plus/DeepSeek · Tavily · SQLAlchemy · MySQL · Redis · Unstructured · Docker Compose
