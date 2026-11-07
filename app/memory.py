"""双层记忆：短期记忆（Redis，近 6 轮上下文）+ 长期记忆（SQLAlchemy 持久化，支持 MySQL/PostgreSQL/SQLite）

设计说明：
- 短期记忆优先使用 Redis（服务重启不丢失、多进程共享）；Redis 不可用时自动降级为进程内 deque，保证开发环境零配置可跑。
- 长期记忆通过 SQLAlchemy ORM 抽象，默认 SQLite；生产环境配置 DATABASE_URL 即可无感切换 MySQL/PostgreSQL，业务代码零改动。
"""
import os
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import (
    Column, DateTime, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

Base = declarative_base()
SHORT_ROUNDS = 6  # 短期记忆保留的对话轮数


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    role = Column(String(16))          # user / assistant
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class ShortTermMemory:
    """短期记忆：Redis List 实现（key: short:{session_id}，LTRIM 保留最近 2*SHORT_ROUNDS 条）"""

    def __init__(self):
        self._local: dict[str, deque] = {}
        self.redis = None
        url = os.getenv("REDIS_URL")
        if url:
            try:
                import redis
                client = redis.Redis.from_url(
                    url, socket_connect_timeout=2, decode_responses=True
                )
                client.ping()
                self.redis = client
            except Exception as e:
                print(f"[memory] Redis 连接失败，降级为进程内记忆: {e}")

    @property
    def backend(self) -> str:
        return "redis" if self.redis else "in-memory"

    def add(self, session_id: str, role: str, content: str):
        if self.redis:
            key = f"short:{session_id}"
            pipe = self.redis.pipeline()
            pipe.rpush(key, f"{role}|{content}")
            pipe.ltrim(key, -(SHORT_ROUNDS * 2), -1)  # 只保留最近 6 轮
            pipe.expire(key, 86400)                   # 24 小时自动过期
            pipe.execute()
        else:
            dq = self._local.setdefault(
                session_id, deque(maxlen=SHORT_ROUNDS * 2)
            )
            dq.append((role, content))

    def history(self, session_id: str) -> list:
        if self.redis:
            raw = self.redis.lrange(f"short:{session_id}", 0, -1)
            return [tuple(item.split("|", 1)) for item in raw]
        return list(self._local.get(session_id, deque()))


class Memory:
    def __init__(self):
        url = os.getenv(
            "DATABASE_URL",
            "sqlite:///./assistant.db",
        )
        # MySQL/PostgreSQL 连接参数：连接池预检 + 失败重连，避免长连接失效
        engine_kwargs = {"pool_pre_ping": True, "pool_recycle": 3600}
        if url.startswith("sqlite"):
            engine_kwargs = {}
        self.backend_name = url.split("+")[0].split(":")[0]  # mysql / postgresql / sqlite
        self.engine = create_engine(url, **engine_kwargs)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.short = ShortTermMemory()

    def add(self, session_id: str, role: str, content: str):
        with self.Session() as s:
            s.add(Message(session_id=session_id, role=role, content=content))
            s.commit()
        self.short.add(session_id, role, content)

    def short_history(self, session_id: str) -> list:
        """近 6 轮（user/assistant 交替），供 Prompt 组装"""
        return self.short.history(session_id)

    def long_history(self, session_id: str, limit: int = 20) -> list:
        """跨会话读取完整历史记录（长期记忆落库）"""
        with self.Session() as s:
            rows = (
                s.query(Message)
                .filter(Message.session_id == session_id)
                .order_by(Message.id.desc())
                .limit(limit)
                .all()
            )
            return [
                {"role": r.role, "content": r.content, "time": r.created_at.isoformat()}
                for r in reversed(rows)
            ]
