"""双层记忆：短期记忆（近 6 轮上下文）+ 长期记忆（SQLAlchemy 持久化）"""
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


class Memory:
    def __init__(self):
        url = os.getenv("DATABASE_URL", "sqlite:///./assistant.db")
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self._short: dict[str, deque] = {}

    def add(self, session_id: str, role: str, content: str):
        with self.Session() as s:
            s.add(Message(session_id=session_id, role=role, content=content))
            s.commit()
        dq = self._short.setdefault(
            session_id, deque(maxlen=SHORT_ROUNDS * 2)
        )
        dq.append((role, content))

    def short_history(self, session_id: str) -> list:
        """近 6 轮（user/assistant 交替），供 Prompt 组装"""
        return list(self._short.get(session_id, deque()))

    def long_history(self, session_id: str, limit: int = 20) -> list:
        """跨会话读取完整历史记录"""
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
