"""Embedding 封装：按环境变量切换 text-embedding-v2 / BAAI-bge-m3"""
import os
from langchain_core.embeddings import Embeddings


class Embedder(Embeddings):
    def __init__(self):
        self.provider = os.getenv("EMBEDDING_PROVIDER", "dashscope")
        if self.provider == "siliconflow":
            from langchain_community.embeddings import OpenAIEmbeddings
            key = os.getenv("SILICONFLOW_API_KEY")
            if not key:
                raise SystemExit("EMBEDDING_PROVIDER=siliconflow 需要配置 SILICONFLOW_API_KEY")
            self._impl = OpenAIEmbeddings(
                model="BAAI/bge-m3",
                openai_api_key=key,
                openai_api_base="https://api.siliconflow.cn/v1",
            )
        else:
            from langchain_community.embeddings import DashScopeEmbeddings
            self._impl = DashScopeEmbeddings(
                model="text-embedding-v2",
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            )

    def embed_documents(self, texts):
        return self._impl.embed_documents(texts)

    def embed_query(self, text):
        return self._impl.embed_query(text)

    def __call__(self, text):
        return self.embed_query(text)
