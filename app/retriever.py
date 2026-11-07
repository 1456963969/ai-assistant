"""混合检索：向量库 + BM25，RRF 融合，可选 BGE-Reranker 精排"""
import os
import pickle
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder

load_dotenv()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_DIR = os.path.join(ROOT, "vectorstore")
CHUNKS_PKL = os.path.join(INDEX_DIR, "chunks.pkl")


def _tokenize(text: str):
    return list(text)  # 中文按单字切分，满足关键词匹配


def rrf_fuse(ranked_lists, k: int = 60):
    """倒数排名融合：每路结果按排名计 1/(k+rank)，同一片段累加"""
    scores, by_id = {}, {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            key = id(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            by_id[key] = chunk
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [by_id[i] for i in ordered]


class HybridRetriever:
    def __init__(self):
        self.embedder = Embedder()
        with open(CHUNKS_PKL, "rb") as f:
            self.chunks = pickle.load(f)
        self.bm25 = BM25Okapi([_tokenize(c.page_content) for c in self.chunks])

        if os.getenv("VECTOR_STORE", "faiss") == "milvus":
            from langchain_community.vectorstores import Milvus
            self.vectorstore = Milvus(
                embedding_function=self.embedder,
                collection_name="ai_assistant_kb",
                connection_args={"uri": os.getenv("MILVUS_URI", "http://localhost:19530")},
            )
        else:
            from langchain_community.vectorstores import FAISS
            self.vectorstore = FAISS.load_local(
                INDEX_DIR, self.embedder, allow_dangerous_deserialization=True
            )

    def vector_search(self, query: str, k: int):
        return self.vectorstore.similarity_search(query, k=k)

    def rerank(self, query: str, chunks, top_k: int):
        """SiliconFlow BGE-Reranker-v2-M3 精排；未配置 Key 时按原序返回"""
        key = os.getenv("SILICONFLOW_API_KEY")
        if not key or not chunks:
            return chunks[:top_k]
        import httpx
        resp = httpx.post(
            "https://api.siliconflow.cn/v1/rerank",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "BAAI/bge-reranker-v2-m3",
                "query": query,
                "documents": [c.page_content[:1000] for c in chunks],
                "top_n": top_k,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return [chunks[r["index"]] for r in resp.json()["results"]]

    def retrieve(self, query: str, top_k: int = 5):
        vec = self.vector_search(query, k=top_k * 2)
        bm25_order = sorted(
            range(len(self.chunks)), key=lambda i: -self.bm25.get_scores(_tokenize(query))[i]
        )
        bm25 = [self.chunks[i] for i in bm25_order[: top_k * 2]]
        fused = rrf_fuse([vec, bm25])
        return self.rerank(query, fused, top_k)

    def top_relevance(self, query: str) -> float:
        """Top-1 片段与查询的余弦相似度（0~1），供 Router 判断是否走联网兜底"""
        try:
            import numpy as np
            res = self.vectorstore.similarity_search_with_score(query, k=1)
            if not res:
                return 0.0
            doc, _dist = res[0]
            vs = self.vectorstore
            pos = next(
                p for p, dsid in vs.index_to_docstore_id.items()
                if vs.docstore.search(dsid) is doc
            )
            vec = np.array(vs.index.reconstruct(pos), dtype=float)
            qv = np.array(self.embedder.embed_query(query), dtype=float)
            return float(np.dot(qv, vec) / (np.linalg.norm(qv) * np.linalg.norm(vec)))
        except Exception:
            return 0.0

    @staticmethod
    def format_context(chunks) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            source = os.path.basename(c.metadata.get("source", "unknown"))
            parts.append(f"[{i}]（来源：{source}）\n{c.page_content}")
        return "\n\n".join(parts)

    @staticmethod
    def build_sources(chunks) -> list:
        return [
            {"ref": i, "doc": os.path.basename(c.metadata.get("source", "unknown"))}
            for i, c in enumerate(chunks, 1)
        ]
