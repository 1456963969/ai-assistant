"""文档入库：加载 data/ -> 切分 -> 向量化 -> 存入 FAISS 或 Milvus"""
import os
import pickle
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.embeddings import Embedder

load_dotenv()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
INDEX_DIR = os.path.join(ROOT, "vectorstore")
CHUNKS_PKL = os.path.join(INDEX_DIR, "chunks.pkl")


def load_documents(urls=None):
    docs = []
    for name in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, name)
        if name.endswith((".txt", ".md")):
            loader = TextLoader(path, encoding="utf-8")
        elif name.endswith(".pdf"):
            loader = PyPDFLoader(path)  # PDF：按页解析，保留页码元数据
        elif name.endswith(".docx"):
            loader = Docx2txtLoader(path)  # Word：文本抽取
        else:
            continue
        docs.extend(loader.load())
    for url in urls or []:  # 网页 URL：抓取正文并解析
        docs.extend(WebBaseLoader(url).load())
    return docs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", action="append", default=[],
                        help="额外接入的网页 URL，可重复传参")
    args = parser.parse_args()

    docs = load_documents(args.url)
    if not docs:
        raise SystemExit("data/ 目录下没有 .txt/.md/.pdf/.docx 文档，且未传 --url")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"加载 {len(docs)} 个文档，切分为 {len(chunks)} 个片段")

    embedder = Embedder()
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(CHUNKS_PKL, "wb") as f:  # BM25 索引启动时需要全量片段
        pickle.dump(chunks, f)

    if os.getenv("VECTOR_STORE", "faiss") == "milvus":
        from langchain_community.vectorstores import Milvus
        uri = os.getenv("MILVUS_URI", "http://localhost:19530")
        Milvus.from_documents(
            chunks, embedder,
            collection_name="ai_assistant_kb",
            connection_args={"uri": uri},
            auto_id=True,
        )
        print(f"已写入 Milvus（{uri}），共 {len(chunks)} 个片段")
    else:
        vs = FAISS.from_documents(chunks, embedder)
        vs.save_local(INDEX_DIR)
        print(f"FAISS 索引已保存到 {INDEX_DIR}")


if __name__ == "__main__":
    main()
