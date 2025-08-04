# services/rag/product_rag/retriever.py
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever
from typing import List
from config import EMBED_MODEL, INDEX_DIR, OPENAI_API_KEY

emb = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)

class DepositRetriever:
    """FAISS + BM25 하이브리드"""
    def __init__(self, k: int = 3):
        faiss_path = Path(INDEX_DIR / "deposit_faiss")
        self.vector = FAISS.load_local(
            str(faiss_path),
            embeddings=emb,
            allow_dangerous_deserialization=True
        )
        self.bm25 = BM25Retriever.from_documents(
            list(self.vector.docstore._dict.values())
        )
        self.k = k

    def invoke(self, query: str):
        return self.__call__(query)

    def __call__(self, query: str)-> List:
        dense = self.vector.similarity_search(query, k=self.k)
        sparse = self.bm25.invoke(query)[: self.k]
        uniq = {d.page_content: d for d in dense + sparse}
        return list(uniq.values())



