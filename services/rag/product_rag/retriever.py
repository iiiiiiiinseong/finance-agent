# services/rag/product_rag/retriever.py
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever
from typing import List
from config import EMBED_MODEL, INDEX_DIR, OPENAI_API_KEY

emb = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)

CATEGORY_MAP = {          # 질문→카테고리 매핑
    "적금": ["적금"],
    "예금": ["예금"],
    "입출금통장": ["통장","계좌"],
    "연금(IRP)": ["IRP","연금"],
    "ISA": ["ISA"],
}

def _infer_category(text: str) -> str|None:
    for cat, kws in CATEGORY_MAP.items():
        if any(kw in text for kw in kws):
            return cat
    return None


class DepositRetriever:
    """FAISS + BM25 하이브리드 + 카테고리 필터"""
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

    def __call__(self, query: str, *, category: str|None = None) -> List:
        """category 지정 시 FAISS metadata_filter 사용"""
        faiss_filter = {"product_category": category} if category else None

        dense = self.vector.similarity_search(
            query, k=self.k, filter=faiss_filter
        )

        sparse = self.bm25.invoke(query)[: self.k]
        if category:
            sparse = [d for d in sparse
                      if d.metadata.get("product_category") == category]

        uniq = {d.page_content: d for d in dense + sparse}
        return list(uniq.values())

    def invoke(self, query: str):
        return self.__call__(query)
