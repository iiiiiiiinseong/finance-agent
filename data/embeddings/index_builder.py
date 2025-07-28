# index_builder.py

"""
FAQ 인덱스를 '존재하면 로드, 없으면 새로 빌드' 방식으로 가져오는 헬퍼.
다른 상품군 인덱스도 동일 인터페이스로 확장 예정.
"""

from pathlib import Path
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings

from scripts.build_faq_index import build_index, INDEX_PATH
from config import EMBED_MODEL, OPENAI_API_KEY

emb = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)


def load_or_build_faq() -> FAISS:
    """FAQ FAISS 인덱스를 반환."""
    if any((INDEX_PATH / f).exists() for f in ["index.faiss", "index.pkl"]):
        return FAISS.load_local(
            str(INDEX_PATH),
            embeddings=emb,
            allow_dangerous_deserialization=True,
        )
    # 없으면 새로 만듦
    return build_index(embeddings=emb)

