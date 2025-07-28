# services/rag/faq_rag/retriever.py

"""
retriever.py
------------
FAQ 도큐먼트를 검색하는 전용 Retriever.
"""
from data.embeddings.index_builder import load_or_build_faq

_FAQ_DB = load_or_build_faq()
faq_retriever = _FAQ_DB.as_retriever(search_kwargs={"k": 3})
