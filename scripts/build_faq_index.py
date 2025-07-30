# build_faq_index.py

"""
build_faq_index.py
------------------
FAQ JSONL → FAISS 인덱스 생성/갱신 스크립트.
"""

from pathlib import Path
import json, sys, logging
sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS
from config import DATA_DIR, INDEX_DIR, EMBED_MODEL, OPENAI_API_KEY

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

SOURCE_JSONL = DATA_DIR / "processed" / "faq_woori_structured.jsonl"
INDEX_PATH = INDEX_DIR / "faq_faiss"

def build_index(source: Path = SOURCE_JSONL, out_dir: Path = INDEX_PATH) -> FAISS:
    """JSONL을 읽어 FAISS 인덱스를 생성하고 저장."""
    logging.info("loading data from %s", source)
    documents = []
    with source.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            content = f"Q: {rec['question']}\nA: {rec['answer']}"
            documents.append(Document(page_content=content, metadata=rec))

    logging.info("embedding %d documents ...", len(documents))
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(documents, embeddings)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    db.save_local(str(out_dir))
    logging.info("FAISS index saved → %s", out_dir)

    return db


if __name__ == "__main__":
    build_index()
