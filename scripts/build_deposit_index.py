# build_deposit_index.py

"""
build_deposit_index
------------------
woori_deoisut DOC → FAISS 인덱스 생성/갱신 스크립트.
"""

from pathlib import Path
import sys, logging
sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBED_MODEL, DATA_DIR, INDEX_DIR
from data.loader.product_loader import load_docs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

SOURCE_JSONL = DATA_DIR / "raw_docs" / "woori_deposit_trust_docs"
INDEX_PATH = INDEX_DIR / "deposit_faiss"

def build_index(source: Path = SOURCE_JSONL, out_dir: Path = INDEX_PATH) -> FAISS:
    DOCS = load_docs(SOURCE_JSONL, save_jsonl=True)   # jsonl 저장
    emb = OpenAIEmbeddings(
        model=EMBED_MODEL,
        chunk_size=200,
        show_progress_bar=True,
    )

    db = FAISS.from_documents(DOCS, emb)
    db.save_local("index/deposit_faiss")
    logging.info("FAISS index saved → %s", out_dir)

    print(f"> {len(DOCS):,} chunks indexed")
    return db

if __name__ == "__main__":
    build_index()
