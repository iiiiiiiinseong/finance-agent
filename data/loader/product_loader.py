# data/loader/product_loader.py

"""
product_loader.py
------------------
PDF 상품설명서를 LangChain Document 리스트로 변환
- 섹션 단위로 분할 → 토큰 기반 chunking
- product, product_code, file_date, section 메타데이터 부착
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import re, datetime

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


__all__ = ["load_docs"]


# ──────────────────────────────────────────────────────────
# 1‑1) 섹션 헤더: "1. 상품개요", "2) 거래조건" 등 다양한 변형을 허용
SECTION_RGX = re.compile(
    r"^\s*(?:\d+\s*[.)-]?)?\s*(상품\s*개요|상품\s*특징|거래\s*조건|가입\s*방법|중도해지|수수료|유의사항)",
    re.IGNORECASE | re.MULTILINE,
)

# 1‑2) 파일명 규칙: PLM_P010000225_DA001_20241212073551000816.pdf
#           └────code────┘         └─     date (YYYYMMDD)     ─┘
FNAME_RGX = re.compile(
    r"^(?P<code>[A-Za-z]{3}_[A-Za-z0-9]+).*?_(?P<date>\d{8})",
)

# ──────────────────────────────────────────────────────────
# 2. 내부 헬퍼 
def _extract_metadata(pdf_path: Path, raw_text: str) -> Dict[str, str]:
    """파일명·본문으로부터 product_code / file_date / product 추출"""
    meta: Dict[str, str] = {}

    # (a) 파일명 기반
    m = FNAME_RGX.match(pdf_path.stem)
    if m:
        meta["product_code"] = m.group("code")
        # YYYYMMDD → YYYY‑MM‑DD
        meta["file_date"] = datetime.datetime.strptime(m.group("date"), "%Y%m%d").date().isoformat()

    # (b) 상품명: 2번쨰 줄 추정 (아닌 경우도 있는데 일단 이렇게 진행)
    first_line = raw_text.splitlines()[1].strip()
    meta["product"] = first_line[:60]  # 과도한 길이 방지
    return meta


def _split_by_section(text: str) -> List[tuple[str, str]]:
    """문서 전체 텍스트 → 섹션별 (header, body) 리스트"""
    spans = [(m.start(), m.end(), m.group(1).strip()) for m in SECTION_RGX.finditer(text)]

    # 섹션 헤더를 하나도 찾지 못하면 통째로 반환
    if not spans:
        return [("기타", text)]

    sections: List[tuple[str, str]] = []
    for idx, (s_start, s_end, header) in enumerate(spans):
        body_start = s_end
        body_end = spans[idx + 1][0] if idx + 1 < len(spans) else len(text)
        body_text = text[body_start:body_end].strip()
        if body_text:
            sections.append((header, body_text))
    return sections

# ────────────────────────────────────────────────────────────────
# 3. Public API 
def load_docs(
    dir_path: str | Path,
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> List[Document]:
    """폴더 내 PDF들을 LangChain Document 리스트로 반환.
    Parameters
    ----------
    dir_path : str | Path
        PDF 파일이 있는 디렉터리.
    chunk_size : int
        토큰 기준 분할 크기. (OpenAI 1 token ≈ 4 chars)
    chunk_overlap : int
        분할 시 앞뒤 겹침 토큰 수.
    """

    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"{dir_path} is not a directory")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". "],
    )

    documents: List[Document] = []

    for pdf_file in dir_path.glob("*.pdf"):
        # 3-1) PDF 읽기 (page 단위)
        loader = PyMuPDFLoader(str(pdf_file))
        pages = loader.load()  # List[Document]
        raw_text = "\n".join(p.page_content for p in pages)

        # 3-2) 메타데이터 공통 부분
        common_meta = _extract_metadata(pdf_file, raw_text)

        # 3-3) 섹션 분할 → chunking
        for section_name, section_text in _split_by_section(raw_text):
            meta_base = {**common_meta, "section": section_name}
            chunks = splitter.split_text(section_text)
            print(f"meta_base: {meta_base} - chunk lenght: {len(chunks)}")
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata=meta_base))

    return documents