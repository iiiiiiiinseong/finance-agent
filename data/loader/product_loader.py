# data/loader/product_loader.py
"""
PDF 상품설명서를 LangChain Document 리스트로 변환
────────────────────────────────────────────
1) 첫 페이지 요약 텍스트 ➜ gpt-4o 로 메타데이터(JSON only) 추출
   • product_code  : PLM_P… 코드 (파일명 기반∙LLM 검증)
   • compliance_id : 준법감시인 심의필 번호
   • product       : 상품명
   • file_date     : YYYY-MM-DD (파일명 또는 심의필 일자)
2) 섹션 헤더(정규식) 기준 분할 후 토큰 Chunk
3) 옵션: 메타 레코드를 product_deposit.jsonl 로 저장
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import re, datetime, json, logging, os, sys

from openai import OpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import LLM_MODEL, OPENAI_API_KEY, ROOT_DIR

# ──────────────────────────────────────────────────────────────
__all__ = ["load_docs"]

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ── 규칙·정규식 ──────────────────────────────────────────
FNAME_RGX      = re.compile(r"^(?P<code>PLM_[A-Za-z0-9]+).*?_(?P<date>\d{8})")
COMPLIANCE_RGX = re.compile(r"준법감시인\s*심의필\s*(\d{4}[-‐]\d+)")

SECTION_RGX = re.compile(
    r"^\s*(?:\d+\s*[.)-]?)?\s*(상품\s*개요|상품\s*특징|거래\s*조건|가입\s*방법|중도해지|수수료|유의사항)",
    re.IGNORECASE | re.MULTILINE,
)

CATEGORY_MAP = {
    "적금": ["적금"],
    "예금": ["예금"],
    "입출금통장": ["통장","계좌"],
    "연금(IRP)": ["IRP","연금"],
    "ISA": ["ISA"],
}


# ── 헬퍼 ────────────────────────────────────────────────
def _guess_category(prod_name: str) -> str:
    for cat, kws in CATEGORY_MAP.items():
        if any(kw in prod_name for kw in kws):
            return cat
    return "기타"

def _clean_first_page(raw: str) -> str:
    cleaned = re.sub(r"^\s*[\dⅰ-ⅺ]+\s*$", "", raw, flags=re.MULTILINE)
    return "\n".join([ln for ln in cleaned.splitlines() if ln.strip()][:40])

# 1. 기본 메타 LLM
def _meta_llm(pdf: Path, first_page: str) -> Dict[str, Any]:
    hints = {}
    if (m := FNAME_RGX.match(pdf.stem)):
        hints = {"file_code": m.group("code"), "file_date_raw": m.group("date")}

    sys_msg = (
        "Extract metadata as JSON with keys: "
        '{"product_code": str, "compliance_id": str, "product": str, '
        '"file_date": "YYYY-MM-DD", "product_category": str}'
    )
    usr_msg = f"Hints: {json.dumps(hints, ensure_ascii=False)}\n\n{first_page}"

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": sys_msg},
                  {"role": "user", "content": usr_msg}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

# 2. attrs LLM
def _attrs_llm(section_txt: str) -> Dict[str, Any]:
    sys = (
  "아래 ‘거래 조건’ 텍스트에서 JSON을 추출한다.\n"
  "- max_rate, min_rate 값은 '2.35%' 형태 **순수 금리**만 대상.\n"
  "- '세율·세금·우대이율·우대금리·추가금리' 문구가 붙은 퍼센트는 제외.\n"
  "- 금리가 구간별로 여러 개이면 최댓값·최솟값을 계산해라.\n"
  '{"max_rate": float|null, "min_rate": float|null, '
  '"term_options": [str], "early_break_fee": str, "extra": dict}'
)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": section_txt[:2500]}],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=200,
    )
    return json.loads(resp.choices[0].message.content)

# 3. 백업 규칙
def _meta_backup(pdf: Path, raw: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if (m := FNAME_RGX.match(pdf.stem)):
        meta["product_code"] = m.group("code")
        meta["file_date"] = datetime.datetime.strptime(
            m.group("date"), "%Y%m%d").date().isoformat()
    if (c := COMPLIANCE_RGX.search(raw)):
        meta["compliance_id"] = c.group(1)
    first_line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    meta["product"] = first_line[:60]
    meta["product_category"] = _guess_category(first_line)
    return meta

def _attrs_backup(txt: str) -> Dict[str, Any]:
    rates = [float(x) for x in re.findall(r"(\d+\.\d+)\s*%", txt)]
    terms = re.findall(r"(\d+년|\d+개월|\d+일)", txt)
    return {
        "max_rate": max(rates) if rates else None,
        "min_rate": min(rates) if rates else None,
        "term_options": list(set(terms)),
        "early_break_fee": "",
        "extra": {},
    }

def _split_sections(text: str):
    spans = [(m.start(), m.end(), m.group(1).strip())
             for m in SECTION_RGX.finditer(text)]
    if not spans:
        return [("기타", text)]
    sections = []
    for i, (s, e, h) in enumerate(spans):
        body = text[e: spans[i+1][0] if i+1<len(spans) else len(text)].strip()
        sections.append((h, body))
    return sections

# ── Public API ──────────────────────────────────────────
def load_docs(
    dir_path: str | Path,
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    save_jsonl: bool = False,
) -> List[Document]:
    dir_path = Path(dir_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n","\n",". "],
    )
    docs, metas = [], []

    for pdf in sorted(dir_path.glob("*.pdf")):
        pages = PyMuPDFLoader(str(pdf)).load()
        first_p = _clean_first_page(pages[0].page_content)
        full_tx = "\n".join(p.page_content for p in pages)

        # 1) 메타
        try:
            meta = _meta_llm(pdf, first_p)
            logging.info("meta✔ %s", pdf.name)
        except Exception as e:
            logging.warning("meta✖ %s (%s) → backup", pdf.name, e)
            meta = _meta_backup(pdf, full_tx)

        # 공통 필드 보장
        for key in ("min_rate","max_rate","term_options",
                    "early_break_fee","extra"):
            meta.setdefault(key, None if key.endswith("rate") else
                                 [] if key=="term_options" else
                                 "" if key=="early_break_fee" else {})
        # 2) 섹션 & attrs
        attrs_added=False
        for sec, txt in _split_sections(full_tx):
            relative_pdf_path = str(pdf.relative_to(ROOT_DIR)).replace('\\', '/')
            base = {**meta, "section": sec, "pdf_path": relative_pdf_path}
            if not attrs_added and sec.startswith("거래"):
                try:
                    attrs = _attrs_llm(txt)
                except Exception:
                    attrs = _attrs_backup(txt)
                meta.update(attrs)
                base.update(attrs)
                attrs_added=True
            for chunk in splitter.split_text(txt):
                docs.append(Document(page_content=chunk, metadata=base))
        metas.append({**meta, "pdf_path": relative_pdf_path})

    if save_jsonl:
        out = Path(__file__).resolve().parents[2] / "data/processed/product_deposit.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m, ensure_ascii=False)+"\n")
        logging.info("JSONL saved → %s", out)
    logging.info("PDF %d → chunks %d", len(metas), len(docs))
    return docs