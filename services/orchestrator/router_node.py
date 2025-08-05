# services/orchestrator/router_node.py

"""
services/orchestrator/router_node.py
------------------------------------
GPT-JSON Intent Classifier  (faq / product / advise / fallback)
라벨별 필요한 체인만 lazy 실행 + 안전 wrapper
Manager Agent가 동적으로 소스 리스트 병합
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, Literal, List

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError

from services.fallback import fallback_llm 
from services.rag.faq_rag import graph as FAQ_GRAPH
from services.rag.product_rag import graph as PROD_GRAPH
from services.advisor.advisor_stub import advise
from config import OPENAI_API_KEY, LLM_MODEL

# ────────────────────────────────────────────────────────────────
# 0. 공용 LLM 인스턴스
# 현재 모두 Gpt 4o-mini를 사용하지만 이후 다른 모델로 변경 가능
LLM_CLASSIFY = ChatOpenAI(
    model          = LLM_MODEL,
    temperature    = 0,
    openai_api_key = OPENAI_API_KEY,
)
LLM_MANAGER = ChatOpenAI(
    model          = LLM_MODEL,
    temperature    = 0.2,
    openai_api_key = OPENAI_API_KEY,
)
# ────────────────────────────────────────────────────────────────
# 1. GPT 분류기  (faq / product / advise / fallback)
class RouteLabel(BaseModel):
    label: Literal["faq", "product", "advise", "fallback"] = Field(
        description="faq = 인터넷/스마트뱅킹·예금·대출·펀드 FAQ, "
                    "product = 예금·적금 상품 설명, "
                    "advise = 상품 추천·가입 상담"
                    "fallback = 일반적인 질문에 대한 답변"
    )

parser = PydanticOutputParser(pydantic_object=RouteLabel)

SYSTEM_ROUTE = SystemMessage(
    content=(
        "너는 은행 챗봇 라우터다. 사용자의 질문을 읽고 "
        "'faq', 'product', 'advise', 'fallback' 중 하나로만 분류하여 "
        "JSON {\"label\": …} 형태로 답하라."
    )
)

FEW_SHOTS = [
    ("OTP 비밀번호 오류 해제 방법 알려줘", "faq"),
    ("정기적금 3년 만기 금리 얼마야?", "product"),
    ("나에게 맞는 적금 추천해줘", "advise"),
    ("오늘 서울 날씨 어때?", "fallback"),
]
def _build_route_messages(q: str) -> list:
    guide = (
        "가능한 라벨: "
        "faq(이용·절차·보안 FAQ), "
        "product(상품 설명서 기반 질의), "
        "advise(맞춤 추천·비교), "
        "fallback(기타 일반 질문). "
        "반드시 JSON {\"label\":\"…\"} 하나만 반환."
    )
    msgs = [SYSTEM_ROUTE, SystemMessage(content=guide)]
    for qs, lab in FEW_SHOTS:
        msgs += [
            HumanMessage(content=qs),
            SystemMessage(content=RouteLabel(label=lab).json())
        ]
    msgs.append(HumanMessage(content=q))
    return msgs

def _route_llm(question: str) -> str:
    raw = LLM_CLASSIFY.invoke(
        _build_route_messages(question),
        response_format={"type": "json_object"}  # JSON-only
    ).content    
    try:
        return parser.parse(raw).label
    except ValidationError:
        return "fallback"  # 이후 fallback llm 노드로 대체 예정

@lru_cache(maxsize=2_048)
def route(question: str) -> str:
    """GPT 기반 Intent 분류 (LRU 캐시)"""
    try:
        return _route_llm(question)
    except Exception:
        return "fallback"
    
# ────────────────────── 안전 실행 래퍼 & 캐시
def _safe(graph, q: str) -> Dict:
    try:
        return graph.invoke({"question": q})
    except Exception:
        return {"answer": "", "context": ""}

@lru_cache(maxsize=2_048)
def _run_faq(q: str)  -> Dict: return _safe(FAQ_GRAPH,  q)

@lru_cache(maxsize=2_048)
def _run_prod(q: str) -> Dict: return _safe(PROD_GRAPH, q)

# ────────────────────────────────────────────────────────────────
# 2. Manager Agent -> 3개의 노드를 병렬 호출해 통합 답변
# 병렬로 FAQ / PRODUCT RAG 호출 후 LLM으로 최종 응답 합성.
# label == 'advise' 인 경우에도 두 RAG 모두 호출해 컨텍스트 확보.

SYSTEM_MANAGER = SystemMessage(content=(
    "너는 우리은행 AI 컨시어지 챗봇의 '답변 통합 엔진'이다.\n"
    "입력으로 (01)(02)(03)… 형식의 **Source** 문단들이 주어진다.\n"
    "각 Source는 다음 셋 중 하나에서 생성된다.\n"
    "  - (FAQ)  : 사내 FAQ RAG가 반환한 답변\n"
    "  - (PROD) : 예·적금 상품 설명 RAG가 반환한 답변 및 PDF 메타\n"
    "  - (FALL) : 범용 LLM Fallback 답변\n"
    "\n"
    "### 통합 규칙\n"
    "1. **정확·최신 우선** — 동일 항목이 충돌하면 더 구체적이거나\n"
    "   '심의필 번호/발행일'이 최신인 Source를 채택한다.\n"
    "2. **근거 인용** — 답변 문장 끝에 반드시 대응 Source 번호를 (①)(②)… 붙인다.\n"
    "   - 여러 Source를 조합할 때는 ‘…¹²’ 처럼 연속 표기 가능.\n"
    "3. **라벨별 특수 처리**\n"
    "   - label == 'faq'    : FAQ Source 위주로, 필요 시 (PROD) 보충.\n"
    "   - label == 'product': 상품 특징·금리·중도해지·세율 중심으로 설명.\n"
    "                        ① 상품명, ② 주요 조건, ③ 유의사항 세 단락으로 나눠라.\n"
    "   - label == 'advise' : 조건에 가장 부합하는 2-3개 상품을 추천 목록 형태로 제시하고,\n"
    "                        각 항목 옆에 최대 금리%와 PDF 보기 안내 문구를 넣어라.\n"
    "4. **규제 문구** — 정보가 부족하거나 불확실하면 다음 안내문을 붙인다.\n"
    "   『해당 내용은 담당 직원 확인 후 최종 안내될 수 있습니다.』\n"
))

def _merge_sources(question: str, label: str,
                   sources: List[Dict]) -> Dict:
    numbered_msgs, merged_ctx = [], ""
    for idx, src in enumerate(sources, start=1):
        tag = f"({idx:02d})"
        numbered_msgs.append(SystemMessage(content=f"{tag} {src['answer']}"))
        if src.get("context"):
            merged_ctx += f"\n\n{src['context']}"
    msgs = [SYSTEM_MANAGER] + numbered_msgs + [
        HumanMessage(content=f"label:{label}\n\n질문:{question}")]
    final = LLM_MANAGER.invoke(msgs).content
    return {"answer": final, "context": merged_ctx[:4000]}

def manager_agent(question: str, label: str) -> Dict:
    jobs = []
    if label in {"faq", "product", "advise"}:
        jobs += [("faq",  _run_faq)]
    if label in {"product", "advise"}:
        jobs += [("prod", _run_prod)]

    # 병렬 실행
    with ThreadPoolExecutor(max_workers=len(jobs)) as exe:
        results = [exe.submit(fn, question) for _, fn in jobs]
        sources  = [f.result() for f in results]

    return _merge_sources(question, label, sources)


# ────────────────────────────────────────────────────────────────
# 3. 외부 API

def invoke(question: str) -> Dict:
    label = route(question)
    print(f'\nlabel: {label}\n')
    if label == "faq":
        return _run_faq(question)
    if label == "product":
        return manager_agent(question, label="product")
    if label == "advise":
        # 가입 상담 라벨이면 Manager Agent + 인사이드 advise stub
        base = manager_agent(question, label="advise")
        extra = advise(question)
        return {
            "answer": base["answer"] + "\n\n---\n" + extra["answer"],
            "context": base["context"]
        }
    return fallback_llm.invoke(question)
