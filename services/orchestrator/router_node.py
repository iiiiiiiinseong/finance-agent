# services/orchestrator/router_node.py

"""
services/orchestrator/router_node.py
------------------------------------
Manager Agent + GPT Classifier + 병렬 RAG 파이프라인
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, Literal

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
# 1. GPT 분류기  (faq / product / advise)

class RouteLabel(BaseModel):
    label: Literal["faq", "product", "advise"] = Field(
        description="faq = 인터넷/스마트뱅킹·예금·대출·펀드 FAQ, "
                    "product = 예금·적금 상품 설명, "
                    "advise = 상품 추천·가입 상담"
    )

parser = PydanticOutputParser(pydantic_object=RouteLabel)

SYSTEM_ROUTE = SystemMessage(
    content=(
        "너는 은행 챗봇 라우터다. 사용자의 질문을 읽고 "
        "'faq', 'product', 'advise' 셋 중 하나로만 분류해 "
        "JSON 형식으로 출력한다."
    )
)

FEW_SHOTS = [
    ("OTP 비밀번호 오류 해제 방법 알려줘", "faq"),
    ("정기적금 3년 만기 금리 얼마야?", "product"),
    ("나에게 맞는 적금 추천해줘", "advise"),
]
def _build_route_messages(question: str):
    msgs = [SYSTEM_ROUTE]
    for q, lab in FEW_SHOTS:
        msgs.append(HumanMessage(content=q))
        msgs.append(SystemMessage(content=RouteLabel(label=lab).json()))

    msgs.append(HumanMessage(content=question)) # 사용자 질문
    return msgs

def _route_llm(question: str) -> str:
    raw = LLM_CLASSIFY.invoke(_build_route_messages(question)).content
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
# ────────────────────────────────────────────────────────────────
# 2. Manager Agent -> 두 RAG를 병렬 호출해 통합 답변

SYSTEM_MANAGER = SystemMessage(content=(
    "너는 우리은행 AI 챗봇이다. 다음 정보를 사용해 질문에 답해라.\n"
    "① FAQ_RAG 답변\n"
    "② PRODUCT_RAG 답변\n"
    "규칙:\n"
    "• 두 답변의 근거/수치가 충돌 시, 더 구체적이고 최신인 쪽을 선택\n"
    "• 필요하면 두 답변을 요약·병합하되, 인용번호(①,②) 유지\n"
    "• label이 'advise'면 간단히 가입·추천 멘트 추가"
))

def _run_faq(q: str) -> Dict:
    return FAQ_GRAPH.invoke({"question": q})

def _run_prod(q: str) -> Dict:
    return PROD_GRAPH.invoke({"question": q})

def manager_agent(question: str, label: str) -> Dict:
    """
    병렬로 FAQ / PRODUCT RAG 호출 후 LLM으로 최종 응답 합성.
    label == 'advise' 인 경우에도 두 RAG 모두 호출해 컨텍스트 확보.
    """
    with ThreadPoolExecutor(max_workers=2) as exe:
        fut_faq  = exe.submit(_run_faq,  question)
        fut_prod = exe.submit(_run_prod, question)
        faq_res, prod_res = fut_faq.result(), fut_prod.result()

    msgs = [
        SYSTEM_MANAGER,
        SystemMessage(content=f"① FAQ_RAG:\n{faq_res['answer']}"),
        SystemMessage(content=f"② PRODUCT_RAG:\n{prod_res['answer']}"),
        HumanMessage(content=f"label: {label}\n\n질문: {question}")
    ]
    final_answer = LLM_MANAGER.invoke(msgs).content

    # 컨텍스트는 두 RAG의 context를 합침
    merged_ctx = (
        (faq_res.get("context", "") or "") +
        ("\n\n" + prod_res.get("context", "") if prod_res.get("context") else "")
    )[:4000]   # 4,000자로 제한

    return {"answer": final_answer, "context": merged_ctx}

# ────────────────────────────────────────────────────────────────
# 3. 외부 API

def invoke(question: str) -> Dict:
    label = route(question)
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
