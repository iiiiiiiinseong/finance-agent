#services/fallback/fallback_llm.py

"""
은행 FAQ·상품 RAG로 라우팅되지 않은 질문을 처리하는 기본 LLM 노드.
- 금융 분야 외 질문 → 정중한 안내 + 금융 질문 유도
- 기본적인 small-talk(인사, 감사) → 짧은 응대 후 금융 질문 유도
Few-shot 예시는 범주별로 다양하게 포함해 LLM이 안전히 가드레일을 지키도록 설계.
"""

from __future__ import annotations
from typing import List, Tuple
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from config import OPENAI_API_KEY, LLM_MODEL

# ────────────────────────────────────────────────────────────────
# 1. Few‑shot 예시 (범주별 2개씩)
# 범주 A: 날씨·일정 등 **비금융 정보**
FS_NON_FIN: List[Tuple[str, str]] = [
    (
        "오늘 날씨가 어때? 우산 챙겨야 할까?",
        "저는 우리은행 금융 AI 컨시어지로 금융·상품·뱅킹 관련 질문에 답변하도록 설계되었습니다."
        " 날씨 정보 대신 예·적금, 대출, 모바일뱅킹 등에 대해 궁금하신 점을 알려주시면 도와드리겠습니다."
    ),
    (
        "내일 일정 알려줘.",
        "죄송합니다. 개인 일정 관리 기능은 제공하지 않습니다. 금융 관련해서 궁금한 점이 있으시면 말씀해 주세요."
    ),
]

# 범주 B: 인사·감사 small talk
FS_SMALL_TALK: List[Tuple[str, str]] = [
    (
        "안녕, 뭐 하고 있어?",
        "안녕하세요! 우리은행 금융 AI 컨시어지입니다. 예·적금 금리나 인터넷뱅킹 이용 방법 등 궁금하신 점이 있으신가요?"
    ),
    (
        "고마워!",
        "도움이 되어 기쁩니다. 추가로 금융 관련 질문이 있으시면 언제든 말씀해 주세요."
    ),
]

# 범주 C: 규정 외 민감정보 요구(주민번호·계좌번호)
FS_PII: List[Tuple[str, str]] = [
    (
        "내 계좌번호 알려줘",
        "보안을 위해 계좌번호 같은 개인 정보는 제공해 드릴 수 없습니다. 가까운 영업점이나 인터넷뱅킹에서 직접 확인해 주세요."
    ),
    (
        "주민등록번호 입력해야 하니?",
        "주민등록번호 전체 입력은 필요하지 않습니다. 금융 거래 시 보안을 위해 일부만 입력하거나 다른 본인확인 수단을 이용합니다. 자세한 내용은 영업점을 통해 안내받으실 수 있습니다."
    ),
]

FEW_SHOT_EXAMPLES: List[Tuple[str, str]] = FS_NON_FIN + FS_SMALL_TALK + FS_PII

# ────────────────────────────────────────────────────────────────
# 2. Prompt builder

def _build_system_prompt() -> str:
    base = (
        "너는 우리은행 AI 금융 컨시어지로서, 금융·상품·뱅킹 관련 질문에만 전문적으로 답변해야 한다. "
        "범위를 벗어난 질문에는 정중히 안내하고 금융 관련 질문을 유도한다."
    )
    for user_q, assistant_a in FEW_SHOT_EXAMPLES:
        base += f"\n\n사용자: {user_q}\n어시스턴트: {assistant_a}"
    return base

SYSTEM = SystemMessage(content=_build_system_prompt())

# ────────────────────────────────────────────────────────────────
# 3. LLM Wrapper

_llm = ChatOpenAI(model=LLM_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

def invoke(question: str) -> dict:
    """Fallback 답변 반환 (dict)"""
    answer = _llm.invoke([SYSTEM, HumanMessage(content=question)]).content
    return {"answer": answer, "context": ""}
