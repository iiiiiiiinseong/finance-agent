# services/orchestrator/router_node.py

"""
services/orchestrator/router_node.py
------------------------------------
GPT-JSON Intent Classifier  (faq / product / advise / fallback)
Conversational Query Rewriting으로 대화 맥락 유지하며 답변
Manager Agent가 동적으로 소스 리스트 병합
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, Literal, List, Any

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError

from services.fallback import fallback_llm 
from services.rag.faq_rag import graph as FAQ_GRAPH
from services.rag.product_rag import graph as PROD_GRAPH
from services.advisor.advisor_stub import advise
from config import OPENAI_API_KEY, LLM_MODEL

# ────────────────────────────────────────────────────────────────
# 0. 공용 LLM 인스턴스
# 현재 모두 Gpt 4o-mini를 사용하지만 이후 역할에 따라 다른 모델로 변경 가능
LLM_REWRITER = ChatOpenAI(
    model          = LLM_MODEL,
    temperature    = 0,
    openai_api_key = OPENAI_API_KEY,
)
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
# 1. 대화 이력 기반 질문 재구성 (신규 추가)
REWRITE_PROMPT = SystemMessage(content=(
    "당신은 금융 상품 챗봇 AI와 유저의 대화 히스토리를 기반으로 마지막 유저의 질문을 분석하여 금융상품 RAG 통합엔진이 문서를 잘 찾을 수 있게 질문을 구체적으로 재작성하세요."
    "재작성된 질문은 길이가 너무 길어지지 않게 하며, 유저가 원히는 핵심이 무엇인지 명확하게 들어나는 문장이어야 합니다."
    "적용되는 RAG의 서치알고리즘은 백터유사도를 기반으로 하기 때문에 이를 고려하여 질문을 재작성하세요.\n"
    "### 당신의 사고 과정:\n"
    "1.  **핵심 주제 파악**: 먼저 전체 대화 내용을 검토하여 대화의 핵심 주제(예: '우리 SUPER 주거래 적금'과 같은 특정 상품명, 'ISA 계좌'와 같은 개념)를 정확히 파악합니다. 핵심 주제는 주로 대화 초반에 언급됩니다.\n"
    "2.  **최신 질문 분석**: 사용자의 마지막 질문을 분석합니다. 이 질문이 핵심 주제에 대한 후속 질문입니까? (예: '금리는 어떻게 되나요?', '더 자세히 알려줘', '다른 상품과 비교해줘').\n"
    "3.  **모호함 해결**: 만약 AI의 직전 답변이 사용자에게 **여러 개의 선택지(예: 여러 상품 목록)를 제시**했고, 사용자의 마지막 질문이 그 중 **무엇을 지칭하는지 모호하다면**, 그 질문을 **'제시된 모든 선택지를 비교해달라'** 는 명확한 질문으로 재구성해야 합니다.\n"
    "4.  **맥락 결합 및 재구성**: \n"
    "    - 후속 질문일 경우, 반드시 **'핵심 주제'** 와 **'최신 질문'** 을 결합하여 완전한 의미의 독립형 질문을 만듭니다.\n"
    "    - 최신 질문이 이미 그 자체로 완전한 독립형 질문이라면, 절대 변경하지 않고 그대로 사용합니다.\n\n"
    "### 출력 규칙:\n"
    "- 당신의 출력은 반드시 재구성된 **질문 문자열 하나**여야 합니다.\n"
    "- 어떠한 설명, 레이블, 서문도 추가하지 마십시오."
))
REWRITE_FEW_SHOTS = [
    # Case 1: 기본 후속 질문
    {
        "history": [HumanMessage(content="우리 첫급여 신용대출 상품에 대해 알려줘."), AIMessage(content="네, 우리 첫급여 신용대출은 사회초년생을 위한 상품으로...")],
        "last_question": "우대 조건은 어떻게 되나요?",
        "rewritten": "우리 첫급여 신용대출의 우대 조건은 무엇인가요?"
    },
    # Case 2: 다단계 후속 질문 (핵심 주제 기억)
    {
        "history": [
            HumanMessage(content="우리 SUPER 주거래 적금에 대해 알려줘."),
            AIMessage(content="네, 해당 상품은 주거래 고객님께 높은 우대금리를 제공하는 적금입니다."),
            HumanMessage(content="방금 말한거 다시 정리해줄래?"),
            AIMessage(content="물론입니다. 우리 SUPER 주거래 적금은...")
        ],
        "last_question": "그럼 기본 금리랑 우대금리는 어떻게 되는지 알려줘",
        "rewritten": "우리 SUPER 주거래 적금의 기본 금리와 우대금리는 어떻게 되나요?"
    },
    # Case 3: '선택 후 모호함' 해결 (신규 추가)
    {
        "history": [
            HumanMessage(content="우리은행 적금 추천해줘"),
            AIMessage(content="네, 고객님께는 '우리 퍼스트 적금2', '급여형 적금', '청약저축형 적금'을 추천해 드립니다.")
        ],
        "last_question": "우대금리 조건이 어떻게 되는지도 설명해줘",
        "rewritten": "'우리 퍼스트 적금2', '급여형 적금', '청약저축형 적금'의 우대금리 조건을 각각 비교 설명해주세요."
    },
    # Case 4: 독립적인 질문 (변경 없음)
    {
        "history": [],
        "last_question": "ISA 계좌가 뭔가요?",
        "rewritten": "ISA 계좌가 뭔가요?"
    }
]
def _get_chat_history(history: List[Dict]) -> List:
    """Streamlit의 세션 기록을 Langchain 메시지 형식으로 변환"""
    buffer = []
    for msg in history:
        if msg.get("role") == "user":
            buffer.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "assistant":
            buffer.append(AIMessage(content=msg.get("content")))
    return buffer

def _rewrite_query_with_history(question: str, history: List[Dict]) -> str:
    """대화 이력을 바탕으로 사용자 질문을 재구성"""
    if not history:
        return question

    few_shot_messages = []
    for shot in REWRITE_FEW_SHOTS:
        few_shot_messages.extend(shot["history"])
        few_shot_messages.append(HumanMessage(content=f"마지막 질문: {shot['last_question']}"))
        few_shot_messages.append(AIMessage(content=shot['rewritten']))

    chat_history_messages = _get_chat_history(history)
        
    # 마지막 질문은 HumanMessage로 추가
    messages = [REWRITE_PROMPT] + chat_history_messages + [HumanMessage(content=f"마지막 질문: {question}")]
    print(f"\nmessages:\n {messages}")
    try:
        response = LLM_REWRITER.invoke(messages)
        rewritten_question = response.content.strip()
        print(f"\n[Rewrite] 원본: {question} -> 재구성: {rewritten_question}\n")
        return rewritten_question
    except Exception as e:
        print(f"질문 재구성 실패: {e}")
        return question

# 2. GPT 분류기  (faq / product / advise / fallback)
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
# 추후 FEW_SHOT 말고 Adaptive logic을 위해 분류 ML 도입 고려 중
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
        return "fallback"

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
    "너는 우리은행의 전문성과 친절함을 갖춘 'AI 금융 컨시어지'이다.\n"
    "주어진 Source 문단들을 조합하여 사용자의 질문에 완벽하고 이해하기 쉬운 답변을 생성해야 한다.\n\n"
    "### 페르소나 및 말투\n"
    "- **전문성**: 정확한 정보를 기반으로 답변하며, 필요시 심의필 번호나 기준일을 언급한다.\n"
    "- **친절함**: 고객의 눈높이에 맞춰 부드럽고 상냥한 말투를 사용한다. 딱딱한 설명은 피한다.\n"
    "- **적극성**: 단순히 답변만 하지 않고, 고객에게 도움이 될 만한 추가 정보를 먼저 제안한다.\n\n"
    "### 답변 생성 규칙\n"
    "1. **답변 구조 (필수)**: 다음 세 부분으로 나누어 답변을 구성하라.\n"
    "   - **핵심 요약**: 사용자의 질문에 대한 가장 중요한 결론을 한두 문장으로 먼저 제시한다.\n"
    "   - **상세 설명**: Source 내용을 바탕으로 구체적인 정보를 항목별로 나누어 명확하게 설명한다. (상품의 경우 특징, 금리, 조건, 유의사항 등)\n"
    "   - **연관 질문 제안**: 사용자가 다음으로 궁금해할 만한 질문 3가지를 제안하여 대화를 유도한다.\n\n"
    "2. **근거 인용**: 답변 내용의 신뢰도를 위해, 문장 끝에 참조한 Source 번호를 `(출처: ①)` 와 같이 명확히 표기한다. 여러 Source를 조합했다면 `(출처: ①, ②)` 형식으로 쓴다.\n\n"
    "3. **라벨별 특수 처리**:\n"
    "   - `label: product`: 상품의 핵심적인 특징을 중심으로 설명한다.\n"
    "   - `label: advise`: 사용자의 요구사항을 파악하고, 가장 적합한 상품 2~3개를 비교하여 추천 목록 형태로 제시한다.\n\n"
    "4. **주의 문구**: 정보가 부족하거나 최신 정보 확인이 필요한 경우, 답변 마지막에 다음 문구를 정중하게 추가한다.\n"
    "   `※ 안내된 내용은 정보 제공을 위한 것이며, 실제 상품 가입 시 조건이 달라질 수 있으니 자세한 내용은 상품설명서를 꼭 확인해주세요.`\n"
))

def _merge_sources(question: str, label: str,
                   sources: List[Dict]) -> Dict:
    numbered_msgs, merged_ctx = [], ""
    for idx, src in enumerate(sources, start=1):
        tag = f"(Source {idx:02d})"
        # 답변이 없는 소스는 제외
        if src.get("answer"):
            numbered_msgs.append(SystemMessage(content=f"{tag} {src['answer']}"))
        if src.get("context"):
            merged_ctx += f"\n\n--- Source {idx} Context ---\n{src['context']}"
    
    if not numbered_msgs:
        return {"answer": "죄송합니다, 요청하신 내용에 대한 정보를 찾을 수 없습니다. 다른 질문을 해주시겠어요?", "context": ""}

    msgs = [SYSTEM_MANAGER] + numbered_msgs + [
        HumanMessage(content=f"label: {label}\n\n질문: {question}")]
    
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

def invoke(question: str, history: List[Dict[str, Any]] = None) -> Dict:
    """
    대화 이력을 받아 질문을 재구성한 후 라우팅 및 답변 생성을 수행.
    """
    history = history or []
    
    # 대화 이력을 바탕으로 질문 재구성
    rewritten_question = _rewrite_query_with_history(question, history)
    
    # 의도 분류
    label = route(rewritten_question)
    print(f'label: {label}\n')
    
    # 적절한 에이전트 실행 
    if label == "faq":
        return _run_faq(rewritten_question)
    if label == "product":
        return manager_agent(rewritten_question, label="product")
    if label == "advise":
        base = manager_agent(rewritten_question, label="advise")
        extra = advise(rewritten_question)
        return {
            "answer": base["answer"] + "\n\n---\n" + extra["answer"],
            "context": base["context"]
        }    
    return fallback_llm.invoke(question)
