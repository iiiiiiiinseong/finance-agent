# services/advisor/advisor.py 

"""
가입 의사 탐지 및 상담 내용 요약 기능을 수행하는 모듈
"""
import re, pandas as pd
from pathlib import Path
from typing import List, Dict, Literal, Any

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from config import OPENAI_API_KEY, LLM_MODEL, ROOT_DIR

LLM_ADVISOR = ChatOpenAI(model=LLM_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
PDF_META_PATH = ROOT_DIR / "data" / "processed" / "product_deposit.jsonl"
PRODUCT_DF = pd.read_json(PDF_META_PATH, lines=True)

# --- 1. 가입 의사 및 상품명 추출 모델 ---
class SignupIntent(BaseModel):
    intent: Literal["confirm", "inquire", "deny"] = Field(
        description="사용자의 가입 의사를 분류합니다. 'confirm'은 명확한 가입 의사, 'inquire'는 추가 질문, 'deny'는 거절을 의미합니다."
    )
    product_name: str | None = Field(
        description="사용자가 가입하려는 상품의 정확한 이름입니다. 대화에서 식별되지 않으면 null입니다."
    )

SIGNUP_PROMPT = SystemMessage(content=(
    "당신은 사용자와 AI 상담사 간의 대화를 분석하여, 사용자의 '상품 가입 의사'와 '상품명'을 추출하는 AI입니다.\n"
    "대화의 전체 맥락을 파악하여 사용자의 마지막 말이 명확한 가입 신청인지, 단순 문의인지, 거절인지를 판단하세요.\n"
    "가입 의사가 확실하다면, 대화에서 언급된 상품명을 정확히 추출해야 합니다."
))

def _get_chat_history(history: List[Dict]) -> List:
    """Streamlit의 세션 기록을 Langchain 메시지 형식으로 변환"""
    buffer = []
    for msg in history:
        if msg.get("role") == "user":
            buffer.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "assistant":
            buffer.append(AIMessage(content=msg.get("content")))
    return buffer

def _check_signup_intent(question: str, history: List[Dict]) -> SignupIntent:
    """사용자의 가입 의사를 분류하고 상품명을 추출합니다."""
    parser = PydanticOutputParser(pydantic_object=SignupIntent)
    messages = [
        SIGNUP_PROMPT,
        SystemMessage(content=f"FORMAT_INSTRUCTIONS: {parser.get_format_instructions()}"),
        *_get_chat_history(history),
        HumanMessage(content=question)
    ]
    response = LLM_ADVISOR.invoke(messages)
    return parser.parse(response.content)

# --- 2. 대화 내용 요약 모델 ---
SUMMARY_PROMPT = SystemMessage(content=(
    "당신은 금융 상담 대화 내용을 간결하게 요약하는 AI입니다.\n"
    "전체 대화 이력을 바탕으로, 고객이 어떤 상품에 대해 문의했고, 최종적으로 어떤 상품에 가입하기로 결정했는지 핵심만 요약해주세요.\n"
    "이 요약본은 고객에게 이메일로 발송될 상담 기록입니다. 정중하고 명확한 문체로 작성하세요."
))

def _summarize_consultation(history: List[Dict], product_name: str) -> str:
    """대화 내용을 이메일 본문 형식으로 요약합니다."""
    messages = [
        SUMMARY_PROMPT,
        *_get_chat_history(history),
        HumanMessage(content=f"고객이 최종적으로 '{product_name}' 상품에 가입하기로 결정했습니다. 위 대화 내용을 바탕으로 상담 내역을 요약해주세요.")
    ]
    response = LLM_ADVISOR.invoke(messages)
    return response.content

# --- 3. 메인 advise 함수 ---
def advise(question: str, history: List[Dict], product_context: str) -> Dict[str, Any]:
    """가입 의사를 판단하고, 상황에 맞는 응답과 액션을 반환합니다."""
    intent_result = _check_signup_intent(question, history)

    if intent_result.intent == "confirm" and intent_result.product_name:
        # 가입 의사가 확인되면, 대화 요약 및 컨텍스트 저장
        summary = _summarize_consultation(history + [{"role": "user", "content": question}], intent_result.product_name)
        final_product_name = intent_result.product_name
        
        pdf_path_str = None
        try:
            matched_product = PRODUCT_DF[PRODUCT_DF['product'] == final_product_name]
            if not matched_product.empty:
                pdf_path_str = matched_product['pdf_path'].iloc[0]
                print(f"Advisor: Found PDF path for '{final_product_name}': {pdf_path_str}")
            else:
                print(f"Advisor: Could not find product '{final_product_name}' in metadata.")
                print("아마도 저장된 metadata랑 추출한 상품명이 달라서 발생한 문제일 수 있음\n")

        except Exception as e:
            print(f"Advisor: Error while looking up PDF path: {e}")

        # action과 signup_context를 포함하여 app.py로 전달
        return {
            "answer": f"네, '{final_product_name}' 상품으로 가입 상담을 도와드리겠습니다. 상담 내역을 이메일로 보내드릴까요? 사이드바에서 이메일 주소를 입력 후 '상담내역 메일로 받기' 버튼을 눌러주세요.",
            "action": "request_email_for_signup",
            "signup_context": {
                "product_name": final_product_name,
                "summary": summary,
                "pdf_path": pdf_path_str
            }
        }

    else:
        # 가입 의사가 아니면 일반적인 상담 응답 반환
        return {
            "answer": "추가적으로 궁금한 점이 있으신가요? 또는 다른 상품을 추천해드릴까요?",
            "action": "continue_advise",
            "context": ""
        }
