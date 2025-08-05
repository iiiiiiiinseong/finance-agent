# services/rag/product_rag/chain.py

"""
chain.py
------------
product RAG LangGraph 파이프라인.
"""
from langgraph.graph import StateGraph
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from config import LLM_MODEL, OPENAI_API_KEY
from .retriever import DepositRetriever
from .retriever import _infer_category

from pydantic import BaseModel, Field
from typing import List, Optional

class ProductQA(BaseModel):
    question: str
    answer: Optional[str] = None
    context: str = ""                    # 기본값 빈 문자열
    contexts: List[str] = Field(default_factory=list)

class ProductDoc(BaseModel):
    product: str
    section: str
    file_date: str

SYS_PROMPT = SystemMessage(
    content=(
        "너는 우리은행 예금·적금 전문 상담사다.\n"
        "아래 [Context]를 참고하여 질문에 한국어로 답하세요.\n"
        "• 근거 문장마다 인용번호(①,②…)를 붙여 근거를 명시한다.\n"
        "• 문서에 정보가 없으면 '해당 내용은 담당 직원 확인 후 안내'라고 답한다."
    )
)

llm = ChatOpenAI(model=LLM_MODEL, 
                 temperature=0,
                 openai_api_key=OPENAI_API_KEY)
retriever = DepositRetriever(k=3)

def retrieve_node(state: ProductQA):
    # 질문에서 예금/적금/통장 … 카테고리 추론
    cat = _infer_category(state.question)
    # 카테고리 지정 필터 검색
    docs = retriever(state.question, category=cat)
    ctx_list = [d.page_content for d in docs]
    return {
        "question": state.question,
        "contexts": ctx_list,          # RAGAS 용 List[str]
        "context": "\n\n".join(ctx_list)  # 사람이 볼 때만 사용
    }

def generate_node(state: ProductQA):
    numbered = "\n".join(f"①{i+1}" + " " + c for i, c in enumerate(state.contexts))
    messages = [
        SYS_PROMPT,
        SystemMessage(content=f"[Context]\n{numbered}"),
        HumanMessage(content=state.question),
    ]
    answer = llm.invoke(messages).content
    return {
        "answer": answer,
        "contexts": state.contexts,
        "context": state.context,
    }

# ---- Graph 정의 ----
builder = StateGraph(state_schema=ProductQA)
builder.add_node("retrieve_node", retrieve_node)
builder.add_node("generate_node", generate_node)

builder.set_entry_point("retrieve_node")
builder.add_edge("retrieve_node", "generate_node")

graph = builder.compile()
