# services/rag/faq_rag/faq_chain.py

"""
faq_chain.py
------------
FAQ 전용 LangGraph 파이프라인.
"""
from __future__ import annotations

from langgraph.graph import StateGraph
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings

from config import LLM_MODEL, OPENAI_API_KEY
from .retriever import faq_retriever 

class FAQState(BaseModel):
    question: str
    context: str | None = None
    contexts: list[str] | None = None  # RAGAS용 리스트 컨텍스트
    answer: str  | None = None

sys_prompt = (
    "당신은 우리은행 FAQ 상담원입니다. 반드시 <context>의 사실만 사용해 간결하게 답하세요. "
    "컨텍스트에 관련 정보가 하나라도 있으면 절대로 거절하지 마세요. "
    "절차/경로/메뉴는 단계형(1. 2. 3.)으로 정리하세요. "
    "근거가 불충분하면 '추가 확인 필요'라고 덧붙이되, 추측은 하지 마세요."
)

prompt = PromptTemplate.from_template(
    """
    당신은 우리은행 FAQ 전문 상담원입니다. <context>의 사실만 사용해 간결하게 답하세요.
    
    아래의 규정에 따라 상담을 해주세요
    
    1. FAQ 자료를 참고해 질문에 답하세요.
    2. 컨텍스트에 관련 정보가 하나라도 있으면 절대로 거절하지 마세요. 
    3. 절차/경로/메뉴는 단계형(1. 2. 3.)으로 정리하세요.
    4. 근거가 불충분하면 '추가 확인 필요'라고 덧붙이되, 추측은 하지 마세요

    <context>
    {context}
    </context>
    질문: {question}
    답변:"""
)

llm = ChatOpenAI(model=LLM_MODEL, 
                 temperature=0,
                 openai_api_key=OPENAI_API_KEY)

def retrieve_node(state: FAQState) -> dict:
    q = state.question
    docs = faq_retriever.invoke(q)
    contexts_list = [d.page_content for d in docs]
    context_text = "\n\n".join(contexts_list)
    return {
        "question": q,
        "context": context_text,    # 사람 읽기용
        "contexts": contexts_list,  # RAGAS용(List[str])
    }

def generate_node(state: FAQState) -> dict:
    # 컨텍스트가 없는 예외 상황
    if not state.contexts or len(state.contexts) == 0:
        return {"answer": "죄송합니다. 준비중입니다.", "contexts": state.contexts, "context": state.context}

    answer_msg = llm.invoke(prompt.format(**state.dict()))
    answer_text = getattr(answer_msg, "content", str(answer_msg))
    return {
        "answer": answer_text,
        "contexts": state.contexts, # RAGAS용(List[str])
        "context": state.context,   # 사람 읽기용 문자열도 보존
    }

# Graph: question → retrieve → generate
builder = StateGraph(state_schema=FAQState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")

graph = builder.compile()
