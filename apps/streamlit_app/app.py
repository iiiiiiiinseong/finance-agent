# apps/streamlit_app/app.py

"""
Streamlit FAQ RAG App (Chat UI)
root에서 실행. streamlit run apps/streamlit_app/app.py
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
from dotenv import load_dotenv
from services.rag.faq_rag.faq_chain import graph
from config import OPENAI_API_KEY

# ---- 초기화 -------------------------------------------------------
load_dotenv(Path(__file__).parents[2] / ".env")
assert OPENAI_API_KEY, "OPENAI_API_KEY is not set in .env"

st.set_page_config(page_title="우리은행 FAQ 챗봇 🏦", page_icon="🏦", layout="wide")
st.title("우리은행 FAQ RAG 데모 🏦")
st.markdown(
    "우리은행의 FAQ 관련 질문에 답해드립니다. "
    "왼쪽 사이드바에서 예시 질문을 선택하거나, 아래 입력창에 직접 질문을 입력해 보세요."
)

# ---- 세션 스테이트 ------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # [{"role": "user"/"assistant", "content": str}]
if "last_context" not in st.session_state:
    st.session_state.last_context = ""

def run_query(q: str):
    """그래프 실행 + 히스토리/컨텍스트 저장"""
    st.session_state.history.append({"role": "user", "content": q})
    with st.spinner("AI가 답변을 생성 중입니다..."):
        res = graph.invoke({"question": q})
    answer = res.get("answer", "죄송합니다. 답변을 준비 중입니다.")
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.session_state.last_context = res.get("context", "") or ""

# ---- 사이드바: 예시 질문 ------------------------------------------
EXAMPLES = [
    "해외에서 인터넷뱅킹 쓰려면 출국 전 확인할 점?",
    "OTP 분실 시 거래 제한 해제 방법 알려줘",
    "인터넷뱅킹 이용자비밀번호 오류 해제 가능?",
    "간편이체 서비스 가입 절차가 궁금해",
    "지정 단말기에서만 접속하도록 설정할 수 있나요?",
]
with st.sidebar:
    st.header("예시 질문")
    for i, q in enumerate(EXAMPLES):
        if st.button(q, key=f"ex_{i}"):
            run_query(q)

# ---- 사용자 입력(중복 placeholder 전달 금지!) -----------------------
query = st.chat_input(placeholder="예) OTP 분실 시 어떻게 하나요?", key="chat_input")
if query:
    run_query(query)

# ---- 대화 내역 표시 ------------------------------------------------
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# ---- 근거 보기 -----------------------------------------------------
if st.session_state.history and st.session_state.history[-1]["role"] == "assistant":
    with st.expander("🔍 마지막 답변의 근거 보기"):
        st.code(st.session_state.last_context or "근거 문서를 찾지 못했습니다.")