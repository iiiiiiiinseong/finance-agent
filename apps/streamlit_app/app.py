# apps/streamlit_app/app.py

"""
Streamlit FAQ RAG App (Chat UI)
root에서 실행. streamlit run apps/streamlit_app/app.py
"""
from pathlib import Path
import sys, json, collections
import streamlit as st
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))

from services.rag.faq_rag.faq_chain import graph
from config import OPENAI_API_KEY

# ---- 초기화 -------------------------------------------------------
load_dotenv(Path(__file__).parents[2] / ".env")
assert OPENAI_API_KEY, "OPENAI_API_KEY is not set in .env"

st.set_page_config(page_title="우리은행 FAQ 챗봇 🏦", page_icon="🏦", layout="wide")
st.title("우리은행 FAQ RAG 데모 🏦")
st.markdown(
    "우리은행 FAQ 관련 질문에 답해드립니다. "
    "왼쪽 사이드바에서 **[주제 ▸ 세부항목 ▸ 예시 질문]**을 골라보거나, 아래 입력창에 직접 질문을 입력해 보세요."
)

# ---- Helper -------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "faq_woori_structured.jsonl"

def default_subdict():
    return collections.defaultdict(list)

@st.cache_data(show_spinner=False)
def load_example_tree():
    """JSONL 로부터 {topic: {subcategory: [questions]}} 딕셔너리 구성 후 예시질문 리스트로 사용"""
    tree = collections.defaultdict(default_subdict)
    with DATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            topic = row.get("topic", "기타")
            subcat = row.get("subcategory", "기타")
            q = row.get("question")
            if q:
                tree[topic][subcat].append(q)

    # 각 subcategory 당 앞쪽 3개만 노출 (과다 표출 방지)
    for topic in tree:
        for sub in tree[topic]:
            tree[topic][sub] = tree[topic][sub][:3]
    return tree

EXAMPLE_TREE = load_example_tree()


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
with st.sidebar:
    st.header("예시 질문")
    for topic, sub_dict in EXAMPLE_TREE.items():
        with st.expander(topic, expanded=False):
            for subcat, q_list in sub_dict.items():
                st.markdown(f"**{subcat}**")
                for idx, q in enumerate(q_list):
                    btn_key = f"ex_{topic}_{subcat}_{idx}"
                    if st.button(q, key=btn_key):
                        run_query(q)

# ---- 사용자 입력 ----------------------------------------------------
placeholder_example = ""
try:
    placeholder_example = next(iter(next(iter(EXAMPLE_TREE.values())).values()))[0]
except StopIteration:
    placeholder_example = "질문을 입력해 주세요"

query = st.chat_input(placeholder=f"예) {placeholder_example}", key="chat_input")
if query:
    run_query(query)


# ---- 대화 내역 표시 ------------------------------------------------
for msg in st.session_state.history:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])

# ---- 근거 보기 -----------------------------------------------------
if st.session_state.history and st.session_state.history[-1]["role"] == "assistant":
    with st.expander("🔍 마지막 답변의 근거 보기"):
        st.code(st.session_state.last_context or "근거 문서를 찾지 못했습니다.")