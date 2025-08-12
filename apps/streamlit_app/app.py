# apps/streamlit_app/app.py

"""
Streamlit finance App (Chat UI)
root에서 실행. streamlit run apps/streamlit_app/app.py
"""
from pathlib import Path
import sys, json, collections, io, re, csv, textwrap, pandas as pd
import pprint
import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))

from services.orchestrator.router_node import invoke as router_invoke
from services.email.adapter import send_email_with_resp
from config import LLM_MODEL, OPENAI_API_KEY
import base64

# ---- 초기화 -------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env.streamlit")
assert OPENAI_API_KEY, "OPENAI_API_KEY is not set in .env"

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="우리은행 AI 컨시어지 챗봇 🏦", page_icon="🏦", layout="wide")
st.title("우리은행 AI 컨시어지 데모 🏦")
st.markdown(
    """
    우리은행 FAQ, 예금/적금, 입출금 상품 관련 질문에 답해드립니다.

    왼쪽 사이드바에서 FAQ 예시 질문을 골라보거나, 맞춤 상품을 찾아보세요.
    """
)

# ---- Helper -------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "faq_woori_structured.jsonl"

def pdf_viewer(pdf_path: str, height: int = 700):
    """
    주어진 경로의 PDF를 Base64로 인코딩하고, <embed> 태그를 사용하여 표시합니다.
    파일을 찾을 수 없는 경우 오류 메시지를 표시합니다.
    """
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        
        pdf_display = (
            f'<embed src="data:application/pdf;base64,{base64_pdf}" '
            f'width="100%" height="{height}" type="application/pdf">'
        )
        st.markdown(pdf_display, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("오류: PDF 파일을 찾을 수 없습니다.")

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

@st.cache_resource
def load_meta_df():
    path = Path(__file__).resolve().parents[2] / "data" / "processed" / "product_deposit.jsonl"
    return pd.read_json(path, lines=True)

META_DF = load_meta_df()

# 추천 함수
def recommend(profile: dict, top_n: int = 3):
    df = META_DF
    # 필터: 카테고리·기간·방식 등 (단순 예시)
    if profile["dtype"]:
        df = df[df.product_category.str.contains(profile["dtype"])]
    if profile["term"]:
        df = df[df.term_options.apply(lambda opts: profile["term"] in opts)]
    df = df.sort_values("max_rate", ascending=False).head(top_n)
    return df.reset_index(drop=True)

def _call_gpt_csv(raw: str) -> str:
    sys =  (
    "너는 은행 상품설명서 분석 봇이다.\n"
    "- 반드시 두 열 CSV(항목,내용)로 출력한다.\n"
    "- 행은 [금리, 중도해지, 세율, 우대 조건] 중 최소 2가지는 포함해야 한다.\n"
    "- 같은 항목이 여러 줄이면 하나로 합쳐도 된다.\n"
    "- 세율·세금·우대이율 설명이 섞여도 열 개수는 2로 맞춘다."
)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":textwrap.shorten(raw, 9000)}],
        temperature=0,
        response_format={"type":"text"},
        max_tokens=500,
    ).choices[0].message.content.strip("` \n")

    # debug_messages=[{"role":"system","content":sys},
    # {"role":"user","content":textwrap.shorten(raw, 9000)}]
    # print('='*100)
    # pprint.pp(debug_messages)
    # print('='*100)
    return resp

@st.cache_data(show_spinner=False)
def parse_table(pdf_path: str) -> pd.DataFrame:
    """
    거래 조건 섹션을 2-열(항목·내용) DataFrame 표준화.
    - LLM CSV 행별 열 수 달라도 처리
    - 실패 시 regex fallback
    """
    # 1) PDF 첫 3쪽 텍스트 로드 (PyMuPDF)
    pages = PyMuPDFLoader(pdf_path).load()[:3]
    raw = "\n".join(p.page_content for p in pages)

    # 2) LLM 프롬프트 → CSV
    try:
        csv_text = _call_gpt_csv(raw)
        # pprint.pp(csv_text) # 디버깅 용
        rows = list(csv.reader(io.StringIO(csv_text)))

        # 유효한 레코드만 추출 (두 칸 이상인 경우만)
        records = []
        for row in rows:
            if len(row) < 2:
                continue

            item = row[0].strip()
            content_list = []
            for cell in row[1:]:
                content_list.append(cell.strip())
            content = " ".join(content_list)

            records.append((item, content))

        if not records:
            raise ValueError("empty")

        # 데이터프레임 생성 및 인덱스 리셋
        df = pd.DataFrame(records, columns=["항목", "내용"])
        return df.reset_index(drop=True)
    
    except Exception as e:
        # 3) regex fallback (금리·세율·중도해지 라인만)
        pattern = r"(금리[^:\n]*[:\s]\s*[^\n]{1,80}|세율[^:\n]*[:\s]\s*[^\n]{1,80}|중도해지[^:\n]*[:\s]\s*[^\n]{1,80})"
        rows = re.findall(pattern, raw)
        if not rows:
            return pd.DataFrame()  # 빈 DF → 추출 실패 처리
        data = [r.split(maxsplit=1) for r in rows]
        df = pd.DataFrame(data, columns=["항목","내용"])
        return df

# ---- 세션 스테이트 ------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # [{"role": "user"/"assistant", "content": str}]
if "last_context" not in st.session_state:
    st.session_state.last_context = ""
if "view_pdf" not in st.session_state:
    st.session_state.view_pdf = None 
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

def run_query(q: str):
    """그래프 실행 + 히스토리/컨텍스트 저장"""
    # 이전 히스토리 로드
    previous_history = st.session_state.history or []
    print(f"DEBUG:\n previous_history:{previous_history}\n")
    st.session_state.history.append({"role": "user", "content": q})
    
    with st.spinner("AI가 답변을 생성 중입니다..."):
        res = router_invoke(q, history=previous_history)
    
    answer = res.get("answer", "죄송합니다. 답변을 준비 중입니다.")
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.session_state.last_context = res.get("context", "")

    # advise 라벨이면 추천 표 자동 펼치기
    if "가입 상담" in answer or "추천" in answer:
        st.sidebar.write("### 추천 결과는 사이드바를 확인하세요!")
        first_pdf = rec_df.iloc[0]["pdf_path"]
        tbl_md = parse_table(first_pdf).head(8).to_markdown(index=False)
        answer += "\n\n**주요 조건 요약**\n" + tbl_md
        st.session_state.history[-1]["content"] = answer

# ---- 사이드바: 예시 질문 ------------------------------------------
with st.sidebar:
    st.header("FAQ 예시 질문")
    for topic, sub_dict in EXAMPLE_TREE.items():
        with st.expander(topic, expanded=False):
            for subcat, q_list in sub_dict.items():
                st.markdown(f"**{subcat}**")
                for idx, q in enumerate(q_list):
                    btn_key = f"ex_{topic}_{subcat}_{idx}"
                    if st.button(q, key=btn_key):
                        run_query(q)

# ---- 사이드바: 조건 수집 & 추천 표 ------------------------------------------
with st.sidebar:
    st.header("맞춤 적금/예금 추천")

    # 1) 조건 입력 폼
    with st.form("cond_form"):
        amt = st.number_input("월 납입액(만원)", 1, 1000, 30)
        term = st.selectbox("기간", ["6개월", "1년", "2년", "3년", "5년"])
        dtype = st.radio("상품 유형", ["적금", "예금", "입출금통장"], horizontal=True)
        submitted = st.form_submit_button("추천 받기")
    if submitted:
        st.session_state.profile = {"amt": amt, "term": term, "dtype": dtype}

    def _on_email_change():
        email_val = st.session_state.get("_email_input", "").strip()
        st.session_state.user_email = email_val

    st.divider()
    st.subheader("상품설명서 메일 수신")
    st.text_input(
        "받으실 이메일 주소",
        key="_email_input",
        value=st.session_state.get("user_email", ""),
        placeholder="you@example.com",
        on_change=_on_email_change,
    )
    
    # 2) 추천 결과 표시 (개선된 코드)
    if "profile" in st.session_state:
        rec_df = recommend(st.session_state.profile)

        st.subheader("추천 상품 Top 3")
        st.dataframe(
            rec_df[["product", "max_rate", "term_options"]],
            hide_index=True,
            use_container_width=True,
        )

        st.divider()

        for _, row in rec_df.iterrows():
            # 상품명과 금리 표시
            st.markdown(f"**{row['product']}** &nbsp; *(최대금리&nbsp;{row['max_rate']}%)*")

            # PDF 경로 및 존재 여부를 한 번만 확인
            pdf_path_str = row.get("pdf_path")
            pdf_exists = isinstance(pdf_path_str, str) and Path(pdf_path_str).exists()

            # 3개의 컬럼 생성
            col1, col2, col3 = st.columns(3)

            if pdf_exists:
                pdf_path = Path(pdf_path_str)
                
                # --- PDF 다운로드 버튼 ---
                with col1:
                    with open(pdf_path, "rb") as fp:
                        st.download_button(
                            label="PDF다운로드",
                            data=fp.read(),
                            file_name=pdf_path.name,
                            mime="application/pdf",
                            key=f"dl_{row['product_code']}",
                            use_container_width=True,
                        )
                
                # --- PDF 보기 버튼 ---
                with col2:
                    if st.button("미리보기", key=f"view_{row['product_code']}", use_container_width=True):
                        st.session_state.view_pdf = pdf_path_str
                
                # --- 메일로 전송 버튼 ---
                with col3:
                    can_send = bool(st.session_state.get("user_email"))
                    if st.button("메일로", key=f"mail_{row['product_code']}", disabled=not can_send, use_container_width=True):
                        with st.spinner("메일 전송 중…"):
                            resp = send_email_with_resp(
                                to_addr=st.session_state.user_email,
                                subject=f"[우리은행] {row['product']} 상품설명서",
                                pdf_path=pdf_path_str,
                                text="첨부된 상품설명서를 확인해 주세요.",
                                html="첨부된 상품설명서를 확인해 주세요.",
                            )
                        if resp.ok:
                            st.success("발송 완료!")
                        else:
                            code = resp.status_code
                            st.error(f"전송 실패({code}). {resp.detail or ''}")
            else:
                # PDF 파일이 없을 경우
                st.caption("PDF 파일 준비 중입니다.")
            
            st.markdown("---") # 상품별 구분선

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

# ---- PDF 뷰어 -------------------------------------------------
if st.session_state.view_pdf:
    st.markdown("---")
    with st.expander("📄 상품설명서 미리보기"):
        with st.container(height=720, border=True):
            with st.spinner("PDF 로딩 중…"):
                pdf_viewer(st.session_state.view_pdf, height=700)

# ---- 근거 보기 -----------------------------------------------------
if st.session_state.history and st.session_state.history[-1]["role"] == "assistant":
    with st.expander("🔍 마지막 답변의 근거 보기"):
        st.code(st.session_state.last_context or "근거 문서를 찾지 못했습니다.")