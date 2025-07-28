# Finance Agent PoC


우리은행의 FAQ만을 대상으로 RAG 파이프라인을 빠르게 검증-시연한다.
LangChain + LangGraph + FAISS + OpenAI LLM, Streamlit UI를 사용하며 추후 예금·대출·보험·펀드 모듈로 확장 가능하도록 모듈 구조를 갖춘다.

## 1. 프로젝트 구조
```bash
woori-finconcierge/
├─ apps/
│  └─ streamlit_app/           # Streamlit 데모
│      └─ app.py
├─ data/
│  ├─ processed/
│  │   └─ faq_woori_structured.jsonl
├─ index/
│  └─ faq_faiss/               # FAISS 인덱스 & 메타
├─ scripts/
│  └─ build_faq_index.py       # 인덱스 생성 스크립트
├─ services/
│  └─ rag/
│      └─ faq_rag/
│          ├─ retriever.py
│          ├─ faq_chain.py
│          └─ __init__.py
├─ tests/
│  └─ test_e2e.ipynb           # E2E 노트북 테스트
├─ config.py                   # 환경변수·상수 중앙 관리
├─ requirements.txt
└─ README.md                   
```

## 2. 핵심 모듈

| 경로                                  | 역할                                    | 주요 포인트                                       |
| ----------------------------------- | ------------------------------------- | -------------------------------------------- |
| `config.py`                         | `.env` 로드, 모델·경로·튜닝 파라미터 상수화          | `OPENAI_API_KEY`, `LLM_MODEL`, `FAQ_TOP_K` 등 |
| `scripts/build_faq_index.py`        | FAQ JSONL → 임베딩 → FAISS 인덱스 저장        | 함수 `build_index()` 호출 가능                     |
| `data/embeddings/index_builder.py`  | 인덱스가 존재하면 로드, 없으면 빌드                  | 모든 RAG 모듈이 공유                                |
| `services/rag/faq_rag/retriever.py` | FAQ 전용 `VectorStoreRetriever`         | `k` 값은 `config.FAQ_TOP_K`                    |
| `services/rag/faq_rag/faq_chain.py` | Pydantic `FAQState` + LangGraph 파이프라인 | `retrieve_node` → `generate_node`            |
| `apps/streamlit_app/app.py`         | 간단한 사용자 UI                            | 질문 입력 → LLM 답변 + 근거 표시                       |


## 3. 사전 준비
 - OpenAI API Key – .env 파일에 OPENAI_API_KEY= 입력
 - 의존 라이브러리 설치
```bash
pip install -r requirements.txt
```

## 4. 환경 변수 예시 (.env)
```bash
# 필수
OPENAI_API_KEY=sk-********************************

```

## 5. 인덱스 생성
```bash
python scripts/build_faq_index.py
```

- data/processed/faq_woori_structured.jsonl 을 읽어 500-토큰 단위 Chunk + 메타데이터 생성
- OpenAI Embeddings → FAISS Index 작성 → index/faq_faiss/ 저장
- 이미 존재하면 재생성하지 않는다.

## 6. Streamlit 데모 실행
```bash
streamlit run apps/streamlit_app/app.py # 프로젝트 루트 경로에서 실행
```
- 브라우저가 열리면 FAQ를 한국어로 입력
- LLM이 FAQ 컨텍스트를 인용해 즉시 답변
- 근거 보기 토글로 검색된 context 확인

## 7. End-to-End 테스트
```bash
jupyter notebook tests/test_e2e.ipynb
```
셀 순서대로 실행하면

- 환경변수 로딩 → 인덱스 로드/빌드
- LangGraph 호출 → 샘플 질문 5개 응답
- ragas 평가로 relevancy/faithfulness/precision 리포트 확인