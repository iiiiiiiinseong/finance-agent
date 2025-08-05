# Finance Agent PoC


우리은행의 FAQ만을 대상으로 RAG 파이프라인을 빠르게 검증-시연한다.
LangChain + LangGraph + FAISS + OpenAI LLM, Streamlit UI를 사용하며 추후 예금·대출·보험·펀드 모듈로 확장 가능하도록 모듈 구조를 갖춘다.

## 1. 프로젝트 구조
```bash
woori-finconcierge/
├─ apps/
│  └─ streamlit_app/              # Streamlit 데모
│      └─ app.py
├─ data/
│  ├─ processed/
│  │   ├─ faq_woori.py
│  │   └─ faq_woori_structured.jsonl
│  └─ raw_docs                    # PDF·FAQ 원본
├─ index/                         # FAISS 인덱스 & 메타
│  ├─ deposit_faiss           
│  └─ faq_faiss/
├─ scripts/
│  ├─ build_deposit_index.py  
│  └─ build_faq_index.py       
│  └─ ragas_eval.py
├─ services/
│   ├─ rag/
│   │   ├─ faq_rag/
│   │   │   ├─ retriever.py
│   │   │   └─ faq_chain.py
│   │   └─ product_rag/
│   │       ├─ retriever.py
│   │       └─ chain.py
│   ├─ orchestrator/
│   │   └─ router_node.py         # GPT Classifier + Manager Agent + fallback
│   ├─ fallback/
│   │   └─ fallback_llm.py        # 범용 안내 LLM 노드
│   └─ advisor/                   # 가입 Stub (TODO)
│       └─ advisor_stub.py                   
├─ tests/                         # E2E 노트북 테스트
│   ├─ test_FAQ_rag_e2e.ipynb
│   ├─ test_product_rag_e2e.ipynb
│   ├─ test_manager_agent.ipynb
│   └─ RAGchecker_framework.ipynb       
├─ config.py                      # 환경변수·상수 중앙 관리
├─ requirements.txt
└─ README.md                   
```

## 2. 핵심 모듈

| 경로 | 역할 | 주요 포인트 |
|-----------------|------|------------|
| <br>**환경변수** | | |
| `config.py` | API 키·모델·경로·튜닝 파라미터 중앙 관리 | `OPENAI_API_KEY`, `LLM_MODEL`, `CHUNK_SIZE`, `FAQ_TOP_K` 등 |
| <br>**데이터 ETL** | | |
| `data/loader/product_loader.py` | PDF 상품설명서 → 섹션-단위 `Document` 리스트 | 정규식 헤더 추출·Mecab token chunking |
| `data/processed/faq_woori_structured.jsonl` | FAQ 원본 32 건 구조화 | topic·subcategory 메타 포함 |
| <br>**인덱스 빌드** | | |
| `scripts/build_faq_index.py` | FAQ JSONL → Embedding → `index/faq_faiss/` | OpenAI `text-embedding-3-small` |
| `scripts/build_deposit_index.py` | 예금/적금 PDF → `index/deposit_faiss/` | 배치 임베딩 `chunk_size=200` |
| <br>**RAG – FAQ** | | |
| `services/rag/faq_rag/retriever.py` | FAISS + BM25 하이브리드 검색 | `k`=`config.FAQ_TOP_K` |
| `services/rag/faq_rag/faq_chain.py` | LangGraph `retrieve → generate` | 근거 인용번호 ①② 첨부 |
| <br>**RAG – 예금/적금 상품설명** | | |
| `services/rag/product_rag/retriever.py` | Deposit FAISS + BM25 검색 | 메타필터 `product_code` |
| `services/rag/product_rag/chain.py` | LangGraph 파이프라인 | FAQ 답변과 동일 포맷 |
| <br>**Fallback / Router** | | |
| `services/fallback/fallback_llm.py` | 비금융·스몰톡 질문 안내 LLM 노드 | Kiwi few-shot 가드레일 |
| `services/orchestrator/router_node.py` | GPT Classifer + Manager Agent + Fallback | FAQ·Product 병렬 호출 후 병합 |
| <br>**Advisor (Stub)** | | |
| `services/advisor/advisor_stub.py` | 가입 추천/전자서명 기능 MVP 자리 | 추후 MCP 연동 예정 |
| <br>**UI / Demo** | | |
| `apps/streamlit_app/app.py` | FAQ·상품 설명 탭, 예시 트리, 근거 토글 | `router_node.invoke()` 호출 |
| <br>**평가 & 테스트** | | |
| `tests/test_FAQ_rag_e2e.ipynb` | FAQ RAG → RAGAS 평가 | answer_relevancy·faithfulness |
| `tests/test_product_rag_e2e.ipynb` | 상품설명 RAG E2E 테스트 | 동일 지표 |
| `tests/RAGchecker_framework.ipynb` | rag-checker (Kiwi tokenizer) 한국어 faithfulness | Mecab/kiwipiepy 기반 |
| `tests/test_manager_agent.ipynb` | Router+Manager 전체 흐름 검증 | fallback·advise 포함 |



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
python scripts/build_deposit_index.py
```

- data를 읽어 Chunk + 메타데이터 생성
- OpenAI Embeddings → FAISS Index 작성 → index/*_faiss/ 저장
- 이미 존재하면 재생성하지 않는다.

## 6. Streamlit 데모 실행
```bash
streamlit run apps/streamlit_app/app.py # 프로젝트 루트 경로에서 실행
```
- 브라우저가 열리면 FAQ 혹은 상품질의를 한국어로 입력
- LLM이 컨텍스트를 인용해 즉시 답변
- 근거 보기 토글로 검색된 context 확인

## Demo
![FAQ RAG Demo](./docs/Manager_agent_high.gif)

## 7. End-to-End 테스트
```bash
jupyter notebook test_FAQ_rag_e2e.ipynb
jupyter notebook test_product_rag_e2e.ipynb
jupyter notebook test_manager_agent
```
셀 순서대로 실행하면

- 환경변수 로딩 → 인덱스 로드/빌드
- LangGraph 호출 → 샘플 질문 응답
- ragas 평가로 relevancy/faithfulness/precision 리포트 확인