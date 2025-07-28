| Section             | 내용                                                                                                              |
| ------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Product name**    | Woori AI Financial Concierge (PoC)                                                                              |
| **Goal**            | 예금·대출·보험·펀드 상담 → 가입까지 단일 대화 flow 구현; 90 % 정답률, 응답 ≤3 s, 상담-to-가입 전환율 ≥10 %                                      |
| **Personas**        | ① 2030 모바일 고객, ② 4050 대면 선호 고객, ③ 콜센터 상담사(Back-office assist)                                                   |
| **User stories**    | *US-01* 질문하면 3 초 내 답을 받는다.<br>*US-02* 여러 상품을 비교해 표로 보고, 바로 한 상품을 신청한다.<br>*US-03* 가입 진행 중 약관을 한국어 간단 요약으로 물어본다. |
| **Scope (PoC)**     | <ul><li>대화형 FAQ & 상품 Q\&A</li><li>4개 상품군 비교 RAG</li><li>전자서명 Stub (mock)</li></ul>                              |
| **Out of scope**    | 실시간 금리·환율 API, 대출 한도 계산, 실제 계좌 개설                                                                               |
| **Success metrics** | Top-1 answer EM ≥ 0.9, CSAT ≥ 4.5/5, 가입 흐름 drop-off ≤ 30 %                                                      |
| **Assumptions**     | Woori Bank dev-sandbox API 제공, 개인정보는 hashed profile 사용                                                          |
| **Risks**           | 금융 규제 준수, hallucination, 데이터 보안                                                                                 |
| **Milestones**      | M0 Kick-off, M1 데이터 수집(1w), M2 RAG MVP(2w), M3 Advisor & Signup stub(1w), M4 E2E demo + metrics(1w)             |
