# ShipLeak-Copilot Valve RAG Vector DB

## 구성
- `vector_db/chunks.csv` : 정비매뉴얼, Trouble Guide, 점검절차서, 분석보고서, 발표자료, Q&A 문서의 Chunk 데이터
- `vector_db/tfidf_matrix.npz` : 로컬 Vector DB 행렬
- `vector_db/tfidf_vectorizer.pkl` : 벡터 변환기
- `retriever.py` : 질의 검색 실행 코드
- `build_chroma_openai.py` : OpenAI Embedding + ChromaDB 구축용 선택 코드

## 현재 구축 결과
- 문서 수: 6개
- Chunk 수: 19개
- 구축 방식: TF-IDF Vector + Cosine Similarity
- 장점: API Key 없이 로컬 실행 가능

## 실행 방법
```bash
cd valve_vector_db_project
python retriever.py
```

## 예시 질문
- RMS가 높으면 어떤 조치를 해야 하나요?
- Seat Leakage의 증상과 조치는 무엇인가요?
- FFT 분석은 왜 필요한가요?
- RAG 구축을 위해 어떤 문서를 Vector DB에 넣었나요?
- 점검 절차는 어떤 순서로 수행하나요?

## 발표용 설명
본 Vector DB는 2주차에서 작성한 정비매뉴얼, Trouble Guide, 점검절차서, 데이터 분석 보고서 및 교수 Q&A 문서를 Chunking한 뒤 벡터화하여 구축하였다.  
다음 단계에서는 사용자의 질문을 벡터로 변환하고, Vector DB에서 관련 Chunk를 검색한 뒤 LLM에 전달하여 근거 기반 답변을 생성하는 RAG 구조로 확장한다.