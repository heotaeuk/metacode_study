# 🚢 ShipLeak-Copilot

선박 배관 및 밸브 누설 진단을 위한 RAG(Retrieval-Augmented Generation) 기반 AI Assistant

---

## 프로젝트 개요

ShipLeak-Copilot은 선박 배관 및 밸브의 누설(Leakage), 이상 진동(Vibration), 소음(Noise) 관련 정비 지식을 검색하고 AI가 진단 결과를 제공하는 챗봇 시스템입니다.

사용자는 자연어 질문을 입력하면 관련 정비 매뉴얼과 Trouble Shooting Guide를 검색한 후 GPT가 근거 기반 답변을 생성합니다.

---

## 프로젝트 목표

- 선박 밸브 이상 증상 진단
- 정비 절차 자동 안내
- Trouble Shooting Guide 검색
- 정비 지식 데이터베이스 구축
- RAG 기반 질의응답 구현

---

## 시스템 구조

사용자 질문

↓

OpenAI Embedding

↓

ChromaDB Similarity Search

↓

관련 문서 검색

↓

GPT-4o-mini

↓

AI 진단 결과 생성

---

## 사용 기술

### Front-End

- Streamlit

### Vector DB

- ChromaDB

### Embedding

- OpenAI text-embedding-3-small

### LLM

- GPT-4o-mini

### Language

- Python

---

## 데이터셋

Valve Maintenance Manual

Valve Inspection Procedure

Valve Trouble Shooting Guide

Valve Quality Inspection Guide

Valve Leakage Diagnosis Guide

---

## Feature

본 프로젝트는 다음 진동 특징 정보를 활용합니다.

- RMS
- FFT Peak
- Dominant Frequency
- Spectral Centroid
- Crest Factor
- MFCC

---

## 구현 기능

### 1. RAG 검색

사용자 질문과 가장 유사한 문서를 검색

### 2. AI 진단

GPT를 활용한 정비 절차 생성

### 3. 근거 문서 제공

검색된 문서를 함께 표시하여 신뢰성 확보

### 4. 유사도 표시

검색 결과의 Similarity Score 제공

---

## 실행 방법

### 패키지 설치

```bash
pip install streamlit chromadb langchain-openai python-dotenv
```

### 실행

```bash
streamlit run app.py
```

---

## 예시 질문

- Seat Leakage 증상은?
- Cavitation 원인은?
- RMS 증가 시 조치사항은?
- 밸브 점검 절차는?

---

## 결과 예시

AI 진단 결과

- 대표 증상
- 원인 분석
- 조치 절차
- 정비 후 확인

검색 근거 문서

- Trouble Guide
- Maintenance Manual
- Inspection Procedure

---

## 기대 효과

- 정비 시간 단축
- 지식 검색 자동화
- 정비 품질 향상
- 신규 엔지니어 교육 지원

---

## Author

허태욱

ShipLeak-Copilot Project