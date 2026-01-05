# ConsultAI

> **AI-Powered Consultation Analysis Platform for Call Centers**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ConsultAI는 콜센터 상담 데이터를 AI로 실시간 분석하는 플랫폼입니다. STT 변환된 음성 데이터를 처리하여 상담사와 관리자가 즉시 활용할 수 있는 구조화된 정보를 자동 생성합니다.

---

## ✨ 주요 기능

### 🚀 듀얼-티어 AI 아키텍처

#### **SLM (Small Language Models)** - 실시간 상담 지원
- **Qwen3-1.7B**: 100% 성공률, 평균 **2.83초** ⚡
- **목표**: 1-3초 이내 즉시 요약 제공
- **용도**: 상담 중 실시간 지원

#### **LLM (Large Language Models)** - 배치 분석
- **Qwen3-4B**: 100% 성공률, 평균 **20.85초**, 품질 **0.990** 🔥
- **목표**: 15-20초 이내 고품질 분석
- **기능**: 요약 + 키워드 추출 + 제목 생성

### 📊 핵심 분석 기능
- ✅ **요약 생성**: 3줄 구조 (고객/상담사/결과)
- ✅ **키워드 추출**: 맥락 기반 1-3개 핵심 키워드
- ✅ **제목 생성**: 키워드형/서술형 타이틀
- ✅ **품질 검증**: 환각 차단 및 자동 품질 점수

### 🔌 센터링크 연동 API
- **실시간 API**: 바운드 키 인증, 1-3초 응답
- **배치 API**: 비동기 처리, 콜백 시스템
- **개발 전용 API**: 인증 없는 빠른 테스트 (DEBUG 모드)

### ✅ 검증된 성능
- **999개** 실제 통화 데이터 테스트 완료
- **100%** 성공률, 평균 품질 **0.990/1.00**
- **6가지** STT JSON 형식 완벽 지원

---

## 🚀 빠른 시작

### 필수 요구사항

- Python 3.10+
- CUDA 12.x (GPU 사용 시)
- 16GB+ RAM (32GB 권장)

### 설치

```bash
# 1. 저장소 클론
git clone https://github.com/YOUR_USERNAME/consultai.git
cd consultai

# 2. Conda 환경 생성
conda create -n consultai python=3.10
conda activate consultai

# 3. 의존성 설치
pip install -r requirements.txt
```

### 모델 다운로드

**옵션 A: 자동 다운로드**
```bash
python scripts/core/download_models.py
```

**옵션 B: 수동 다운로드**

HuggingFace에서 다운로드 후 `models/` 폴더에 저장:
- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [Qwen3-4B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

```
models/
├── Qwen3-1.7B/
└── Qwen3-4B/
```

### 환경 설정

```bash
# .env 파일 생성
cp .env.example .env
```

`.env` 파일 수정:
```bash
DEBUG=true
HOST=0.0.0.0
PORT=8000
BOUND_KEYS=your_secure_key_min_20_characters
```

### 서버 실행

```bash
# 개발 서버
python main.py

# 프로덕션 서버
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API 테스트

**브라우저**:
- API 문서: http://localhost:8000/docs
- 대시보드: http://localhost:8000/static/consultation_dashboard.html

**cURL (개발 전용 API - 인증 없음)**:
```bash
curl -X POST http://localhost:8000/api/v1/dev/realtime-analyze-no-auth \
  -H "Content-Type: application/json" \
  -d '{
    "consultation_id": "TEST_001",
    "stt_data": {
      "conversation_text": "상담사: 안녕하세요.\n고객: 보험 문의합니다."
    }
  }'
```

**cURL (운영 API - 바운드 키 인증)**:
```bash
curl -X POST http://localhost:8000/api/v1/consultation/realtime-analyze \
  -H "X-Bound-Key: test_key_centerlink_2025" \
  -H "Content-Type: application/json" \
  -d '{
    "bound_key": "test_key_centerlink_2025",
    "consultation_id": "TEST_001",
    "stt_data": {
      "conversation_text": "상담사: 안녕하세요.\n고객: 보험 문의합니다."
    }
  }'
```

---

## 📖 문서

### 빠른 시작
- [빠른 테스트 가이드](docs/QUICK_TEST_START.md) - 2-5분 안에 시작하기
- [개발 전용 API](docs/DEV_API_GUIDE.md) - 인증 없는 빠른 테스트

### 상세 문서
- [API 명세서](docs/API_SPECIFICATION_CENTERLINK.md) - 완전한 API 스펙
- [테스트 매뉴얼](docs/CENTERLINK_API_TEST_MANUAL.md) - 단계별 테스트 절차
- [시스템 아키텍처](docs/dual_model_architecture.md) - 듀얼-티어 AI 구조

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────┐
│         FastAPI Application             │
├─────────────────────────────────────────┤
│  실시간 API  │  배치 API  │  개발 API   │
│   (SLM)     │   (LLM)   │  (No Auth)   │
├─────────────────────────────────────────┤
│          AI Analysis Engine             │
│  ┌──────────┐        ┌──────────┐      │
│  │   SLM    │        │   LLM    │      │
│  │ Qwen3    │        │ Qwen3    │      │
│  │  1.7B    │        │   4B     │      │
│  └──────────┘        └──────────┘      │
├─────────────────────────────────────────┤
│      STT Data Processor (6 formats)    │
└─────────────────────────────────────────┘
```

---

## 🔧 기술 스택

**Backend**
- FastAPI, Uvicorn, Pydantic

**AI/ML**
- PyTorch, Transformers (HuggingFace)
- Qwen3-1.7B (SLM), Qwen3-4B (LLM)

**Infrastructure**
- CUDA 12.x, ngrok (외부 노출)

---

## 📊 성능 지표

검증된 결과 (999개 실제 통화 테스트):

| 티어 | 모델 | 성공률 | 평균 시간 | 품질 점수 | 용도 |
|-----|------|--------|---------|----------|------|
| SLM | Qwen3-1.7B | 100% | 2.83초 | 0.800 | 실시간 지원 |
| LLM | Qwen3-4B | 100% | 20.85초 | 0.990 | 배치 분석 |

- ✅ 요약 성공률: **100%** (999/999)
- ✅ 제목 생성 성공률: **91.36%** (856/937)
- ✅ 평균 품질 점수: **0.990/1.00**

---

## 🛠️ 개발

### 프로젝트 구조

```
consultai/
├── main.py                     # FastAPI 엔트리포인트
├── src/
│   ├── core/                   # 핵심 비즈니스 로직
│   │   ├── ai_analyzer.py     # AI 분석 오케스트레이터
│   │   ├── models/            # 듀얼-티어 모델
│   │   │   ├── qwen3_1_7b/   # SLM (실시간)
│   │   │   └── qwen3_4b/     # LLM (배치)
│   │   └── file_processor.py  # STT 데이터 처리
│   ├── api/                   # API 라우트
│   │   └── routes/
│   │       ├── realtime.py   # 실시간 API
│   │       ├── batch.py      # 배치 API
│   │       └── dev.py        # 개발 전용
│   └── schemas/               # 데이터 스키마
├── scripts/                   # 유틸리티 스크립트
├── docs/                      # 문서
├── models/                    # AI 모델 (Git LFS 또는 제외)
└── tests/                     # 테스트
```

### 테스트 실행

```bash
# SLM 테스트 (실시간)
cd scripts
python local_test_selective_ai.py --model-tier slm --only-summary -c 10

# LLM 테스트 (배치)
python local_test_selective_ai.py --model-tier llm -c 10

# API 테스트
cd ..
python test_external_api.py
```

### 코드 품질

```bash
# 포맷팅
black src/ scripts/

# Import 정리
isort src/ scripts/

# 린팅
flake8 src/ scripts/
```

---

## 🔒 보안

### 인증 시스템
- **바운드 키 인증**: X-Bound-Key 헤더 또는 Authorization Bearer
- **권한 관리**: realtime, batch 권한 분리
- **개발 모드**: DEBUG=true일 때만 인증 없는 API 활성화

---

### 지원되는 STT 형식

ConsultAI는 다양한 STT 시스템과 호환됩니다:
1. `conversation_text` 형식 (우선)
2. `raw_call_data.details` 형식
3. 기타 4가지 커스텀 형식
