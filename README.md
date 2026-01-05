# ConsultAI

> **AI-Powered Consultation Analysis Platform for Call Centers**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

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

### 🔌 연동 API
- **실시간 API**: 바운드 키 인증, 1-3초 응답
- **배치 API**: 비동기 처리, 콜백 시스템
- **개발 전용 API**: 인증 없는 빠른 테스트 (DEBUG 모드)

### ✅ 검증된 성능
- **999개** 실제 통화 데이터 테스트 완료
- **100%** 성공률, 평균 품질 **0.990/1.00**
- **6가지** STT JSON 형식 지원

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

`.env` 파일 수정:
```bash
DEBUG=true
HOST=0.0.0.0
PORT=8000
BOUND_KEYS=your_secure_key_min_20_chars
EXTERNAL_SYSTEM_KEY=your_key
```
---

### API 테스트

**브라우저**:
- API 문서: http://localhost:8000/docs
- 대시보드: http://localhost:8000/static/consultation_dashboard.html

---

기본적으로 Qwen3 계열의 모델을 사용 **더 성능이 우수한 AI 모델이 출시될 경우 손쉽게 교체할 수 있는 유연한 아키텍처**로 설계.
- `src/core/models/` 디렉토리에 새 모델 어댑터를 추가하여 적용 가능
- 설정 파일(`config.py`) 변경으로 모델 교체 가능

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
│  └──────────┘        └──────────┘      │
├─────────────────────────────────────────┤
│      STT Data Processor (6 formats)    │
└─────────────────────────────────────────┘
```


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

---

### 지원되는 STT 형식

ConsultAI는 다양한 STT 시스템과 호환됩니다:
1. `conversation_text` 형식 (우선)
2. `raw_call_data.details` 형식
3. 기타 4가지 커스텀 형식
