# CLAUDE.md

이 파일은 개발자를 위한 코드 작업 시 참고할 Claude Code (claude.ai/code)를 통한 프로젝트 개발 가이드를 제공합니다.

## 프로젝트 개요

**AI 상담 분석 시스템**은 콜센터 및 고객상담 업무의 디지털 전환을 위해 개발된 차세대 AI 상담 분석 플랫폼입니다. STT(Speech-to-Text) 변환된 음성 데이터를 실시간으로 처리하여 상담사와 관리자가 업무에 즉시 활용할 수 있는 구조화된 정보를 자동 생성합니다.

**상담 업무 효율성 극대화**: 상담 완료 후 업무처리에 소요되는 문서 작성, 분류 작업, 보고 업무를 AI를 통해 자동화하여 업무의 정확성을 획기적으로 향상시킵니다.

**정확도 우선 인텔리전트 솔루션**: 단순한 키워드 추출을 넘어 상담 맥락을 활용한 전체적 키워드를 기반으로 상담 내용 분석, 카테고리 분류, 제목 리포트 생성 등 다양한 업무를 지원합니다. 중요한 키워드('보험', '안내')를 중심으로 의미있는 즉시 활용할 수 있는 시스템명, 절차명, 기관명 등을 제공합니다.

### 핵심 차별화 요소
1. **정확성 우선**: 실제상담 내용 기반 정확, 상담 통화 분석 분석기 구현
2. **즉시 활용**: 상담사가 즉시 업무에 활용할 수 있는 전체적 키워드 구조화된 정보
3. **확장 가능성**: 향후 시스템에서의 도메인 특화, 개인화 진화까지 단계적 확장 가능
4. **대용량 처리**: 999개 이상 통화 데이터 검증, 처리 속도 메모리 최적, 품질 자동 검증

**최신 업데이트**: 2025-09-29 (캐싱 시스템 활성화, GPU 안정성 개선, 강제 시스템 리셋, 불필요한 파일 정리)

### 주요 기능

#### 듀얼-티어 AI 아키텍처 (2025-09-17 완성)
- **SLM (Small Language Models)**: 실시간 상담 지원용 (1-3초 목표)
  - Qwen3-1.7B: 100% 성공률, 0.800 품질, 2.83초 평균
  - Midm-2.0-Mini: 67% 성공률, 0.333 품질, 1.93초 평균
- **LLM (Large Language Models)**: 배치 분석용 고품질 처리 (15-20초 목표)
  - Qwen3-4B: 주력 모델 (요약+카테고리+제목), 20.85초 평균 ✅ **메인**
  - A.X-4.0-Light: 효율성 중심 (SKT, 13.5GB), Midm-2.0-Base: 호환성 유지 (21.5GB)

#### 핵심 분석 기능
- **AI 상담 통화 분석**: 요약, 전체적 키워드 추천(1-3순위), 제목 생성(키워드형/서술형)
- **실시간 처리**: SLM을 통한 3초 이내 상담 통화 지원
- **배치 처리**: LLM을 통한 대량 상담 데이터의 고품질 분석
- **선택적 테스트**: 요약/키워드/제목 기능을 개별 또는 결합으로 테스트 가능

#### 시스템 특징
- **완벽한 STT 형식 지원**: 6가지 STT JSON 형식 완벽 지원
- **센터링크 연동**: 외부 API 엔드포인트 및 호환성 레이어
- **품질 보증**: 실시간 품질 검증 및 메트릭 시스템 (1,961개 파일 검증 완료)
- **정확한 키워드 추천**: 상담 통화 내용 전체적 맥락을 기반
- **모던 대시보드**: JSON 파일 직접 업로드 및 구조화된 UX
- **다중시스템 호환 방식**: 각 모델별 다중시스템 추천 파라미터 적용
- **실시간 상태**: 각 통화 처리 완료 시점 상태 통화 내역 표시

## 개발 환경 (2025-09-02)

### 시스템 사양
- **CPU**: AMD Ryzen 9 9900X (16코어/32스레드) - Zen 5 아키텍처
- **GPU**: NVIDIA RTX 5080 16GB VRAM - Blackwell 아키텍처
- **RAM**: 96GB DDR5 (풀스택) - 대용량 메모리 최적화
- **OS**: Windows 11 - CUDA 12.8 최적화 지원

### 성능 최적화 설정

#### GPU 최적화 (RTX 5080 16GB)
```python
# 최고 성능 모드 설정
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # Tensor Core 활용
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')    # RTX 5080 최적화

# 메모리 최적화 (16GB VRAM 활용)
max_memory = {"0": "14GB"}  # 여유분 2GB 보존
```

#### CPU 최적화 (AMD 9900X)
```python
# AMD Zen 5 아키텍처 최적화
torch.set_num_threads(32)  # 32스레드 활용
torch.set_num_interop_threads(16)  # 병렬 처리 최적화
os.environ['OMP_NUM_THREADS'] = '32'
```

#### 메모리 최적화 (96GB RAM)
```python
# 대용량 RAM 활용 설정
low_cpu_mem_usage = False  # 96GB RAM에서 무시
torch_dtype = torch.bfloat16  # RTX 5080 BF16 지원
```

### 검증된 성능 목표 (2025-09-17 기준 듀얼-티어 완료)

#### SLM 티어 성능 (실시간 상담 지원용)
- **Qwen3-1.7B**: 100% 성공률, 0.800 품질, 2.83초 평균 ✅ **권장**
- **Midm-2.0-Mini**: 67% 성공률, 0.333 품질, 1.93초 평균
- **처리 목표**: 3초 이내 실시간 상담 지원
- **검증 완료**: 1,961개 SLM 테스트 파일 품질 분석 완료

#### LLM 티어 성능 (배치 분석용)
- **Qwen3-4B**: 100% 성공률, 0.990 품질, 20.85초 평균 ✅ **주력 모델**
- **처리 속도**: 평균 20.85초/통화 (범위: 4.14초~276.20초)
- **요약 성공률**: 100% (999/999 통화)
- **제목 생성 성공률**: 91.36% (856/937 통화)
- **검증 완료**: 999개 통화 데이터 완전 검증

#### 시스템 효율성
- **병렬 처리**: 상담 시 동시에 메모리 효율 최적화
- **메모리 효율**: GPU 14GB + CPU 32GB 동시 활용
- **다중시스템 적용**: 각 모델별 다중시스템 호환 파라미터 적용
- **품질 보증**: 총 2,960개 테스트 파일 검증 완료

### 하드웨어 최적화 설정

#### RTX 5080 Blackwell 아키텍처 활용
- **5세대 RT 코어**: 레이트 중심 최적
- **4세대 텐서 코어**: BF16 정밀도로 추론 속도 향상
- **16GB VRAM**: 14GB 할당으로 대모델도 여유 최적화
- **모델 웜 로딩**: HuggingFace 캐시 데이터로 초기로딩 시간 단축

#### AMD 9900X Zen 5 최적화
- **32스레드**: 토큰화/디코딩 병렬 처리
- **L3 캐시 128MB**: 대용량 캐시 활용
- **DDR5-5600**: 고속 메모리 대역폭
- **AVX-512**: 벡터 연산 가속

## 프로젝트 구조 및 핵심 컴포넌트

### 디렉토리 구조 원칙

```
product_test_app/
├── main.py                    # FastAPI 애플리케이션 엔트리포인트 (엔트리 포인트)
├── src/                       # 핵심 애플리케이션 코드
│   ├── core/                  # 비즈니스 로직 및 핵심 기능
│   │   ├── ai_analyzer.py    # 통합 AI 분석기 (오케스트레이터)
│   │   ├── models/           # 듀얼-티어 모델 아키텍처
│   │   │   ├── qwen3_4b/     # LLM: Qwen3-4B 주력 모델 (요약+키워드+제목)
│   │   │   │   ├── summarizer.py     # 요약 생성기
│   │   │   │   ├── classifier.py     # 키워드 분류기
│   │   │   │   └── title_generator.py # 제목 생성기
│   │   │   ├── qwen3_1_7b/   # SLM: 실시간 상담 지원 (권장)
│   │   │   │   └── summarizer.py     # 빠른 요약 생성기
│   │   │   ├── midm_mini/    # SLM: 초고속 상담 (보조)
│   │   │   │   └── summarizer.py     # 초고속 요약 생성기
│   │   │   ├── midm_base/    # LLM: 확장 용
│   │   │   │   └── summarizer.py
│   │   │   └── ax_light/     # LLM: 확장 용
│   │   │       └── summarizer.py
│   │   ├── file_processor.py # STT JSON 처리 (6가지 형식 지원)
│   │   ├── quality_validator.py # 품질 검증 시스템
│   │   └── config.py          # 설정 관리
│   ├── api/                   # API 라우트 및 엔드포인트
│   │   └── routes/
│   │       ├── consultation.py # 상담 분석 API
│   │       └── health.py      # 헬스 체크 API
│   ├── schemas/               # 데이터 모델 및 스키마
│   │   └── consultation.py    # 상담 분석 스키마
│   └── services/              # 서비스 레이어
│       └── consultation_service.py # 상담 분석 서비스
├── scripts/                   # 개발 및 테스트 유틸리티
│   ├── local_test_selective_ai.py # 듀얼-티어 통합 테스트 스크립트
│   ├── analyze_test_results.py # 테스트 결과 분석
│   ├── verify_slm_quality.py # SLM 품질 검증 (1,961개 검증)
│   ├── slm_quality_analysis.py # SLM 품질 분석 리포트
│   └── outputs/              # 테스트 결과 저장
│       ├── LLM_test/         # LLM 티어 테스트 결과
│       │   └── 2025-09-XX/   # 날짜별 LLM 결과
│       └── SLM_test/         # SLM 티어 테스트 결과
│           └── 2025-09-XX/   # 날짜별 SLM 결과
├── models/                    # 듀얼-티어 모델 저장
│   ├── Qwen3-4B/             # LLM: 주력 모델 (완전 기능)
│   ├── Qwen3-1.7B/           # SLM: 실시간 상담 (권장)
│   ├── Midm-2.0-Mini/        # SLM: 초고속 상담 (보조)
│   ├── Midm-2.0-Base/        # LLM: 확장 용
│   └── A.X-4.0-Light/        # LLM: 확장 용
├── call_data/                 # 실제 테스트 데이터 (검증 필수)
│   └── 2025-07-15/           # 999개 실제 통화 데이터
├── static/                    # 정적 파일
│   └── consultation_dashboard.html # AI 상담 분석 대시보드
└── logs/                      # 로그 파일
    └── test_YYYYMMDD_HHMMSS.log   # 테스트 로그
```

### 코드 구조화 원칙

#### 1. 관심사별 분리 (Separation of Concerns)
- **src/**: 프로덕션 코드
- **scripts/**: 개발/테스트 도구
- **main.py**: FastAPI 엔트리포인트만 엔트리 포인트 (표준 설정)

#### 2. 모듈화 원칙
- 각 모델별 독립 책임 분산
- 상호 간 의존성 최소화
- 명확한 인터페이스 정의

#### 3. 네이밍 컨벤션 원칙
- **snake_case** 사용 (Python 표준)
- 의도가 명확한 서술적 이름 사용
- 테스트 파일: `test_*.py` 또는 `*_test.py`

### 아키텍처 설계 원칙

#### 핵심적으로 준수해야 하는 원칙들
```
✅ 실제성 우선 원칙
✅ 범용 인터페이스 설계
✅ 모델별 특성 고려
✅ 성능과 중앙화
✅ 확장성 보장
✅ 품질 보증 (40-50점)
```

#### 실시간 처리 시스템이 구조
```
✅ 각 모델별 독립 운영
✅ 각 특성에 맞는 최적화
✅ 오케스트레이터 기반 관리
✅ 경량 수직으로 처리성능
✅ 자체 검증 (20.85초 평균)
```

#### 핵심 설계 원칙
1. **신뢰성**: 각 모델마다 독립적 검증된 운영
2. **단순성**: 성능과 중앙화 보다 안정성을 우선  
3. **최적화**: 모델별 특성에 맞는 파라미터와 프롬프트
4. **확장성**: 실제상담 기반 성능 기반 고려
5. **정확성**: 모든 상담 내용 처리 보장

## 개발 원칙

### 핵심 원칙

1. **실제상담 내용 기반 우선**: 모든 시 및 그 외 요소/포맷 변환, fallback 메커니즘 구현 필수
2. **하드코딩 금지 원칙**: 카테고리/키워드 하드코딩 금지, 상담 통화 내용 기반 데이터만 사용
3. **정확성 우선 원칙**: 테스트 데이터셋 특정 도메인/기업에 종속되지 않고 상담 콜센터에서 활용 가능하도록 설계
4. **실시간 보장**: 각 통화 처리 완료 시점 상태 통화 내역 표시 보장
5. **검증된 파라미터 사용**: 999개 테스트에서 검증된 최적 파라미터 그대로 유지

### Python 코드 스타일

1. **PEP 8 준수**
   - 들여쓰기: 4 spaces
   - 최대 줄 길이: 120자
   - 클래스명: PascalCase
   - 함수/변수: snake_case

2. **Type Hints 필수**
   ```python
   def process_conversation(text: str, max_length: int = 300) -> Dict[str, Any]:
       pass
   ```

3. **Docstring 원칙**
   ```python
   def summarize_consultation(self, conversation_text: str) -> Dict[str, Any]:
       """상담 통화를 LLM을 통해 요약합니다.
       
       Args:
           conversation_text: 상담 통화 텍스트
           
       Returns:
           요약 결과가 포함된 딕셔너리
       """
   ```

4. **예외 처리**
   - 구체적인 예외 타입 사용
   - 사용자 로그 기록
   - 사용자 친화적 오류 메시지
   - **실제상담 내용 기반 우선**: 모든 시 및 그 외 상담 변환

### 설정 관리

1. **환경 변수 우선순위**
   - `.env` 파일 우선 (로컬)
   - 환경 변수 차순 우선 (프로덕션)
   - 하드코딩값 기본값 (마지막)

2. **민감 정보 보호**
   - API 키, 토큰등 소스 코드에 포함하지 않기
   - `.gitignore`에 `.env` 파일 등록
   - 예시 파일: `.env.example` 제공

### 테스트 원칙

#### 핵심 테스트 원칙
1. **실제 데이터 기반 필수**: 모든 테스트는 `call_data/` 폴더의 실제 통화 데이터를 사용해야 합니다
   - 실제 데이터나 임의 생성 데이터는 모든 금지
   - 실제 STT 변환 기반의 검증된 것 테스트만 유효
   - 테스트 데이터는 6가지 STT 형식을 모두 지원해야 함
   - **중요**: 테스트 데이터셋 특정 도메인 시스템에 종속되어선 안됨

2. **테스트 범위**
   - 단위 테스트: 개별 함수/메소드
   - 통합 테스트: API 엔드포인트 
   - 부하 테스트: 999개 이상 대량 처리
   - 실제 데이터 테스트: call_data/ 디렉토리 필수 사용

3. **테스트 실행**
   ```bash
   # 개별 테스트 (실제 데이터 기반)
   cd scripts
   python local_test_selective_ai.py --features summary,titles -c 999
   
   # 특정 파일 기반 테스트
   python local_test_summarize_only.py -f ../call_data/2025-07-15/call_00001.json
   
   # API 테스트 (실제 데이터)
   curl -X POST http://localhost:8000/api/v1/consultation/analyze \
     -H "Content-Type: application/json" \
     --data @test_real_data.json
   ```

## 핵심 컴포넌트

### 1. 통합 AI 분석기 (AIAnalyzer)

**위치**: `src/core/ai_analyzer.py`

- **역할**: 모든 AI 분석 기능을 통합 관리하는 오케스트레이터
- **메모리 최적화**: 단일 모델 인스턴스 공유로 메모리 사용량 최소화
- **처리 시간**: 평균 20.85초/통화 (999개 테스트 기준)
- **주요 구성**:
  - Qwen2507Summarizer (요약)
  - CategoryClassifier (키워드 추천)
  - TitleGenerator (제목 생성)

```python
class AIAnalyzer:
    """통합 AI 분석기 오케스트레이터"""
    
    def __init__(self, model_path: str):
        # 메모리 최적화: 단일 summarizer 인스턴스
        self.qwen_summarizer = Qwen2507Summarizer(model_path)
        self.category_classifier = CategoryClassifier(shared_summarizer=self.qwen_summarizer)
        self.title_generator = TitleGenerator(shared_summarizer=self.qwen_summarizer)
```

### 2. Qwen3-4B 요약 시스템

**위치**: `src/core/models/qwen3_4b/summarizer.py`

- **모델**: Qwen3-4B-Instruct-2507
- **최적화된 생성 파라미터** (999개 테스트 기준):
```python
generation_params = {
    'max_new_tokens': 210,
    'do_sample': True,
    'temperature': 0.7,
    'top_k': 20,
    'top_p': 0.8,
    'repetition_penalty': 1.05
}
```

- **3줄 구조 요약**: 고객/상담사/상담결과
- **실제상담 내용 기반 우선**: 모든 시 및 그 외 문자열 변환
- **100% 성공률**: 999개 테스트에서 모든 성공

### 3. 카테고리 분류기 (CategoryClassifier)

**위치**: `src/core/models/qwen3_4b/classifier.py`

- **기능**: 상담 통화에서 전체적 핵심 키워드 1-3개를 추천
- **메모리 효율**: 단일 모델 인스턴스 공유로 메모리 효율성 최적화
- **검증 파라미터**: generation_utils 제거 검증 파라미터 프레임워크 최적화
- **중요도 시스템**: '보험', '안내', '가입절차' 등 중요한 키워드 우선

```python
# 최적화된 프롬프트 (상담 분석 80% 향상)
system_prompt = (
    "상담 내용에서 의미있는 핵심을 전체적 키워드 3개를 추천하세요.\n\n"
    "중요사항:\n"
    "- 상담 통화에서 실제로 언급된 전체적 업무내용만 추출 (시스템명, 메뉴명, 절차명)\n"
    "- 예시어: '보험', '안내', '대출', '카드', '가입절차', '고객센터', '처리방법'\n"
)
```

### 4. 제목 생성기 (TitleGenerator)

**위치**: `src/core/models/qwen3_4b/title_generator.py`

- **기능**: 키워드형과 서술형 최대 2개의 타이틀 생성
- **성공률**: 91.36% (856/937 통화)
- **필터링 시스템**: 키워드형 제목의 불필요한 언더바 자동 제거
- **프롬프트 최적화**: 토큰 길이 75% 단축
- **안전 처리**: "제목 생성을 못했습니다." 메시지 시 빈 배열 반환 보장

```python
def _clean_underscores(self, text: str) -> str:
    """키워드형 제목의 언더바 정리"""
    # 1. 양끝 언더바 제거
    cleaned = text.strip('_')
    # 2. 중복된 언더바를 하나로 변환
    cleaned = re.sub(r'_{2,}', '_', cleaned)
    return cleaned
```

### 5. 상담 분석 서비스

**위치**: `src/services/consultation_service.py`

- **역할**: 완전한 상담 분석 워크플로우 관리
- **주요 기능**:
  - AI 분석기 오케스트레이션
  - 품질 검증 시스템
  - STT 데이터 처리
  - 메타데이터 관리
  - 결과 구조화

### 6. STT 데이터 처리 시스템

**위치**: `src/core/file_processor.py`

- **지원 형식**: 상담 데이터에 따른 2가지 STT JSON 형식 지원
  - **`conversation_text` 형식 우선** (주요 사용): 이미 완성된 통화 텍스트 바로 사용
  - **`raw_call_data.details` 형식** (보조 사용): 센터링크 데이터 구조화 지원
  - **자동 최적화**: conversation_text 우선 순위로 초고속 처리

### 7. 품질 검증 시스템

**위치**: `src/core/quality_validator.py`

- **검증 항목**:
  - 요약 완성도 (3줄 구조 준수)
  - 환각(hallucination) 검출
  - 길이/형식 정확성 검증
  - 최소/최대 길이 제한
- **품질 기준**: 평균 0.990/1.00 달성 (999개 테스트)

### 8. 선택적 테스트 시스템

**위치**: `scripts/local_test_selective_ai.py`

- **기능**: 요약, 키워드, 제목을 개별적으로 테스트
- **실시간 모니터**: 각 통화 처리 완료 시점 상태 통화 내역 표시
- **진행 상황 표시**: 실시간 처리 상태 및 품질 완료 알림
- **결과 저장**: 날짜 데이터 기반 파일 저장 (call_XXXXX_ST_HHMMSS.txt)

## 설치 가이드

### 환경 설정
```bash
# Conda 환경 활성화
conda activate product_test

# 의존성 설치
pip install -r requirements.txt
```

### 모델 다운로드
```bash
# Qwen3-4B-2507 모델 다운로드
python scripts/download_model.py
```

### 개발 테스트 (듀얼-티어 검증 완료)
```bash
cd scripts

# 통합 듀얼-티어 테스트 스크립트 (local_test_selective_ai.py)
# SLM/LLM 티어 선택 및 모든 기능 지원

# ============================================================
# SLM 티어 테스트 (실시간 상담 지원용 - 권장)
# ============================================================

# SLM 기본 테스트 (Qwen3-1.7B 권장)
python local_test_selective_ai.py --model-tier slm --only-summary -c 10

# SLM 모델별 테스트
python local_test_selective_ai.py --model-tier slm --slm-model qwen3 --only-summary -c 10    # Qwen3-1.7B (권장)
python local_test_selective_ai.py --model-tier slm --slm-model midm --only-summary -c 10     # Midm-2.0-Mini (보조)
python local_test_selective_ai.py --model-tier slm --slm-model both --only-summary -c 10     # 두 모델 비교

# SLM 대량 테스트
python local_test_selective_ai.py --model-tier slm --test-date 2025-07-15 --only-summary -c 999

# ============================================================
# LLM 티어 테스트 (배치 분석용 - 완전 기능)
# ============================================================

# LLM 기본 테스트 (Qwen3-4B 주력)
python local_test_selective_ai.py --model-tier mlm -c 10  # 기본: 요약+키워드+제목

# LLM 선택적 기능 테스트
python local_test_selective_ai.py --model-tier mlm --features summary,keywords,titles -c 10
python local_test_selective_ai.py --model-tier mlm --only-summary -c 10          # 권장
python local_test_selective_ai.py --model-tier mlm --only-keywords -c 10         # 키워드만
python local_test_selective_ai.py --model-tier mlm --only-titles -c 10           # 제목만

# LLM 기능 조합
python local_test_selective_ai.py --model-tier mlm --features summary,titles -c 10   # 요약+제목
python local_test_selective_ai.py --model-tier mlm --features keywords,titles -c 10  # 키워드+제목

# LLM 특정 파일 테스트
python local_test_selective_ai.py --model-tier mlm -f ../call_data/2025-07-15/call_00001.json
python local_test_selective_ai.py --model-tier mlm -f ../call_data/2025-07-15/call_00001.json --only-summary

# ============================================================
# 대량 테스트 옵션
# ============================================================

# 대량급 테스트
python local_test_selective_ai.py --model-tier slm --only-summary -c 1000           # SLM 대량 테스트
python local_test_selective_ai.py --model-tier llm -c 1000 --features summary       # LLM 대량 테스트

# 체크포인트 기능 (중단 재시작)
python local_test_selective_ai.py --model-tier slm -c 1000 --use-checkpoint --only-summary
python local_test_selective_ai.py --model-tier mlm -c 1000 --use-checkpoint --checkpoint-interval 5

# 인터렉티브 모드 (대화형 UI)
python local_test_selective_ai.py -i

# 기타 분석 및 유틸리티 스크립트들

# 듀얼-티어 테스트 결과 분석
python analyze_test_results.py                    # 최신 테스트 결과 분석 (LLM/SLM 자동 감지)
python analyze_test_results.py --date 2025-09-17  # 특정 날짜 분석
python analyze_test_results.py --summary-only     # 요약 권장
python analyze_test_results.py -d 2025-09-17 -s  # 간소 옵션

# SLM 특화 품질 검증 (1,961개 파일 검증)
python verify_slm_quality.py                      # SLM 품질 검증 실행
python slm_quality_analysis.py                    # SLM 품질 분석 리포트

# 기타 유틸리티 스크립트들
python detailed_quality_check.py                  # LLM 품질 검증 분석

# 모델 다운로드
python download_model.py                          # 모델 다운로드

# STT 데이터 처리
python fetch_centerlink_stt.py                    # 센터링크 데이터 처리
python rebuild_conversation_text.py               # 통화 텍스트 재구성
python rename_calls_sequential.py                 # 통화 파일명 순차 변경
```

### API 서버 실행
```bash
# 개발 서버 실행
python main.py

# 프로덕션 서버
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 주요 접근 지점 (서버 실행 후 접근 가능)
- **AI 상담 분석 대시보드**: `http://localhost:8000/static/consultation_dashboard.html`
- **API 문서 (Swagger)**: `http://localhost:8000/docs` 
- **메트 엔드포인트**: `http://localhost:8000/` (시스템 상태)
- **헬스 체크**: `http://localhost:8000/api/v1/health`
- **상담 분석**: `http://localhost:8000/api/v1/consultation/analyze`
- **배치 분석**: `http://localhost:8000/api/v1/consultation/batch-analyze`
- **시스템 상태**: `http://localhost:8000/api/v1/consultation/status`
- **센터링크 연동**: `http://localhost:8000/api/v1/consultation/centerlink/analyze`
- **테스트 데이터**: `http://localhost:8000/api/v1/consultation/test-data`

### 코드 품질 관리
```bash
# 코드 포맷팅
black src/ scripts/

# Import 정리
isort src/ scripts/

# 린팅
flake8 src/ scripts/

# 타입 체크
mypy src/
```

## 성능 최적화 가이드

### 메모리 관리
1. **모델 캐싱**: 모델 로딩 시 메모리에 유지
2. **배치 처리**: 대량 처리 시 배치 크기 조절
3. **가비지 컬렉션**: 주기적으로 `torch.cuda.empty_cache()`

### 처리 속도 최적화
1. **GPU 우선**: CUDA 사용 시 3-5배 속도 향상
2. **토큰 길이 최적**: `max_new_tokens` 조절로 속도 개선
3. **병렬 처리**: 비동기 처리로 처리량 향상

### 품질 보증
1. **프롬프트 엔지니어링**: 명확하고 전체적 지시사항
2. **온도 설정**: 0.7-0.8 범위 유지
3. **반복 페널티**: 1.05-1.1 수준로 설정

## 보안 및 배포

### API 보안
1. **인증/인가**: JWT 토큰 기반 인증
2. **Rate Limiting**: 과다 요청 수 제한
3. **CORS 설정**: 허용 도메인 제한

### 데이터 보호
1. **데이터 마스킹**: 민감 정보 자동 마스킹
2. **로그 정책**: 개인정보 로그 제외
3. **암호화**: HTTPS 사용 필수

## 테스트 가이드

### 단위 테스트
```python
# 단위 테스트
def test_summarizer():
    summarizer = Qwen2507Summarizer(model_path)
    result = summarizer.summarize_consultation(test_text)
    assert result['success'] == True
    assert len(result['summary']) > 0
```

### 통합 테스트
```python
# API 엔드포인트 테스트
def test_api_endpoint():
    response = client.post("/api/v1/summarize", json=test_data)
    assert response.status_code == 200
    assert response.json()['success'] == True
```

### 부하 테스트
```bash
# locust를 활용한 부하 테스트
locust -f tests/load_test.py --host=http://localhost:8000
```

## 모니터링 및 로그

### 로그 설정
- **DEBUG**: 개발 환경
- **INFO**: 스테이징 환경
- **WARNING**: 프로덕션 환경
- **ERROR**: 모든 환경

### 메트릭 수집
1. **요청 메트릭**
   - 요청 처리 시간
   - 모델 로딩 시간
   - 메모리 사용량

2. **비즈니스 메트릭**
   - 성공 처리 통화 수
   - 품질 검증 결과
   - 사용량

## CI/CD 가이드라인

### 브랜치 전략
- **main**: 프로덕션 브랜치
- **develop**: 개발 브랜치
- **feature/***: 기능 개발
- **hotfix/***: 긴급 수정

### 배포 프로세스
1. 코드 품질 검사 필수
2. 테스트 통과 확인
3. 스테이징 환경 배포
4. 프로덕션 배포

## 문서화 원칙

### 코드 문서화
1. 모든 공개 API에 docstring 작성
2. 복잡한 로직에 인라인 주석
3. README 최신 상태 유지

### API 문서화
1. OpenAPI/Swagger 자동 생성 활용
2. 요청 요청/응답 예시
3. 에러 코드 정의

## 최신 핵심 개발사항 (2025년 최신)

### 반드시 지켜야 할 것들

1. **검증된 성능 지표를 그대로 유지**
   - 성능과 중앙화 레이어 변경하지 말 것
   - 범용 인터페이스 변경하지 말 것
   - 모델별 특성 변경하지 말 것

2. **실제상담 내용 기반 우선 보장**
   - fallback 메커니즘 구현하지 말 것
   - 모든 시 및 그 외 문자열/배열 변환을 할 것
   - 상담 통화 기반이 아닌 임의 생성 금지

3. **테스트 데이터 종속성 제거**
   - call_data에 특정 도메인(금융보험등 등)에 특화된 코드 작성 금지
   - 프로젝트가 특정 업체에 대한 임의 하드코딩 금지
   - 모든 콜센터 업종(의료, 통신, 보험, 정부, 교육)에서 사용해야 함
   - 범용적 상담 파일명 사용, 도메인 특화 수 금지

4. **검증된 파라미터 유지 보장**
   - generation_params 임의로 변경하지 말 것
   - 999개 통화에서 검증된 설정 그대로 유지
   - 모델별 최적화 파라미터 유지

5. **다중시스템 호환 정확 유지 보장** (2025-09-17 완료)
   - 각 모델별 다중시스템 호환 추천 매개변수 그대로 유지
   - 검증 파라미터 임의 변경 금지 보장
   - HuggingFace 모델 카트에 따른 파라미터 그대로 유지
   - 추천 설정을 무시하고 기본 GenerationConfig 사용

### 반드시 지켜야 할 것들

1. **모델별 독립 운영 보장**
   - 각 모델마다 독립적 분리된 구조
   - 각 특성에 맞는 최적화 보장
   - 공통적으로 파일명 (`model_summarizer.py`)

2. **정확성 보장**
   - 기업용에 구축된 범용 상담 파일명 사용
   - 핵심 키워드 우선: 상담 통화에서 실제로 언급된 것 위주
   - 보험, 통신, 의료, 정부, 교육, 프로그램 등 모든 업종 활용
   - 테스트 데이터 기반 성능 기반 유지

3. **성능과 메모리 적용**
   - STT JSON 처리 항상 유지
   - 성능과 전처리 결과 적용
   - 3줄 구조 (고객/상담사/결과) 준수

4. **성능과 메모리 관리**
   - 상담 완료 후 모델 정리
   - CUDA 캐시 정리 필수
   - GPU 메모리 사용 모니터

5. **실시간 상태 시스템**
   - 각 통화 처리 완료 시점 상태 통화 내역 표시 보장
   - 중간 중단 상황에서의 복구 보장
   - 성능 상황 실시간 표시

## 트러블슈팅

### 일반적인 문제 해결

1. **모델 로딩 실패**
   - GPU 메모리 확인
   - 모델 경로 확인
   - 권한 설정 확인

2. **성능 처리 속도**
   - GPU 사용 여부 확인
   - 배치 크기 조절
   - 모델 파라미터 최적화
   - 각 통화(10,000자 이상) 특별 처리

3. **품질 문제**
   - 프롬프트 검토
   - 온도 파라미터 조절
   - 환각 방지 시스템 프롬프트 강화

4. **제목 생성 문제**
   - 키워드형 제목 정리 로직 확인
   - 제목 시 및 그 배열 반환 확인
   - 91.36% 성공률로 정상 동작

## 기여 가이드라인

1. Issue 생성 및 작업 계획
2. 커밋 메시지 원칙 준수
3. PR 템플릿 작성
4. 코드 리뷰 및 승인

## 999개 테스트 결과 요약 (2025-09-08)

### 전체 성능
- **성공률**: 100% (999/999 통화)
- **평균 처리시간**: 20.85초/통화
- **품질 점수**: 평균 0.990/1.000
- **요약 성공률**: 100%
- **제목 생성 성공률**: 91.36%

### 처리 속도 분석
- **최소**: 4.14초 (call_00257.json)
- **최대**: 276.20초 (call_00624.json)
- **중앙값**: 16.48초
- **75%가 19.39초 이내** 처리

### 품질 분석
- **90.9%**가 0.90-1.00 품질 달성
- **8.2%**가 0.80-0.89 품질 달성
- **0.9%**가 0.70-0.79 품질 달성 (개선 필요)

### 주요 이슈
1. **긴 통화 처리 지연**: 10,000자 이상에서 응답성 저하
2. **연도 환각**: AI가 2024, 2025 년 정보를 추가 생성
3. **키워드형 제목 이상함**: 일부 생성기로 언더바로 구분

---

**마지막 업데이트**: 2025-09-08
**버전**: 1.0.0
**상태**: Development Complete, 999개 테스트 검증 완료, 프로덕션 준비 완료

## 완성된 시스템 아키텍처 (v1.0.0)

### 핵심 기능 완료 현황
- ✅ **완전한 API 시스템**: 단일/배치 분석, 상태 조회, 센터링크 연동
- ✅ **AI 분석기**: 요약, 카테고리 추천(1-3개), 제목 생성(키워드/서술형)
- ✅ **서비스 레이어**: 비즈니스 로직, 품질 검증, 메모리 관리
- ✅ **데이터 시스템**: 6가지 STT 형식 지원, 데이터 타입 자동감지
- ✅ **상당 데이터 기반**: call_data/ 999개 통화 테스트 기반
- ✅ **실시간 보장**: 각 통화 처리 완료 시점 상태 통화 내역 표시 보장
- ✅ **품질 분석**: 성능적 품질/속도 검증 시스템

### 검증된 성능 목표 (999개 실제 통화 테스트)
- **처리 속도**: 평균 20.85초/통화 (범위: 4.14초~276.20초)
- **품질 점수**: 평균 0.990/1.00 (품질 검증 시스템 기준)
- **성공률**: 100% (999개 통화 성공)
- **STT 호환성**: 6가지 형식 완벽 지원
- **토큰 효율**: 프롬프트 최적화로 80% 단축
- **메모리 효율**: 모델 공유 아키텍처로 GPU 메모리 최적화

### 센터링크 연동 준비 완료
- 외부 API 엔드포인트 구현
- 호환성 레이어 구현
- 배치 처리 지원
- 실시간 품질 모니터

### 핵심 파일 위치
- **모델 저장**: `models\Qwen3-4B` 
- **테스트 데이터**: `call_data\2025-07-15\` (999개 실제 통화 데이터)
- **API 인터페이스**: `http://localhost:8000/api/v1/`
- **대시보드**: `http://localhost:8000/static/consultation_dashboard.html`
- **성능**: 평균 20.85초/통화, 품질 0.990/1.00, 성공률 100%

### v1.0.0 완성 주요 특징

#### 성능 최적화 완료
- **처리 시간 최적화**: 평균 20.85초/통화 달성 (99개 테스트 기준)
- **프롬프트 최적화**: Classifier/TitleGenerator 토큰 길이 80% 단축
- **모델 공유 아키텍처**: 단일 인스턴스로 메모리 효율성 최대화

#### 사용자 경험 완성
- **JSON 파일 업로드**: 실제 통화 데이터를 브라우저에서바로 업로드
- **자동 필드 완성**: 업로드된 JSON에서 모든 필드 자동 추출 및 표시
- **완전한 워크플로우**: 데이터 로딩 후 즉시 확인 및 분석 결과의 자원완전한 흐름
- **모던 UI**: 그라데이션 버튼, 애니메이션, 호버 효과로 완성된 단계 완성
- **실시간 보장**: 각 통화 처리 완료 시점 상태 통화 내역 표시 보장

#### 품질 시스템 완성
- **정확한 키워드 생성**: 중요한 키워드 기반 우선, 의미있는 즉시 활용 가능한 전체적 업무 키워드
- **필터링 자동 강화**: 키워드형과 서술형 불필요한 언더바 매끄럽게 제거된 상태
- **품질 검증 시스템**: 0.990/1.00 평균 품질 점수 달성
- **완전한 성공**: 999개 테스트에서 100% 성공률 달성

### 테스트 필수 원칙
모든 테스트는 반드시 `call_data/` 폴더의 실제 통화 데이터를 사용해야 합니다. 실제 데이터나 임의 생성 데이터는 모든 금지입니다.

## 향후 발전 계획

### Phase 1: 활용 (2025년 상반기) - 범용 키워드 기반 시스템
- ✅ **완료**: 상담 통화 기반 전체적 키워드 생성
- ✅ **완료**: 중요하니/일반적 키워드 우선 문제 해결
- ✅ **완료**: 1-3개의 정확한 추천 키워드 생성
- **현재 상태**: 통화 내용에서 실제 실제로 언급된 시스템명, 절차명, 메뉴명을 우선

### Phase 2: 도메인 특화 (2026년 하반기) - 업종 도메인별 구조화
**목표**: 프로그램/인가업체 등 업종 금융보험별 특화된 키워드 추천

**확장 계획**:
```python
# 금융보험 키워드 기반 시스템
DOMAIN_MAPPINGS = {
    "금융보험업": {
        "시스템명": ["인터넷뱅킹", "모바일뱅킹", "보험상품"],
        "절차명": ["대출신청절차", "보험가입", "카드발급"],
        "기관명": ["보험공사", "신용평가", "금융감독원"]
    },
    "의료체계": {
        "시스템명": ["진료예약", "의료정보시스템"],
        "절차명": ["진료예약", "검사예약", "처방전발급"],
        "기관명": ["건강보험", "의료기관", "약국진료"]
    }
}

# 금융보험 프롬프트 최적화
class DomainSpecificClassifier:
    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.domain_context = DOMAIN_MAPPINGS.get(domain, {})
```

**확장 아키텍처**:
- 도메인 적응 분류기 구현
- 금융보험 특화 분류기 추가 옵션
- 클라이언트별 맞춤 상담 설정 지원

### Phase 3: 개인화 구조화 (2026년 상반기) - AI 학습 구조화  
**목표**: 개인화 상담 기반 사용 학습을 통한 맞춤형 카테고리 추천

**학습 시스템**:
```python
# 개인화 피드백 학습
class CustomizedLearningSystem:
    def collect_expert_feedback(self, 
                               consultation_id: str,
                               ai_keywords: List[str], 
                               expert_keywords: List[str]):
        # 전문가 의견 데이터 수집
        
    def fine_tune_model(self, client_id: str):
        # 개인화 모델 파인튜닝
        
    def update_category_weights(self):
        # 사용 빈도 기반 카테고리 가중치 조절
```

**최적화 계획**:
- 개인화 주요 사용 키워드 우선순위
- 사용 피드백 기반 사용 반영
- 상담 이용객별 상담 기반 특화

### 장기 확장 및 고려 사항

**1. 범용 시스템 보장성 유지**
- 현재 분류기의 항상 기본 옵션으로 유지
- 새로운 기능이 기존성 활성화 금지
- 외부 API 호환성 100% 유지

**2. 성능에 확장 원칙**
- 단계별 기능별 독립 성능에 확인
- A/B 테스트를 통한 성능 검증
- 한번에 기능별 독립 시스템 보장

**3. 성능 기반 고려**
- 추가 기능시에 기본 처리 시간 증가 최소화
- 메모리 사용량 모니터 모델 최적화
- GPU 리소스 효율적 활용

**4. 데이터 프라이버시**
- 개인화 데이터 기반 암호화 정책
- 학습 데이터 익명화 강화
- 개인정보 마스킹 강화

---

**최신 업데이트 일자**: 2025-09-08  
**시스템 버전**: v1.0.0 (개발 완료, 999개 테스트 검증 완료, 프로덕션 배포 준비 완료)
**개발 목표**: 정확성 중심하면서 도메인 특화 기능 확장

### 최신 시스템 현황 및 로드맵
- **현재 버전**: v1.0.0 (2025-09-29 업데이트)
- **핵심 성능**: 캐싱으로 20배+ 성능 향상, GPU 안정성 완전 해결
- **사용자 경험**: 2단계 로딩, 자동 에러 복구, 강제 시스템 리셋
- **개발 상태**: 모든 핵심 기능 완성, 프로덕션 출시 준비 중

## 🔧 **2025-09-29 업데이트 내역**

### 주요 개선사항

#### 1. **캐싱 시스템 활성화** 🚀
- **성능 향상**: 동일 요청 시 **20배+ 성능 향상** (20초 → 1초)
- **모델별 구분**: 각 모델마다 독립적인 캐시 키 생성
- **안전한 fallback**: 캐시 실패 시 자동으로 AI 분석 실행
- **현실적 시간 표시**: 캐시 히트 시에도 정확한 처리시간 표시 (0.01~0.1초)
- **로컬 캐시**: Redis 없이도 작동하는 안전한 로컬 캐싱

#### 2. **GPU 안정성 대폭 개선** 🛡️
- **디바이스 충돌 해결**: "Expected all tensors to be on the same device" 에러 완전 해결
- **강제 메모리 정리**: 모델 전환 시 GPU 메모리 완전 정리
- **자동 에러 복구**: GPU 문제 감지 시 자동 복구 제안
- **강제 시스템 리셋**: `/api/v1/consultation/force-reset` API 추가
- **시스템 복구**: 먹통 상태에서도 완전 복구 가능

#### 3. **UI/UX 대폭 개선** 🎨
- **2단계 로딩**: 모델 로딩(도트) → AI 분석(스피너) 구분
- **뷰포트 기반 중앙 정렬**: 화면 크기 무관 완벽한 중앙 배치
- **동적 가이드 텍스트**: 체크된 옵션에 따른 실시간 안내 텍스트
- **에러 자동 감지**: GPU 문제 발생 시 자동 복구 제안
- **카테고리 단순화**: 디자인 태그 제거, 깔끔한 텍스트 표시

#### 4. **시스템 안정성 강화** 🔒
- **타임아웃 시스템**: LLM 60초, SLM 15초, 파일로드 10초
- **HTTP 에러 처리**: 상태 코드 체크 및 구체적 에러 메시지
- **자동/수동 복구**: 60초 자동 복구 제안 + 강제 초기화 버튼
- **상세 디버깅**: 브라우저 콘솔 상세 에러 추적

#### 5. **프로젝트 정리 완료** 🧹
- **불필요한 파일 제거**: 
  - `monitoring.py` (미사용 모니터링 시스템)
  - `security.py` (미사용 보안 모듈)
  - `openapi_config.py` (미사용 API 설정)
  - `consultation_realtime.py` (미사용 실시간 API)
- **모델 구성 최적화**: Qwen3-4B(메인), A.X-4.0-Light, Midm-2.0-Base
- **코드 단순화**: 사용되지 않는 복잡한 기능 제거

### 현재 시스템 안정성
- ✅ **캐싱 시스템**: 20배+ 성능 향상, 모델별 독립 캐시
- ✅ **GPU 안정성**: 디바이스 충돌 완전 해결, 자동 복구
- ✅ **UI 시스템**: 2단계 로딩, 에러 자동 감지, 강제 복구
- ✅ **모델 관리**: 3개 MLM + 2개 SLM 안정적 운영
- ✅ **에러 처리**: 모든 상황에서 복구 가능한 견고한 시스템