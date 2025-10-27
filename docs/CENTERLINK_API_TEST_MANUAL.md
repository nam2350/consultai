# 센터링크 API 연동 테스트 메뉴얼

**작성일**: 2025-10-01
**대상**: 센터링크 개발팀 + AI 분석팀
**목적**: 실제 API 연동 테스트 진행

---

## 📋 테스트 개요

### 테스트 목적
- 실시간 상담 분석 API (SLM) 동작 확인
- 비동기 배치 분석 API (LLM) 동작 확인
- 콜백 시스템 정상 작동 확인
- 바운드 키 인증 검증

### 테스트 환경
- **AI 분석 서버**: 개발 환경 (로컬 서버 + ngrok)
- **센터링크 서버**: 개발/테스트 환경
- **네트워크**: 인터넷 (ngrok 터널)

---

## 🔧 사전 준비사항

### AI 분석팀 준비사항

**1. 필수 소프트웨어 설치**
- Python 환경: `conda activate product_test`
- ngrok: https://ngrok.com/download

**2. 서버 실행 준비**
```bash
# 프로젝트 디렉토리로 이동
cd C:\Workspace\product_test_app

# Conda 환경 활성화
conda activate product_test

# 환경 확인
python --version  # Python 3.10 이상
```

**3. 방화벽 설정** (필요시)
```powershell
# PowerShell 관리자 권한
New-NetFirewallRule -DisplayName "AI API Server" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

### 센터링크팀 준비사항

**1. 콜백 서버 실행**

프로젝트에 포함된 `centerlink_callback_server.py`를 사용하면 즉시 테스트 가능합니다.

**실행**:
```bash
cd C:\Workspace\product_test_app
python centerlink_callback_server.py
```

**예상 출력**:
```
================================================================================
센터링크 콜백 서버 시작
================================================================================
포트: 5000
콜백 엔드포인트: http://localhost:5000/api/ai-callback
헬스 체크: http://localhost:5000/health
결과 저장 폴더: callback_results/
================================================================================

외부 노출을 위해 다른 터미널에서 실행:
  ngrok http 5000

 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.x.x:5000
```

**주요 기능**:
- ✅ AI 배치 결과 수신 및 로그 출력
- ✅ 수신 데이터를 `callback_results/` 폴더에 JSON 저장
- ✅ 통화별 처리 결과 요약 표시
- ✅ 자동으로 200 OK 응답

**커스터마이징 (선택)**:
실제 DB 저장 로직을 추가하려면 `centerlink_callback_server.py` 파일의 79번째 줄 수정:
```python
# TODO: 센터링크 DB에 저장
# save_to_database(data)
```

**2. ngrok으로 콜백 서버 외부 노출**

```bash
ngrok http 5000
```

출력 예시:
```
Forwarding   https://xyz-456-abc.ngrok-free.app -> http://localhost:5000
```

**3. API 테스트 도구 준비**
- Postman 또는 curl
- 테스트용 상담 데이터 (JSON)

---

## 🚀 연동 테스트 절차

### Phase 0: 빠른 검증 (1분) ⚡

**목적**: 자동화 스크립트로 전체 API 빠르게 검증

프로젝트에 포함된 `test_external_api.py`를 사용하면 모든 테스트를 자동으로 실행할 수 있습니다.

**Step 0-1: AI 서버 실행**

터미널 1:
```bash
cd C:\Workspace\product_test_app
conda activate product_test
python main.py
```

**Step 0-2: ngrok으로 외부 노출**

터미널 2:
```bash
ngrok http 8000
```

생성된 URL 복사 (예: `https://abc-123-def.ngrok-free.app`)

**Step 0-3: 테스트 스크립트 URL 설정**

`test_external_api.py` 파일 열기:

```python
# 11번째 줄 수정
BASE_URL = "https://abc-123-def.ngrok-free.app"  # ngrok URL로 변경
```

**Step 0-4: 자동 테스트 실행**

터미널 3:
```bash
cd C:\Workspace\product_test_app
python test_external_api.py
```

**예상 출력**:
```
================================================================================
외부 API 테스트 시작
서버 URL: https://abc-123-def.ngrok-free.app
================================================================================

================================================================================
[1] 헬스 체크 테스트
================================================================================
Status Code: 200
Response: {
  "status": "healthy",
  "timestamp": "2025-10-16T10:30:00Z"
}

================================================================================
[2] 실시간 분석 API 테스트 (SLM)
================================================================================
요청 URL: https://abc-123-def.ngrok-free.app/api/v1/consultation/realtime-analyze
요청 시작...
Status Code: 200
처리 시간: 2.35초
Response:
{
  "success": true,
  "consultation_id": "EXTERNAL_TEST_REALTIME_001",
  "summary": "...",
  ...
}

================================================================================
[3] 배치 분석 API 테스트 (LLM)
================================================================================
...

================================================================================
테스트 결과 요약
================================================================================
헬스 체크            : ✅ 성공
실시간 API          : ✅ 성공
배치 API            : ✅ 성공
인증 실패            : ✅ 성공

총 4개 테스트 중 4개 성공 (100.0%)
```

**자동 테스트 항목**:
- ✅ 헬스 체크 (서버 정상 동작)
- ✅ 실시간 분석 API (SLM, 1-3초)
- ✅ 배치 분석 API (LLM, 비동기)
- ✅ 인증 실패 테스트 (401)

**장점**:
- 1분 안에 전체 API 검증 완료
- 수동 curl 명령어 입력 불필요
- 자동으로 성공/실패 판정

**다음 단계**:
- ✅ 모든 테스트 성공 → 연동 완료!
- ❌ 일부 테스트 실패 → Phase 1-3로 상세 검증

---

### Phase 1: AI 분석 서버 시작

**Step 1-1: 메인 서버 실행**

터미널 1:
```bash
cd C:\Workspace\product_test_app
conda activate product_test
python main.py
```

**예상 출력**:
```
INFO:     [라이프사이클] 애플리케이션 시작 - 배치 워커 초기화 중...
INFO:     [배치워커] 배치 워커 초기화 완료
INFO:     [배치워커] 워커 시작
INFO:     [라이프사이클] 배치 워커 시작 완료
INFO:     서버 시작: http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Step 1-2: ngrok으로 외부 노출**

터미널 2:
```bash
ngrok http 8000
```

**예상 출력**:
```
Session Status     online
Account            your_account (Plan: Free)
Forwarding         https://abc-123-def.ngrok-free.app -> http://localhost:8000
```

**중요**: `https://abc-123-def.ngrok-free.app` URL을 복사 → 센터링크 팀에 전달

---

### Phase 2: 센터링크 콜백 서버 시작

**Step 2-1: 콜백 서버 실행**

터미널 3:
```bash
cd C:\Workspace\product_test_app
python centerlink_callback_server.py
```

**참고**: Phase 0에서 이미 실행했다면 생략 가능

**Step 2-2: ngrok으로 외부 노출**

터미널 4:
```bash
ngrok http 5000
```

**예상 출력**:
```
Forwarding         https://xyz-456-abc.ngrok-free.app -> http://localhost:5000
```

**중요**: `https://xyz-456-abc.ngrok-free.app` URL을 복사 → 배치 API 호출 시 사용

---

### Phase 3: 연동 테스트 실행

#### 테스트 1: 헬스 체크 ✅

**목적**: 서버 정상 동작 확인

**요청**:
```bash
curl https://abc-123-def.ngrok-free.app/api/v1/health
```

**예상 응답** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2025-10-01T10:30:00Z"
}
```

---

#### 테스트 2: 실시간 분석 API (SLM) ⚡

**목적**: 1-3초 이내 빠른 요약 생성 확인

**요청**:
```bash
curl -X POST https://abc-123-def.ngrok-free.app/api/v1/consultation/realtime-analyze \
  -H "Content-Type: application/json" \
  -H "X-Bound-Key: test_key_centerlink_2025" \
  -d '{
    "bound_key": "test_key_centerlink_2025",
    "consultation_id": "CENTERLINK_TEST_001",
    "stt_data": {
      "conversation_text": "상담사: 안녕하세요. 무엇을 도와드릴까요?\n고객: 보험 상품에 대해 문의드립니다.\n상담사: 어떤 보험 상품을 찾으시나요?\n고객: 건강 보험이요.\n상담사: 건강보험 상품을 안내해드리겠습니다. 현재 가입 가능한 상품은 여러 가지가 있습니다.\n고객: 보장 범위는 어떻게 되나요?\n상담사: 입원비, 수술비, 통원치료비 등을 보장합니다.\n고객: 월 보험료는 얼마인가요?\n상담사: 기본 상품은 월 5만원부터 시작합니다.\n고객: 알겠습니다. 자세한 상담 부탁드립니다.\n상담사: 네, 상세 안내를 위해 고객님 연락처를 남겨주시면 담당자가 연락드리겠습니다."
    }
  }'
```

**예상 응답** (200 OK, 1-3초 이내):
```json
{
  "success": true,
  "consultation_id": "CENTERLINK_TEST_001",
  "summary": "**고객**: 건강 보험 상품에 대해 문의하였습니다.\n**상담사**: 건강보험 상품의 보장 범위와 월 보험료를 안내하였습니다.\n**상담결과**: 고객이 상세 상담을 요청하여 연락처를 남기기로 하였습니다.",
  "processing_time": 2.5,
  "model": "Qwen3-1.7B",
  "timestamp": "2025-10-01T10:35:00Z"
}
```

**확인 사항**:
- ✅ `success: true`
- ✅ `summary` 필드에 한글 요약 존재
- ✅ `processing_time` 1-3초 이내
- ✅ HTTP 200 OK

---

#### 테스트 3: 배치 분석 API (LLM) 🔄

**목적**: 비동기 배치 처리 + 콜백 시스템 확인

**Step 3-1: 배치 요청**

```bash
curl -X POST https://abc-123-def.ngrok-free.app/api/v1/consultation/batch-analyze-async \
  -H "Content-Type: application/json" \
  -H "X-Bound-Key: test_key_centerlink_2025" \
  -d '{
    "bound_key": "test_key_centerlink_2025",
    "batch_id": "CENTERLINK_BATCH_001",
    "consultations": [
      {
        "consultation_id": "CALL_001",
        "stt_data": {
          "conversation_text": "상담사: 안녕하세요. 무엇을 도와드릴까요?\n고객: 대출 상품 문의합니다.\n상담사: 어떤 대출 상품을 원하시나요?\n고객: 주택담보대출이요.\n상담사: 주택담보대출 상품을 안내해드리겠습니다. 현재 금리는 연 3.5%부터 시작합니다.\n고객: 대출 한도는 어떻게 되나요?\n상담사: 주택 감정가의 최대 70%까지 가능합니다.\n고객: 상환 기간은요?\n상담사: 최대 30년까지 설정 가능합니다.\n고객: 알겠습니다. 신청 절차를 알려주세요.\n상담사: 네, 필요 서류를 안내해드리겠습니다."
        }
      },
      {
        "consultation_id": "CALL_002",
        "stt_data": {
          "conversation_text": "상담사: 반갑습니다. 무엇을 도와드릴까요?\n고객: 카드 발급 문의합니다.\n상담사: 신용카드를 원하시나요, 체크카드를 원하시나요?\n고객: 신용카드요.\n상담사: 어떤 혜택을 원하시나요?\n고객: 주유 할인이요.\n상담사: 주유 할인 카드를 안내해드리겠습니다. 리터당 100원 할인됩니다.\n고객: 연회비는 얼마인가요?\n상담사: 국내전용은 1만원, 해외겸용은 1만5천원입니다.\n고객: 신청하겠습니다.\n상담사: 네, 신청서를 보내드리겠습니다."
        }
      }
    ],
    "callback_url": "https://xyz-456-abc.ngrok-free.app/api/ai-callback",
    "llm_model": "qwen3_4b",
    "priority": 1
  }'
```

**예상 응답** (202 Accepted, 즉시):
```json
{
  "success": true,
  "batch_id": "CENTERLINK_BATCH_001",
  "status": "queued",
  "total_count": 2,
  "estimated_completion_time": 15.7,
  "callback_url": "https://xyz-456-abc.ngrok-free.app/api/ai-callback"
}
```

**확인 사항**:
- ✅ HTTP 202 Accepted (즉시 응답)
- ✅ `status: "queued"`
- ✅ `batch_id` 반환

**Step 3-2: 배치 상태 조회** (선택)

```bash
curl -X GET https://abc-123-def.ngrok-free.app/api/v1/consultation/batch-status/CENTERLINK_BATCH_001 \
  -H "X-Bound-Key: test_key_centerlink_2025"
```

**예상 응답**:
```json
{
  "success": true,
  "batch_id": "CENTERLINK_BATCH_001",
  "status": "processing",
  "total_count": 2,
  "processed_count": 1,
  "success_count": 1,
  "failed_count": 0
}
```

**Step 3-3: 콜백 수신 확인**

15-20초 후 센터링크 콜백 서버(터미널 3)에서 다음과 같은 로그 확인:

```
================================================================================
[AI 콜백 수신] 2025-10-01 10:40:15
================================================================================
Batch ID: CENTERLINK_BATCH_001
Status: completed
Total: 2
Success: 2
Results: 2개
================================================================================
```

**콜백 데이터 내용**:
```json
{
  "batch_id": "CENTERLINK_BATCH_001",
  "status": "completed",
  "total_count": 2,
  "success_count": 2,
  "failed_count": 0,
  "results": [
    {
      "consultation_id": "CALL_001",
      "success": true,
      "summary": "**고객**: 주택담보대출 상품에 대해 문의...",
      "categories": ["대출상품", "주택담보대출", "대출상담"],
      "titles": ["주택담보대출_상품_문의", "고객 주택담보대출 상담"],
      "processing_time": 7.5,
      "model": "Qwen3-4B-2507"
    },
    {
      "consultation_id": "CALL_002",
      "success": true,
      "summary": "**고객**: 주유 할인 신용카드 발급을 문의...",
      "categories": ["카드발급", "신용카드", "주유할인"],
      "titles": ["신용카드_발급_문의", "주유 할인 카드 신청 상담"],
      "processing_time": 8.2,
      "model": "Qwen3-4B-2507"
    }
  ],
  "total_processing_time": 15.8,
  "completed_at": "2025-10-01T10:40:15Z"
}
```

**확인 사항**:
- ✅ 콜백 서버가 200 OK 응답
- ✅ `status: "completed"`
- ✅ 모든 통화에 `summary`, `categories`, `titles` 포함
- ✅ 처리 시간 15-20초 이내

---

#### 테스트 4: 인증 실패 테스트 🔒

**목적**: 잘못된 바운드 키로 401 에러 확인

**요청**:
```bash
curl -X POST https://abc-123-def.ngrok-free.app/api/v1/consultation/realtime-analyze \
  -H "Content-Type: application/json" \
  -H "X-Bound-Key: invalid_key_12345" \
  -d '{
    "bound_key": "invalid_key_12345",
    "consultation_id": "AUTH_FAIL_TEST",
    "stt_data": {
      "conversation_text": "테스트"
    }
  }'
```

**예상 응답** (401 Unauthorized):
```json
{
  "detail": {
    "error": "유효하지 않은 바운드 키입니다",
    "error_code": "AUTH_KEY_INVALID"
  }
}
```

**확인 사항**:
- ✅ HTTP 401 Unauthorized
- ✅ `error_code: "AUTH_KEY_INVALID"`

---

## 📊 테스트 체크리스트

### AI 분석팀 체크리스트

- [ ] 서버 실행 확인 (`python main.py`)
- [ ] 배치 워커 시작 로그 확인
- [ ] ngrok 실행 및 URL 확인
- [ ] ngrok URL을 센터링크 팀에 전달
- [ ] 헬스 체크 API 정상 응답 확인
- [ ] 실시간 API 1-3초 응답 확인
- [ ] 배치 API 202 Accepted 확인
- [ ] 콜백 전송 성공 로그 확인

### 센터링크팀 체크리스트

- [ ] 콜백 서버 구현 완료
- [ ] 콜백 서버 실행 (`python centerlink_callback_server.py`)
- [ ] ngrok으로 콜백 서버 외부 노출
- [ ] 콜백 URL을 AI 팀에 전달
- [ ] 실시간 API 호출 성공 (200 OK)
- [ ] 배치 API 호출 성공 (202 Accepted)
- [ ] 콜백 수신 확인 (15-20초 후)
- [ ] 콜백 데이터 파싱 및 DB 저장 테스트
- [ ] 인증 실패 케이스 확인 (401)

---

## 🔍 문제 해결

### 문제 1: ngrok 연결 안됨

**증상**:
```
ERR_NGROK_108: Failed to connect to ngrok server
```

**해결**:
1. 인터넷 연결 확인
2. ngrok 재시작
3. ngrok 계정 로그인 (필요시)
   ```bash
   ngrok authtoken YOUR_AUTH_TOKEN
   ```

### 문제 2: 서버가 시작되지 않음

**증상**:
```
ImportError: cannot import name 'extract_conversation_text'
```

**해결**:
```bash
# 최신 코드로 업데이트 확인
git pull

# 환경 재설정
conda activate product_test
pip install -r requirements.txt
```

### 문제 3: 배치 API가 응답하지 않음

**증상**:
- 202 Accepted는 받았지만 콜백이 오지 않음

**해결**:
1. 배치 워커 로그 확인 (터미널 1)
2. 배치 상태 조회:
   ```bash
   curl -X GET https://abc-123-def.ngrok-free.app/api/v1/consultation/batch-status/BATCH_ID \
     -H "X-Bound-Key: test_key_centerlink_2025"
   ```
3. 큐 통계 확인:
   ```bash
   curl -X GET https://abc-123-def.ngrok-free.app/api/v1/consultation/batch-queue-stats \
     -H "X-Bound-Key: test_key_centerlink_2025"
   ```

### 문제 4: 콜백을 받지 못함

**증상**:
- 배치 처리 완료되었지만 콜백 서버에 데이터가 오지 않음

**해결**:
1. 콜백 서버 실행 상태 확인
2. ngrok 콜백 URL 정확성 확인
3. 콜백 서버가 200 OK 응답하는지 확인
4. 방화벽 설정 확인

### 문제 5: 한글이 깨짐

**증상**:
```json
{"summary": "\ubd88\ub7ec\uc628"}
```

**해결**:
- 이미 해결됨 (UTF-8 응답 설정 완료)
- curl에서 확인: `curl -s URL | jq -r .summary`

---

## 📞 연락처

### 테스트 중 문제 발생 시

**AI 분석팀**:
- 이메일: ai-support@company.com
- 전화: 000-0000-0000

**센터링크팀**:
- 이메일: dev-team@centerlink.com
- 전화: 000-0000-0000

---

## 📈 테스트 성공 기준

### 최소 성공 기준

- ✅ 실시간 API 3회 연속 성공 (1-3초 이내)
- ✅ 배치 API 1회 성공 (콜백 수신 확인)
- ✅ 인증 실패 케이스 정상 동작 (401)

### 권장 성공 기준

- ✅ 실시간 API 10회 연속 성공 (평균 2.5초 이내)
- ✅ 배치 API 5회 성공 (콜백 100% 수신)
- ✅ 20개 통화 배치 성공 (3분 이내 완료)
- ✅ 에러 케이스 처리 확인 (STT 데이터 없음, 배치 크기 초과 등)

---

## 📝 테스트 결과 보고서 양식

```markdown
# API 연동 테스트 결과

**테스트 일시**: 2025-10-01 10:00 - 11:00
**참여자**: AI 분석팀 (홍길동), 센터링크팀 (김철수)

## 테스트 환경
- AI 서버 URL: https://abc-123-def.ngrok-free.app
- 콜백 서버 URL: https://xyz-456-abc.ngrok-free.app
- 바운드 키: test_key_centerlink_2025

## 테스트 결과

### 1. 실시간 API (SLM)
- 총 시도: 10회
- 성공: 10회
- 실패: 0회
- 평균 응답시간: 2.3초
- 결과: ✅ 성공

### 2. 배치 API (LLM)
- 총 시도: 3회
- 성공: 3회
- 실패: 0회
- 평균 처리시간: 15.8초 (2개 통화 기준)
- 콜백 수신: 3/3 (100%)
- 결과: ✅ 성공

### 3. 인증 테스트
- 잘못된 키로 401 확인: ✅ 성공

## 발견된 이슈
- 없음

## 다음 단계
- 운영 환경 준비
- 실제 데이터로 파일럿 테스트
```

---

**작성**: AI 분석팀
**최종 수정**: 2025-10-01
**버전**: 1.0.0
