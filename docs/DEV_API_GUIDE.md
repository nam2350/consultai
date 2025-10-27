# 개발 전용 API 가이드 (인증 없음)

**⚠️ 경고**: 이 API는 개발/테스트 전용입니다.
- DEBUG 모드에서만 활성화됩니다
- 운영 환경에서는 절대 사용하지 마세요
- 바운드 키 인증이 필요 없습니다

---

## 📋 개요

### 목적
개발 및 테스트 단계에서 **빠른 API 검증**을 위해 인증 절차를 생략한 엔드포인트를 제공합니다.

### 활성화 조건
```python
# .env 파일
DEBUG=true  # DEBUG 모드일 때만 활성화
```

서버 시작 시 다음 로그 확인:
```
⚠️ [개발 모드] 인증 없는 개발 전용 API가 활성화되었습니다 (/api/v1/dev/*)
```

---

## 🔗 엔드포인트

### 1. 실시간 상담 분석 (인증 없음)

**엔드포인트**: `POST /api/v1/dev/realtime-analyze-no-auth`

**특징**:
- ✅ 바운드 키 불필요
- ✅ 1-3초 빠른 응답
- ✅ SLM 모델 사용 (Qwen3-1.7B)

**요청 예시**:
```bash
curl -X POST http://localhost:8000/api/v1/dev/realtime-analyze-no-auth \
  -H "Content-Type: application/json" \
  -d '{
    "consultation_id": "DEV_TEST_001",
    "stt_data": {
      "conversation_text": "상담사: 안녕하세요. 무엇을 도와드릴까요?\n고객: 보험 상품에 대해 문의드립니다."
    }
  }'
```

**응답 예시** (200 OK):
```json
{
  "success": true,
  "consultation_id": "DEV_TEST_001",
  "summary": "**고객**: 보험 상품에 대해 문의하였습니다.\n**상담사**: 안내를 제공하였습니다.\n**상담결과**: 상담이 진행되었습니다.",
  "processing_time": 2.3,
  "model": "Qwen3-1.7B (개발 모드)",
  "timestamp": "2025-10-16T10:30:00Z",
  "error": null,
  "error_code": null
}
```

---

### 2. 개발 API 상태 조회

**엔드포인트**: `GET /api/v1/dev/status`

**요청**:
```bash
curl http://localhost:8000/api/v1/dev/status
```

**응답**:
```json
{
  "status": "active",
  "warning": "⚠️ 이 엔드포인트는 개발/테스트 전용입니다. 운영 환경에서는 사용하지 마세요.",
  "authentication": "disabled",
  "model_loaded": true,
  "model_name": "Qwen3-1.7B",
  "endpoints": {
    "realtime_no_auth": "/api/v1/dev/realtime-analyze-no-auth"
  },
  "usage_note": "바운드 키 없이 API를 호출할 수 있습니다 (테스트용)"
}
```

---

### 3. 개발 API 테스트

**엔드포인트**: `GET /api/v1/dev/test`

**요청**:
```bash
curl http://localhost:8000/api/v1/dev/test
```

**응답**:
```json
{
  "message": "개발 전용 API가 정상 작동 중입니다",
  "timestamp": "2025-10-16T10:30:00Z",
  "warning": "⚠️ 이 엔드포인트는 개발/테스트 전용입니다",
  "authentication": "disabled"
}
```

---

## 🆚 인증 방식 비교

### 개발 전용 (인증 없음)

**장점**:
- ⚡ 가장 빠른 테스트
- 📝 헤더 설정 불필요
- 🔧 간편한 디버깅

**단점**:
- ⚠️ 보안 검증 불가
- ❌ 운영 환경 사용 불가
- ❌ 권한 관리 불가

**사용 예시**:
```bash
# 헤더 없이 바로 호출
curl -X POST http://localhost:8000/api/v1/dev/realtime-analyze-no-auth \
  -H "Content-Type: application/json" \
  -d '{...}'
```

---

### 운영 방식 (바운드 키 인증)

**장점**:
- ✅ 실제 운영 환경과 동일
- ✅ 보안 검증 가능
- ✅ 권한 관리 가능

**단점**:
- 📝 매번 헤더 추가 필요
- 🔑 키 관리 필요

**사용 예시**:
```bash
# 바운드 키 헤더 필수
curl -X POST http://localhost:8000/api/v1/consultation/realtime-analyze \
  -H "X-Bound-Key: test_key_centerlink_2025" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

---

## 📊 사용 시나리오

### 시나리오 1: 로컬 개발/디버깅

```bash
# 개발 전용 API 사용 (가장 빠름)
curl -X POST http://localhost:8000/api/v1/dev/realtime-analyze-no-auth \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

**추천**: 로컬에서 빠르게 테스트할 때

---

### 시나리오 2: 센터링크와 연동 테스트

```bash
# 운영 방식 API 사용 (바운드 키)
curl -X POST https://abc-123.ngrok-free.app/api/v1/consultation/realtime-analyze \
  -H "X-Bound-Key: test_key_centerlink_2025" \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

**추천**: 실제 연동 테스트 시

---

### 시나리오 3: 자동화 스크립트

```python
import requests

# 개발 모드: 인증 없음
response = requests.post(
    "http://localhost:8000/api/v1/dev/realtime-analyze-no-auth",
    json={
        "consultation_id": "AUTO_001",
        "stt_data": {"conversation_text": "..."}
    }
)

# 운영 모드: 바운드 키
response = requests.post(
    "https://api.example.com/api/v1/consultation/realtime-analyze",
    headers={"X-Bound-Key": "test_key_centerlink_2025"},
    json={
        "bound_key": "test_key_centerlink_2025",
        "consultation_id": "AUTO_001",
        "stt_data": {"conversation_text": "..."}
    }
)
```

---

## ⚠️ 주의사항

### 1. 운영 환경에서 비활성화

**자동 비활성화**:
```bash
# .env 파일
DEBUG=false  # 운영 환경
```

서버 시작 시 개발 API가 로드되지 않음:
```
INFO: 서버 시작...
INFO: 애플리케이션 초기화 완료
# ⚠️ 개발 API 관련 로그 없음
```

---

### 2. 보안 고려사항

**개발 전용 API는**:
- ❌ 인증 검증 없음
- ❌ 권한 관리 없음
- ❌ 사용량 제한 없음
- ❌ 감사 로그 없음

**운영 환경에서는 반드시**:
- ✅ 바운드 키 인증 사용
- ✅ 권한 관리 활성화
- ✅ Rate Limiting 설정
- ✅ 접근 로그 기록

---

### 3. 에러 처리

**개발 API에서도 동일한 에러 응답**:

```json
{
  "success": false,
  "consultation_id": "DEV_TEST_001",
  "summary": null,
  "processing_time": 0.5,
  "model": "Qwen3-1.7B",
  "timestamp": "2025-10-16T10:30:00Z",
  "error": "STT 데이터 처리 실패: 대화 내용이 너무 짧거나 없습니다",
  "error_code": "DATA_INVALID_STT"
}
```

---

## 🔄 개발 → 운영 전환 가이드

### Step 1: 개발 단계 (인증 없음)

```bash
# 로컬에서 빠른 테스트
curl http://localhost:8000/api/v1/dev/realtime-analyze-no-auth \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### Step 2: 통합 테스트 (테스트 키)

```bash
# 센터링크와 연동 테스트
curl https://ngrok-url.com/api/v1/consultation/realtime-analyze \
  -H "X-Bound-Key: test_key_centerlink_2025" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### Step 3: 운영 배포 (운영 키)

```bash
# 운영 환경 (개발 API 자동 비활성화)
DEBUG=false
BOUND_KEYS=centerlink_prod_key_2025_secure_random_string

# 운영 API 호출
curl https://api.production.com/api/v1/consultation/realtime-analyze \
  -H "X-Bound-Key: centerlink_prod_key_2025_secure_random_string" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

---

## 📝 요약

| 항목 | 개발 전용 API | 운영 API |
|------|--------------|---------|
| **엔드포인트** | `/api/v1/dev/realtime-analyze-no-auth` | `/api/v1/consultation/realtime-analyze` |
| **인증** | ❌ 불필요 | ✅ 바운드 키 필수 |
| **활성화 조건** | `DEBUG=true` | 항상 활성화 |
| **사용 목적** | 로컬 개발/디버깅 | 운영 환경 |
| **보안** | ⚠️ 없음 | ✅ 완전 |
| **권장 사용처** | 로컬 테스트 | 실제 연동 |

---

**작성**: AI 분석팀
**최종 수정**: 2025-10-16
**버전**: 1.0.0
