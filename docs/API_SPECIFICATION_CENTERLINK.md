# AI 상담 분석 시스템 API 명세서 (센터링크 연동용)

**버전**: 1.0.0
**최종 수정일**: 2025-10-01
**담당**: AI 상담 분석팀
**연동 대상**: 센터링크팀

---

## 목차

1. [개요](#개요)
2. [인증 방식](#인증-방식)
3. [API 엔드포인트](#api-엔드포인트)
4. [에러 코드](#에러-코드)
5. [사용 예시](#사용-예시)
6. [FAQ](#faq)

---

## 개요

### 시스템 구성

AI 상담 분석 시스템은 **듀얼-티어 아키텍처**로 구성되어 있습니다:

| 티어 | 모델 | 용도 | 응답시간 | 기능 |
|------|------|------|----------|------|
| **SLM** | Qwen3-1.7B | 실시간 상담 지원 | 1-3초 | 간략 요약 |
| **LLM** | Qwen3-4B | 배치 분석 | 7-23초/통화 | 상세 요약+키워드+제목 |

### 사용 시나리오

#### 1. 실시간 상담 지원 (SLM)
상담사가 상담 중 또는 직후 빠르게 요약이 필요한 경우

```
상담 종료 → STT 데이터 전송 → 1-3초 내 간략 요약 수신 → 상담사 확인
```

#### 2. 배치 상담 분석 (LLM)
15분 단위로 약 20개 통화를 모아서 상세 분석이 필요한 경우

```
15분 대기 → 배치 전송 → 큐 접수 → 백그라운드 처리 → 콜백으로 결과 수신
```

---

## 인증 방식

### 바운드 키 (Bound Key) 인증

모든 API 요청에는 **바운드 키**가 필수입니다.

#### 발급 방법

1. AI 상담 분석팀에 바운드 키 발급 요청
2. 용도(실시간/배치) 및 환경(개발/운영) 명시
3. 발급된 키는 환경변수 또는 안전한 저장소에 보관

#### 사용 방법

**방법 1: X-Bound-Key 헤더 (권장)**
```http
POST /api/v1/consultation/realtime-analyze
Host: api.example.com
Content-Type: application/json
X-Bound-Key: your_bound_key_here_20_chars_minimum

{...}
```

**방법 2: Authorization 헤더**
```http
POST /api/v1/consultation/realtime-analyze
Host: api.example.com
Content-Type: application/json
Authorization: Bearer your_bound_key_here_20_chars_minimum

{...}
```

#### 권한 종류

| 권한 | 설명 | 사용 가능 API |
|------|------|---------------|
| `realtime` | 실시간 처리 권한 | `/realtime-analyze` |
| `batch` | 배치 처리 권한 | `/batch-analyze-async`, `/batch-status` |

---

## API 엔드포인트

### 1. 실시간 상담 분석 (SLM)

#### `POST /api/v1/consultation/realtime-analyze`

상담사 실시간 지원을 위한 빠른 상담 요약 API

**요청 (Request)**

```json
{
  "bound_key": "your_bound_key_here",
  "consultation_id": "CALL_20251001_001",
  "stt_data": {
    "conversation_text": "상담사: 안녕하세요. 무엇을 도와드릴까요?\n고객: 보험 상품 문의드립니다..."
  },
  "metadata": {
    "agent_id": "AGENT_001",
    "customer_id": "CUST_12345"
  }
}
```

**요청 필드 설명**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `bound_key` | string | ✅ | 센터링크 바운드 키 (20자 이상) |
| `consultation_id` | string | ✅ | 상담 고유 ID |
| `stt_data` | object | ✅ | STT 변환 데이터 |
| `stt_data.conversation_text` | string | ✅ | 완성된 대화 텍스트 |
| `metadata` | object | ❌ | 추가 메타데이터 (선택) |

**응답 (Response) - 성공 (200 OK)**

```json
{
  "success": true,
  "consultation_id": "CALL_20251001_001",
  "summary": "**고객**: 보험 상품에 대해 문의하였습니다.\n**상담사**: 가입 가능한 보험 상품을 안내하고 가입 절차를 설명하였습니다.\n**상담결과**: 고객이 보험 상품 정보를 확인하고 추후 가입 의사를 밝혔습니다.",
  "processing_time": 2.5,
  "model": "Qwen3-1.7B",
  "timestamp": "2025-10-01T12:00:00Z"
}
```

**응답 필드 설명**

| 필드 | 타입 | 설명 |
|------|------|------|
| `success` | boolean | 분석 성공 여부 |
| `consultation_id` | string | 상담 ID (요청과 동일) |
| `summary` | string | 3줄 구조 요약 (고객/상담사/상담결과) |
| `processing_time` | number | 처리 시간(초) |
| `model` | string | 사용된 모델명 |
| `timestamp` | string | 처리 완료 시각 (ISO 8601) |

**응답 - 실패 (401 Unauthorized)**

```json
{
  "success": false,
  "consultation_id": "CALL_20251001_001",
  "summary": null,
  "processing_time": 0.01,
  "model": "Qwen3-1.7B",
  "timestamp": "2025-10-01T12:00:00Z",
  "error": "유효하지 않은 바운드 키입니다",
  "error_code": "AUTH_KEY_INVALID"
}
```

**성능 목표**

- **응답 시간**: 1-3초 (평균 2.83초)
- **성공률**: 100%
- **품질**: 0.800/1.000

---

### 2. 비동기 배치 분석 (LLM)

#### `POST /api/v1/consultation/batch-analyze-async`

대량 상담 데이터를 비동기로 처리하는 API

**요청 (Request)**

```json
{
  "bound_key": "your_bound_key_here",
  "batch_id": "BATCH_20251001_0001",
  "consultations": [
    {
      "consultation_id": "CALL_001",
      "stt_data": {
        "conversation_text": "상담 내용..."
      }
    },
    {
      "consultation_id": "CALL_002",
      "stt_data": {
        "conversation_text": "상담 내용..."
      }
    }
    // ... 최대 20개
  ],
  "callback_url": "https://centerlink.example.com/api/callback",
  "llm_model": "qwen3_4b",
  "priority": 1
}
```

**요청 필드 설명**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `bound_key` | string | ✅ | 센터링크 바운드 키 |
| `batch_id` | string | ✅ | 배치 고유 ID (중복 불가) |
| `consultations` | array | ✅ | 상담 데이터 배열 (최대 20개) |
| `callback_url` | string | ✅ | 결과 전송할 콜백 URL |
| `llm_model` | string | ❌ | LLM 모델 선택 (기본: qwen3_4b) |
| `priority` | number | ❌ | 우선순위 (1: 높음, 2: 중간, 3: 낮음) |

**LLM 모델 옵션**

| 모델 코드 | 모델명 | 평균 처리시간 | 품질 | 특징 |
|-----------|--------|---------------|------|------|
| `qwen3_4b` | Qwen3-4B-2507 | 7.85초 | 0.992 | 빠른 속도 (권장) |
| `ax_light` | A.X-4.0-Light | 22.58초 | 0.995 | 최고 품질 |

**응답 (Response) - 성공 (202 Accepted)**

```json
{
  "success": true,
  "batch_id": "BATCH_20251001_0001",
  "status": "queued",
  "total_count": 20,
  "estimated_completion_time": 157.0,
  "callback_url": "https://centerlink.example.com/api/callback"
}
```

**응답 필드 설명**

| 필드 | 타입 | 설명 |
|------|------|------|
| `success` | boolean | 접수 성공 여부 |
| `batch_id` | string | 배치 ID |
| `status` | string | 배치 상태 (queued/processing/completed/failed) |
| `total_count` | number | 총 상담 수 |
| `estimated_completion_time` | number | 예상 완료 시간(초) |
| `callback_url` | string | 콜백 URL |

---

### 3. 배치 콜백 (센터링크 수신용)

#### 배치 처리 완료 후 센터링크로 전송

배치 처리가 완료되면 AI 시스템이 센터링크의 `callback_url`로 결과를 POST 전송합니다.

**콜백 요청 (AI 시스템 → 센터링크)**

```json
{
  "batch_id": "BATCH_20251001_0001",
  "status": "completed",
  "total_count": 20,
  "success_count": 20,
  "failed_count": 0,
  "results": [
    {
      "consultation_id": "CALL_001",
      "success": true,
      "summary": "**고객**: ...\n**상담사**: ...\n**상담결과**: ...",
      "categories": ["보험", "안내", "가입절차"],
      "titles": [
        {
          "title": "보험상품_문의",
          "type": "keyword"
        },
        {
          "title": "고객이 보험 상품에 대해 문의함",
          "type": "descriptive"
        }
      ],
      "processing_time": 7.5
    }
    // ... 나머지 19개
  ],
  "total_processing_time": 157.2,
  "completed_at": "2025-10-01T12:05:00Z"
}
```

**콜백 응답 (센터링크 → AI 시스템)**

센터링크는 콜백을 받으면 **200 OK** 응답을 반환해야 합니다:

```json
{
  "received": true,
  "batch_id": "BATCH_20251001_0001"
}
```

**콜백 재시도 정책**

- 최대 재시도 횟수: 3회
- 재시도 간격: 5초
- 타임아웃: 30초

---

### 4. 배치 상태 조회

#### `GET /api/v1/consultation/batch-status/{batch_id}`

배치 작업의 현재 상태 및 진행률 조회

**요청 (Request)**

```http
GET /api/v1/consultation/batch-status/BATCH_20251001_0001
X-Bound-Key: your_bound_key_here
```

**응답 (Response)**

```json
{
  "success": true,
  "batch_id": "BATCH_20251001_0001",
  "status": "processing",
  "total_count": 20,
  "processed_count": 12,
  "success_count": 12,
  "failed_count": 0,
  "llm_model": "qwen3_4b",
  "priority": 1,
  "callback_url": "https://centerlink.example.com/api/callback",
  "created_at": "2025-10-01T12:00:00Z",
  "started_at": "2025-10-01T12:00:05Z",
  "completed_at": null,
  "processing_time": null
}
```

---

### 5. 배치 작업 취소

#### `DELETE /api/v1/consultation/batch-cancel/{batch_id}`

대기 중인 배치 작업 취소 (처리 중/완료된 작업은 취소 불가)

**요청 (Request)**

```http
DELETE /api/v1/consultation/batch-cancel/BATCH_20251001_0001
X-Bound-Key: your_bound_key_here
```

**응답 (Response) - 성공**

```json
{
  "success": true,
  "batch_id": "BATCH_20251001_0001",
  "message": "배치 작업이 취소되었습니다"
}
```

**응답 - 실패 (이미 처리 중)**

```json
{
  "detail": {
    "error": "배치 작업을 취소할 수 없습니다 (이미 처리 중이거나 완료됨)",
    "error_code": "BATCH_CANCEL_FAILED",
    "batch_id": "BATCH_20251001_0001"
  }
}
```

---

## 에러 코드

### 인증 관련 에러

| 코드 | HTTP 상태 | 설명 | 해결 방법 |
|------|-----------|------|-----------|
| `AUTH_KEY_MISSING` | 401 | 바운드 키가 제공되지 않음 | X-Bound-Key 헤더 추가 |
| `AUTH_KEY_INVALID_FORMAT` | 401 | 키 형식 오류 (20자 미만) | 올바른 키 형식 사용 |
| `AUTH_KEY_INVALID` | 401 | 유효하지 않은 키 | AI팀에 키 확인 요청 |
| `AUTH_KEY_EXPIRED` | 401 | 만료된 키 | 새 키 발급 요청 |
| `AUTH_PERMISSION_DENIED` | 403 | 권한 없음 | 해당 API 권한 요청 |

### 데이터 관련 에러

| 코드 | HTTP 상태 | 설명 | 해결 방법 |
|------|-----------|------|-----------|
| `DATA_INVALID_STT` | 400 | STT 데이터 형식 오류 | conversation_text 필드 확인 |
| `BATCH_SIZE_EXCEEDED` | 400 | 배치 크기 초과 (20개 초과) | 배치를 20개 이하로 분할 |
| `BATCH_EMPTY` | 400 | 빈 배치 | 최소 1개 이상 상담 포함 |
| `BATCH_DUPLICATE_ID` | 400 | 중복된 배치 ID | 새 배치 ID 생성 |
| `BATCH_NOT_FOUND` | 404 | 배치를 찾을 수 없음 | 배치 ID 확인 |

### AI 처리 관련 에러

| 코드 | HTTP 상태 | 설명 | 해결 방법 |
|------|-----------|------|-----------|
| `AI_SUMMARIZATION_FAILED` | 500 | AI 요약 생성 실패 | AI팀에 문의 |
| `SERVER_INTERNAL_ERROR` | 500 | 서버 내부 오류 | AI팀에 문의 |

---

## 사용 예시

### 예시 1: 실시간 상담 분석 (Python)

```python
import requests

# 설정
API_URL = "https://api.example.com/api/v1/consultation/realtime-analyze"
BOUND_KEY = "your_bound_key_here_20_chars_minimum"

# 요청 데이터
payload = {
    "bound_key": BOUND_KEY,
    "consultation_id": "CALL_20251001_001",
    "stt_data": {
        "conversation_text": "상담사: 안녕하세요\n고객: 보험 문의드립니다..."
    }
}

# 요청 전송
response = requests.post(
    API_URL,
    json=payload,
    headers={"X-Bound-Key": BOUND_KEY}
)

# 응답 처리
if response.status_code == 200:
    result = response.json()
    print(f"요약: {result['summary']}")
    print(f"처리시간: {result['processing_time']}초")
else:
    print(f"에러: {response.json()}")
```

### 예시 2: 비동기 배치 분석 (Python)

```python
import requests

# 설정
API_URL = "https://api.example.com/api/v1/consultation/batch-analyze-async"
BOUND_KEY = "your_bound_key_here_20_chars_minimum"

# 배치 데이터
payload = {
    "bound_key": BOUND_KEY,
    "batch_id": "BATCH_20251001_0001",
    "consultations": [
        {
            "consultation_id": f"CALL_{i:03d}",
            "stt_data": {"conversation_text": f"상담 내용 {i}..."}
        }
        for i in range(1, 21)  # 20개
    ],
    "callback_url": "https://centerlink.example.com/api/callback",
    "llm_model": "qwen3_4b",
    "priority": 1
}

# 배치 접수
response = requests.post(
    API_URL,
    json=payload,
    headers={"X-Bound-Key": BOUND_KEY}
)

# 응답 처리
if response.status_code == 202:
    result = response.json()
    print(f"배치 접수 완료: {result['batch_id']}")
    print(f"예상 완료 시간: {result['estimated_completion_time']}초")
else:
    print(f"에러: {response.json()}")
```

### 예시 3: 콜백 수신 (센터링크 측 구현 필요)

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/callback', methods=['POST'])
def receive_batch_result():
    """AI 시스템으로부터 배치 결과 수신"""

    data = request.json
    batch_id = data['batch_id']
    results = data['results']

    print(f"배치 완료: {batch_id}")
    print(f"성공: {data['success_count']}/{data['total_count']}")

    # 결과 처리 (DB 저장 등)
    for result in results:
        consultation_id = result['consultation_id']
        summary = result['summary']
        # DB에 저장...

    # 200 OK 응답 필수
    return jsonify({
        "received": True,
        "batch_id": batch_id
    }), 200

if __name__ == '__main__':
    app.run(port=5000)
```

---

## FAQ

### Q1. 바운드 키는 어떻게 발급받나요?

**A**: AI 상담 분석팀(담당자: xxx@company.com)에 다음 정보와 함께 요청하세요:
- 사용 용도 (실시간/배치)
- 환경 (개발/스테이징/운영)
- 예상 사용량 (일 통화 수)

### Q2. 실시간 API와 배치 API 중 무엇을 사용해야 하나요?

**A**:
- **실시간 API**: 상담사가 즉시 확인해야 하는 경우 (1-3초 응답)
- **배치 API**: 15분 단위로 모아서 상세 분석이 필요한 경우 (요약+키워드+제목)

### Q3. 배치 크기는 왜 20개로 제한되나요?

**A**: 메모리 효율성과 처리 시간의 균형을 위해 20개로 제한됩니다. 20개 초과 시 배치를 나누어 전송하세요.

### Q4. 콜백이 실패하면 어떻게 되나요?

**A**: 최대 3회 재시도(5초 간격) 후, 배치 상태 조회 API로 결과를 확인할 수 있습니다.

### Q5. 테스트 환경이 있나요?

**A**: 개발 환경 URL: `https://dev-api.example.com`
테스트 키: `dev_key_12345678901234`

---

**문의**: AI 상담 분석팀
**이메일**: ai-support@company.com
**문서 버전**: 1.0.0
**최종 수정**: 2025-10-01
