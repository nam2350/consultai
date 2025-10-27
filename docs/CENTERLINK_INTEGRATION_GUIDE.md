# 센터링크 API 연동 가이드

**최종 수정**: 2025-10-16
**대상**: 센터링크 개발팀
**소요 시간**: 15분

---

## 📋 목차

1. [빠른 시작 (5분)](#빠른-시작)
2. [개발 환경 설정](#개발-환경-설정)
3. [API 사용법](#api-사용법)
4. [문제 해결](#문제-해결)

---

## 🚀 빠른 시작

### 1. 인증 방식 선택

**방식 A: 개발 전용 (인증 없음) - 가장 빠름** ⚡
```bash
# 바운드 키 없이 바로 테스트 가능
curl -X POST https://api.example.com/api/v1/dev/realtime-analyze-no-auth \
  -H "Content-Type: application/json" \
  -d '{...}'
```

**방식 B: 운영 방식 (바운드 키 인증) - 권장** ✅
```bash
# 바운드 키: test_key_centerlink_2025
curl -X POST https://api.example.com/api/v1/consultation/realtime-analyze \
  -H "X-Bound-Key: test_key_centerlink_2025" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

> **참고**: 개발 전용 API는 DEBUG 모드에서만 활성화됩니다. 운영 환경에서는 반드시 바운드 키 인증을 사용하세요.

---

### 2. 실시간 API 테스트 (1-3초 응답)

**A. 개발 전용 (인증 없음)** ⚡:

```bash
curl -X POST https://api.example.com/api/v1/dev/realtime-analyze-no-auth \
  -H "Content-Type: application/json" \
  -d '{
    "consultation_id": "TEST_001",
    "stt_data": {
      "conversation_text": "상담사: 안녕하세요.\n고객: 보험 문의합니다."
    }
  }'
```

**B. 운영 방식 (바운드 키 인증)** ✅:

```bash
curl -X POST https://api.example.com/api/v1/consultation/realtime-analyze \
  -H "Content-Type: application/json" \
  -H "X-Bound-Key: test_key_centerlink_2025" \
  -d '{
    "bound_key": "test_key_centerlink_2025",
    "consultation_id": "TEST_001",
    "stt_data": {
      "conversation_text": "상담사: 안녕하세요.\n고객: 보험 문의합니다."
    }
  }'
```

**응답 예시** (공통):
```json
{
  "success": true,
  "consultation_id": "TEST_001",
  "summary": "**고객**: 보험 상품 문의\n**상담사**: 안내 제공\n**결과**: 상담 완료",
  "processing_time": 2.5,
  "model": "Qwen3-1.7B",
  "timestamp": "2025-10-16T12:00:00Z"
}
```

### 3. 배치 API 테스트 (비동기)

```bash
curl -X POST https://api.example.com/api/v1/consultation/batch-analyze-async \
  -H "Content-Type: application/json" \
  -H "X-Bound-Key: test_key_centerlink_2025" \
  -d '{
    "bound_key": "test_key_centerlink_2025",
    "batch_id": "BATCH_001",
    "consultations": [
      {
        "consultation_id": "CALL_001",
        "stt_data": {"conversation_text": "상담 내용..."}
      }
    ],
    "callback_url": "https://your-server.com/api/callback",
    "llm_model": "qwen3_4b"
  }'
```

**즉시 응답 (202 Accepted)**:
```json
{
  "success": true,
  "batch_id": "BATCH_001",
  "status": "queued",
  "estimated_completion_time": 15.7
}
```

**15-20초 후 콜백 수신**:
- 센터링크 서버의 `callback_url`로 결과 전송
- 반드시 200 OK 응답 필요

---

## 🔧 개발 환경 설정

### 로컬 테스트 (ngrok 사용)

**AI 서버 측 (개발팀)**:

```bash
# 1. 서버 실행
python main.py

# 2. ngrok으로 외부 노출
ngrok http 8000

# 3. 생성된 URL 공유
# https://abc-123.ngrok-free.app
```

**센터링크 측**:

```bash
# 1. 콜백 서버 실행
python centerlink_callback_server.py  # 포트 5000

# 2. ngrok으로 외부 노출
ngrok http 5000

# 3. 콜백 URL 사용
# https://xyz-456.ngrok-free.app/api/callback
```

### 콜백 엔드포인트 구현

**Python Flask**:
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/callback', methods=['POST'])
def receive_callback():
    data = request.json

    # 결과 처리
    batch_id = data['batch_id']
    results = data['results']

    # DB 저장 로직
    save_to_database(results)

    # 반드시 200 OK 응답
    return jsonify({"received": True}), 200

if __name__ == '__main__':
    app.run(port=5000)
```

**Node.js Express**:
```javascript
const express = require('express');
const app = express();

app.use(express.json());

app.post('/api/callback', (req, res) => {
  const { batch_id, results } = req.body;

  // 결과 처리
  saveToDatabase(results);

  // 반드시 200 OK 응답
  res.json({ received: true });
});

app.listen(5000);
```

---

## 📖 API 사용법

### 실시간 분석 API (SLM)

**용도**: 상담 중/직후 즉시 요약

**특징**:
- 응답 시간: 1-3초
- 모델: Qwen3-1.7B
- 기능: 간략 요약만

**요청 형식**:
```json
{
  "bound_key": "your_key",
  "consultation_id": "unique_id",
  "stt_data": {
    "conversation_text": "상담 내용..."
  }
}
```

**응답 형식**:
```json
{
  "success": true,
  "consultation_id": "unique_id",
  "summary": "3줄 요약",
  "processing_time": 2.5,
  "model": "Qwen3-1.7B",
  "timestamp": "2025-10-16T12:00:00Z"
}
```

### 배치 분석 API (LLM)

**용도**: 15분마다 ~20개 통화 고품질 분석

**특징**:
- 응답: 즉시 202 Accepted
- 처리: 배경에서 비동기 처리
- 결과: 콜백 URL로 전송
- 기능: 요약 + 키워드 + 제목

**요청 형식**:
```json
{
  "bound_key": "your_key",
  "batch_id": "unique_batch_id",
  "consultations": [
    {
      "consultation_id": "call_001",
      "stt_data": {"conversation_text": "..."}
    }
  ],
  "callback_url": "https://your-server.com/callback",
  "llm_model": "qwen3_4b",
  "priority": 1
}
```

**콜백 데이터**:
```json
{
  "batch_id": "unique_batch_id",
  "status": "completed",
  "total_count": 20,
  "success_count": 20,
  "results": [
    {
      "consultation_id": "call_001",
      "success": true,
      "summary": "3줄 요약",
      "categories": ["키워드1", "키워드2", "키워드3"],
      "titles": ["제목1", "제목2"],
      "processing_time": 7.5,
      "model": "Qwen3-4B-2507"
    }
  ]
}
```

### 배치 상태 조회

```bash
curl -X GET https://api.example.com/api/v1/consultation/batch-status/{batch_id} \
  -H "X-Bound-Key: your_key"
```

**응답**:
```json
{
  "batch_id": "unique_batch_id",
  "status": "processing",
  "total_count": 20,
  "processed_count": 10,
  "success_count": 10
}
```

---

## 🔍 문제 해결

### 401 Unauthorized

**원인**: 바운드 키 오류

**해결**:
1. 키가 20자 이상인지 확인
2. 헤더 이름: `X-Bound-Key` 또는 `Authorization: Bearer {key}`
3. AI 팀에 키 유효성 확인

### 403 Forbidden

**원인**: 권한 부족

**해결**:
- `realtime` 권한: 실시간 API 필요
- `batch` 권한: 배치 API 필요
- AI 팀에 권한 추가 요청

### 콜백이 안 옴

**원인**: 콜백 URL 문제

**해결**:
1. URL이 외부에서 접근 가능한지 확인 (ngrok 사용)
2. 콜백 서버가 200 OK 응답하는지 확인
3. 배치 상태 조회로 수동 확인

### 처리 시간 너무 느림

**원인**: 모델 선택 또는 배치 크기

**해결**:
1. 실시간: SLM 사용 (자동)
2. 배치: `qwen3_4b` 권장 (7.85초/통화)
3. 배치 크기: 최대 20개

---

## 📋 체크리스트

### 개발 환경

- [ ] 테스트 키 확보
- [ ] 실시간 API 호출 성공
- [ ] 배치 API 호출 성공 (202)
- [ ] 콜백 서버 구현
- [ ] 콜백 수신 확인

### 운영 환경

- [ ] 운영 키 발급 요청
- [ ] 운영 URL로 변경
- [ ] 콜백 URL 운영 서버로 변경
- [ ] 에러 처리 구현
- [ ] 모니터링 설정
- [ ] 부하 테스트 완료

---

## 📚 추가 문서

**빠른 시작**:
- **[빠른 테스트 시작](./QUICK_TEST_START.md)** - 2-5분 안에 API 테스트 시작 (인증/비인증 방식 모두 포함)
- [개발 전용 API 가이드](./DEV_API_GUIDE.md) - 인증 없는 개발 전용 API 상세 가이드

**상세 문서**:
- [API 상세 명세서](./API_SPECIFICATION_CENTERLINK.md) - 완전한 API 스펙 및 에러 코드
- [테스트 매뉴얼](./CENTERLINK_API_TEST_MANUAL.md) - 단계별 상세 테스트 절차
- [시스템 아키텍처](./dual_model_architecture.md) - 듀얼-티어 AI 아키텍처

---

## 📞 지원

**문의**: ai-support@company.com
**긴급**: 별도 협의
**응답 시간**: 영업일 24시간 이내
