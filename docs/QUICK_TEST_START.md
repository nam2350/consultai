# 센터링크 API 빠른 테스트 시작 가이드

**소요 시간**: 2-5분
**목적**: API 연동 테스트 빠르게 시작하기

---

## 📋 테스트 방식 선택

### 방식 A: 개발 전용 (인증 없음) ⚡ - 가장 빠름 (2분)

**장점**:
- ✅ 바운드 키 불필요
- ✅ 헤더 설정 최소화
- ✅ 가장 빠른 테스트

**단점**:
- ⚠️ DEBUG 모드에서만 동작
- ⚠️ 보안 검증 불가

👉 **[방식 A로 시작하기](#방식-a-개발-전용-인증-없음)**

---

### 방식 B: 운영 방식 (바운드 키 인증) 🔒 - 권장 (5분)

**장점**:
- ✅ 실제 운영 환경과 동일
- ✅ 보안 검증 가능
- ✅ 배치 API 테스트 포함

**단점**:
- 📝 바운드 키 설정 필요
- 📝 콜백 서버 추가 실행

👉 **[방식 B로 시작하기](#방식-b-운영-방식-바운드-키-인증)**

---

## 방식 A: 개발 전용 (인증 없음)

### 필수 준비물

- [ ] Python 환경 (Conda: `product_test`)
- [ ] `.env` 파일에 `DEBUG=true` 설정

### Step 1: DEBUG 모드 확인 (30초)

**.env 파일 확인**:
```bash
DEBUG=true  # 👈 이게 있어야 개발 API 활성화
```

만약 `.env` 파일이 없다면:
```bash
copy .env.example .env
# .env 파일 열어서 DEBUG=true 확인
```

### Step 2: 서버 실행 (1분)

**터미널 1**:
```bash
cd C:\Workspace\product_test_app
conda activate product_test
python main.py
```

**로그 확인** (중요!):
```
⚠️ [개발 모드] 인증 없는 개발 전용 API가 활성화되었습니다 (/api/v1/dev/*)
```
👆 **이 로그가 보여야 개발 API 사용 가능!**

### Step 3: 로컬 테스트 (30초)

**test_request.json 파일 생성**:
```json
{
  "consultation_id": "LOCAL_TEST_001",
  "stt_data": {
    "conversation_text": "상담사: 안녕하세요. 무엇을 도와드릴까요?\n고객: 보험 상품에 대해 문의드립니다.\n상담사: 건강보험 상품을 안내해드리겠습니다."
  }
}
```

**터미널 2**:
```bash
curl -X POST http://localhost:8000/api/v1/dev/realtime-analyze-no-auth \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

**예상 응답** (1-3초):
```json
{
  "success": true,
  "consultation_id": "LOCAL_TEST_001",
  "summary": "**고객**: 보험 상품에 대해 문의하였습니다.\n**상담사**: 건강보험 상품을 안내하였습니다.\n**상담결과**: 상담이 진행되었습니다.",
  "processing_time": 2.5,
  "model": "Qwen3-1.7B (개발 모드)"
}
```

### ✅ 방식 A 완료!

센터링크와 연동하려면 [ngrok으로 외부 노출](#ngrok으로-외부-노출-선택) 참고

---

## 방식 B: 운영 방식 (바운드 키 인증)

### 필수 준비물

- [ ] Python 환경 (Conda: `product_test`)
- [ ] ngrok 설치 ([다운로드](https://ngrok.com/download))
- [ ] 프로젝트 파일 (`C:\Workspace\product_test_app`)

### Step 1: AI 서버 실행

**터미널 1** (AI 분석 서버):
```bash
cd C:\Workspace\product_test_app
conda activate product_test
python main.py
```

### Step 2: AI 서버 외부 노출

**터미널 2** (ngrok - AI 서버):
```bash
ngrok http 8000
```

**생성된 URL 복사**:
```
https://abc-123-def.ngrok-free.app
```

### Step 3: 콜백 서버 실행 (배치 테스트용)

**터미널 3** (콜백 서버):
```bash
cd C:\Workspace\product_test_app
python centerlink_callback_server.py
```

**터미널 4** (ngrok - 콜백 서버):
```bash
ngrok http 5000
```

**생성된 URL 복사**:
```
https://xyz-456-ghi.ngrok-free.app
```

### Step 4: 자동 테스트 실행

**터미널 5** (테스트 스크립트):

1. `test_external_api.py` 파일 열기
2. 11번째 줄 수정:
   ```python
   BASE_URL = "https://abc-123-def.ngrok-free.app"  # Step 2에서 복사한 URL
   ```
3. 108번째 줄 수정:
   ```python
   "callback_url": "https://xyz-456-ghi.ngrok-free.app/api/ai-callback",  # Step 3에서 복사한 URL
   ```
4. 테스트 실행:
   ```bash
   cd C:\Workspace\product_test_app
   python test_external_api.py
   ```

### ✅ 방식 B 완료!

**예상 결과**:
```
================================================================================
테스트 결과 요약
================================================================================
헬스 체크            : ✅ 성공
실시간 API          : ✅ 성공
배치 API            : ✅ 성공
인증 실패            : ✅ 성공

총 4개 테스트 중 4개 성공 (100.0%)
```

---

## ngrok으로 외부 노출 (선택)

센터링크와 연동 테스트 시 필요합니다.

### Step 1: ngrok 실행

**터미널 3**:
```bash
ngrok http 8000
```

### Step 2: URL 확인 및 공유

```
Forwarding    https://abc-123-def.ngrok-free.app -> http://localhost:8000
```

### Step 3: 센터링크에게 전달

**방식 A (개발 전용)**:
```markdown
베이스 URL: https://abc-123-def.ngrok-free.app
엔드포인트: POST /api/v1/dev/realtime-analyze-no-auth
특징: 바운드 키 불필요, Content-Type 헤더만 필요

테스트:
curl -X POST https://abc-123-def.ngrok-free.app/api/v1/dev/realtime-analyze-no-auth \
  -H "Content-Type: application/json" \
  -d '{
    "consultation_id": "TEST_001",
    "stt_data": {"conversation_text": "..."}
  }'
```

**방식 B (운영 방식)**:
```markdown
베이스 URL: https://abc-123-def.ngrok-free.app
바운드 키: test_key_centerlink_2025
엔드포인트: POST /api/v1/consultation/realtime-analyze

테스트:
curl -X POST https://abc-123-def.ngrok-free.app/api/v1/consultation/realtime-analyze \
  -H "X-Bound-Key: test_key_centerlink_2025" \
  -H "Content-Type: application/json" \
  -d '{
    "bound_key": "test_key_centerlink_2025",
    "consultation_id": "TEST_001",
    "stt_data": {"conversation_text": "..."}
  }'
```

---

## 🔧 문제 해결

### DEBUG 모드 활성화 안됨 (방식 A)

**증상**: 개발 API 활성화 로그가 안 보임

**해결**:
```bash
# .env 파일 확인
DEBUG=true

# 서버 재시작
python main.py
```

### ngrok 실행 안됨

**해결**:
```bash
# 로그인 필요한 경우
ngrok authtoken YOUR_AUTH_TOKEN
```

### 서버 포트 충돌

**해결**:
```bash
# Windows: 포트 사용 프로세스 확인
netstat -ano | findstr :8000
netstat -ano | findstr :5000

# 프로세스 종료
taskkill /PID <PID번호> /F
```

### Python 환경 문제

**해결**:
```bash
# Conda 환경 재활성화
conda deactivate
conda activate product_test
python --version  # Python 3.10 이상 확인
```

---

## 📊 방식 비교

| 항목 | 방식 A (개발 전용) | 방식 B (운영 방식) |
|------|------------------|------------------|
| **소요 시간** | 2분 | 5분 |
| **인증** | 불필요 | 바운드 키 필수 |
| **테스트 범위** | 실시간 API만 | 실시간 + 배치 |
| **활성화 조건** | DEBUG=true | 항상 |
| **추천 용도** | 로컬 개발/디버깅 | 실제 연동 테스트 |

---

## 📚 다음 단계

- ✅ **방식 A 성공** → 운영 방식(방식 B)으로 전환 권장
- ✅ **방식 B 성공** → 실제 데이터로 통합 테스트 진행
- ❌ **테스트 실패** → [테스트 매뉴얼](./CENTERLINK_API_TEST_MANUAL.md) 참고
- 📖 **API 사용법** → [통합 가이드](./CENTERLINK_INTEGRATION_GUIDE.md) 참고
- 🔧 **개발 API 상세** → [개발 전용 API 가이드](./DEV_API_GUIDE.md) 참고

---

## 📞 지원

- **이메일**: ai-support@company.com
- **상세 문서**: [CENTERLINK_API_TEST_MANUAL.md](./CENTERLINK_API_TEST_MANUAL.md)
