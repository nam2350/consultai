# 실시간 상담 데이터 분석 시스템

## 프로젝트 개요
FastAPI 기반 실시간 상담 분석 시스템으로, JSON 파일을 즉시 분석하여 요약/분류/제목을 생성합니다.

## 기술 스택
- Python 3.11.13, FastAPI, Pydantic
- AI Models: Qwen3-4B, Kanana-1.5, Mi:dm-2.0
- conda env: product_test

## 주요 기능
1. 실시간 상담 분석 (DB 없이 메모리 처리)
2. 다중 AI 모델 동적 전환
3. 외부 API 연동 (센터링크)

## 디렉토리 구조
- src/core/: 설정, 로깅, 모니터링
- src/models/: AI 모델 시스템
- src/services/: 비즈니스 로직
- src/api/: FastAPI 라우터

## 핵심 API 엔드포인트
- POST /api/v1/consultation/analyze
- POST /api/v1/consultation/summarize
- POST /api/v1/models/switch
- GET /api/v1/models/status

## 개발 규칙
- 비동기 처리 우선 (async/await)
- Pydantic 스키마로 입출력 검증
- 타입 힌트 필수
- 구조화된 로깅 사용