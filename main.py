import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 핵심 모듈만 import
from src.core.config import get_application_settings
from src.core.logger import logger
from src.core.exceptions import register_application_exception_handlers
from src.api.routes import health

# 환경변수 로드
load_dotenv()

# 애플리케이션 설정 로드
settings = get_application_settings()

# 상담 분석 라우터와 cleanup 함수 import
from src.api.routes.consultation import router as consultation_router, cleanup_consultation_service
from src.api.routes.realtime import router as realtime_router
from src.api.routes.batch import router as batch_router

# 배치 워커 import
from src.core.batch_worker import start_batch_worker, stop_batch_worker

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # 시작 시 초기화 작업
    logger.info("[라이프사이클] 애플리케이션 시작 - 배치 워커 초기화 중...")
    await start_batch_worker()
    logger.info("[라이프사이클] 배치 워커 시작 완료")

    yield

    # 종료 시 정리 작업
    logger.info("[라이프사이클] 애플리케이션 종료 - 리소스 정리 중...")
    await stop_batch_worker()
    await cleanup_consultation_service()
    logger.info("[라이프사이클] 모든 리소스 정리 완료")

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="상담 통화 요약 시스템",
    description="AI 기반 상담 통화 어시스턴트 시스템",
    version="1.0.0",
    lifespan=lifespan,
)

# UTF-8 인코딩을 위한 설정
from fastapi import Response
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import json

# JSON 응답에서 한글이 제대로 표시되도록 설정
class UTF8JSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            indent=2,
            separators=(',', ': ')
        ).encode('utf-8')

# 기본 응답 클래스를 UTF8JSONResponse로 설정
app.default_response_class = UTF8JSONResponse

@app.middleware("http")
async def ensure_utf8_encoding(request, call_next):
    response = await call_next(request)

    # Content-Type에 charset=utf-8 명시적으로 설정
    if "application/json" in response.headers.get("content-type", ""):
        response.headers["content-type"] = "application/json; charset=utf-8"
    elif "text/html" in response.headers.get("content-type", ""):
        response.headers["content-type"] = "text/html; charset=utf-8"
    elif "text/plain" in response.headers.get("content-type", ""):
        response.headers["content-type"] = "text/plain; charset=utf-8"

    return response

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 예외 핸들러 등록
register_application_exception_handlers(app)

# 정적 파일 서빙
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"정적 파일 마운트 실패: {e}")

# 라우터 등록
app.include_router(health.router, prefix="/api/v1")

# 상담 분석 라우터 등록 (LLM 배치 처리)
app.include_router(consultation_router, prefix="/api/v1")

# 실시간 상담 분석 라우터 등록 (SLM 실시간 처리 - 외부 시스템 연동용)
app.include_router(realtime_router, prefix="/api/v1")

# 비동기 배치 처리 라우터 등록 (LLM 배치 처리 - 외부 시스템 연동용)
app.include_router(batch_router, prefix="/api/v1")

# 개발 전용 라우터 등록 (인증 없음 - DEBUG 모드에서만)
if settings.DEBUG:
    from src.api.routes.dev import router as dev_router
    app.include_router(dev_router, prefix="/api/v1")
    logger.warning("⚠️ [개발 모드] 인증 없는 개발 전용 API가 활성화되었습니다 (/api/v1/dev/*)")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    endpoints = {
        # 시스템 상태
        "health": "/api/v1/health",

        # 실시간 처리 (SLM - 외부 시스템 연동용)
        "realtime_analyze": "/api/v1/consultation/realtime-analyze",
        "realtime_status": "/api/v1/consultation/realtime-status",

        # 비동기 배치 처리 (LLM - 외부 시스템 연동용)
        "batch_analyze_async": "/api/v1/consultation/batch-analyze-async",
        "batch_status": "/api/v1/consultation/batch-status/{batch_id}",
        "batch_cancel": "/api/v1/consultation/batch-cancel/{batch_id}",
        "batch_queue_stats": "/api/v1/consultation/batch-queue-stats",

        # 기존 동기 배치 처리 (LLM)
        "consultation_analyze": "/api/v1/consultation/analyze",
        "consultation_batch": "/api/v1/consultation/batch-analyze",
        "consultation_status": "/api/v1/consultation/status",

        # 외부 시스템 호환 연동
        "external_analyze": "/api/v1/consultation/external/analyze"
    }

    # DEBUG 모드일 때만 개발 전용 엔드포인트 추가
    if settings.DEBUG:
        endpoints["dev_realtime_no_auth"] = "/api/v1/dev/realtime-analyze-no-auth (개발 전용)"
        endpoints["dev_status"] = "/api/v1/dev/status (개발 전용)"
        endpoints["dev_test"] = "/api/v1/dev/test (개발 전용)"

    return {
        "message": "AI 상담 분석 시스템 (외부 시스템 연동)",
        "version": "1.0.0",
        "status": "running",
        "debug_mode": settings.DEBUG,
        "docs_url": "/docs",
        "api_endpoints": endpoints
    }

if __name__ == "__main__":
    logger.info(f"서버 시작: http://{settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )