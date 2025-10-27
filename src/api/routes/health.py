from fastapi import APIRouter
from datetime import datetime
from src.core.logger import logger
from src.core.config import settings

router = APIRouter()

@router.get("/")
async def health_check():
    """기본 헬스 체크"""
    return {
        "status": "healthy",
        "service": "상담 데이터 분석 시스템",
        "version": settings.VERSION
    }

@router.get("/detailed")
async def detailed_health_check():
    """상세 헬스 체크"""
    try:
        return {
            "status": "healthy",
            "service": "상담 데이터 분석 시스템",
            "version": settings.VERSION,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"상세 헬스 체크 실패: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "상담 데이터 분석 시스템",
            "version": settings.VERSION,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/ready")
async def readiness_check():
    """서비스 준비 상태 확인"""
    try:
        return {
            "status": "ready",
            "message": "서비스가 요청을 처리할 준비가 되었습니다."
        }
    except Exception as e:
        logger.error(f"준비 상태 확인 실패: {str(e)}")
        return {
            "status": "not_ready",
            "message": "서비스가 요청을 처리할 준비가 되지 않았습니다.",
            "error": str(e)
        }

@router.get("/live")
async def liveness_check():
    """서비스 생존 상태 확인"""
    return {
        "status": "alive",
        "message": "서비스가 정상적으로 실행 중입니다."
    } 