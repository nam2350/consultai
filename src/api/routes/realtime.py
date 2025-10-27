"""
실시간 상담 분석 API (센터링크 연동용)

SLM 모델을 사용한 빠른 상담 요약 제공 (1-3초 목표)
상담사 실시간 지원을 위한 경량 분석 API
"""

import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Header, status, Depends
from fastapi.responses import JSONResponse

from ...schemas.consultation import (
    RealtimeAnalysisRequest,
    RealtimeAnalysisResponse,
    ErrorResponse,
    ErrorDetail
)
from ...core.logger import logger
from ...core.file_processor import extract_conversation_text
from ...api.dependencies.auth import verify_bound_key, verify_realtime_permission

# 라우터 생성
router = APIRouter(prefix="/consultation", tags=["실시간 상담 분석 (SLM)"])

# SLM 서비스 인스턴스 (지연 초기화)
_slm_service = None
_slm_model_type = "qwen3_1_7b"  # 기본: Qwen3-1.7B (2.83초, 0.800 품질)


async def get_slm_service():
    """SLM 서비스 인스턴스 반환 (지연 초기화)"""
    global _slm_service

    if _slm_service is None:
        logger.info("[실시간 API] SLM 서비스 초기화 시작...")

        # Qwen3-1.7B 모델 로드
        from ...core.models.qwen3_1_7b.summarizer import Qwen17BSummarizer

        try:
            _slm_service = Qwen17BSummarizer(
                model_path=r"models\Qwen3-1.7B"
            )

            success = _slm_service.load_model()

            if not success:
                logger.error("[실시간 API] SLM 모델 로드 실패")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="SLM 모델 초기화 실패"
                )

            logger.info(f"[실시간 API] SLM 모델 로드 완료: Qwen3-1.7B")

        except Exception as e:
            logger.error(f"[실시간 API] SLM 서비스 초기화 오류: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"SLM 서비스 초기화 실패: {str(e)}"
            )

    return _slm_service


@router.post(
    "/realtime-analyze",
    response_model=RealtimeAnalysisResponse,
    summary="실시간 상담 분석 (SLM)",
    description="""
    상담사 실시간 지원을 위한 빠른 상담 요약 API

    - **목표 응답시간**: 1-3초
    - **사용 모델**: Qwen3-1.7B (SLM)
    - **제공 기능**: 간략 요약 (3줄 구조)
    - **인증**: 바운드 키 필수 (X-Bound-Key 헤더 또는 Authorization 헤더)
    - **권한**: realtime 권한 필요
    """
)
async def realtime_analyze(
    request: RealtimeAnalysisRequest,
    key_info: Dict[str, Any] = Depends(verify_bound_key)
) -> RealtimeAnalysisResponse:
    """
    실시간 상담 분석 API

    상담사가 상담 중 또는 직후 빠르게 요약을 받기 위한 API

    Args:
        request: 실시간 분석 요청
            - bound_key: 센터링크 바운드 키
            - consultation_id: 상담 ID
            - stt_data: STT 변환 데이터

    Returns:
        RealtimeAnalysisResponse: 간략 요약 결과 (1-3초 이내)
    """
    start_time = time.time()

    try:
        logger.info(f"[실시간 API] 요청 수신 - ID: {request.consultation_id}, 키: {key_info['name']}")

        # 1. 권한 확인
        permissions = key_info.get("permissions", [])
        if "realtime" not in permissions:
            logger.warning(f"[실시간 API] 실시간 처리 권한 없음: {key_info['name']}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "실시간 처리 권한이 없습니다",
                    "error_code": "AUTH_PERMISSION_DENIED"
                }
            )

        # 2. STT 데이터 추출
        try:
            conversation_text = extract_conversation_text(request.stt_data.dict())
            if not conversation_text or len(conversation_text.strip()) < 10:
                raise ValueError("대화 내용이 너무 짧거나 없습니다")
        except Exception as e:
            logger.error(f"[실시간 API] STT 데이터 추출 실패: {e}")
            return RealtimeAnalysisResponse(
                success=False,
                consultation_id=request.consultation_id,
                summary=None,
                processing_time=time.time() - start_time,
                model="Qwen3-1.7B",
                timestamp=datetime.now(timezone.utc).isoformat(),
                error=f"STT 데이터 처리 실패: {str(e)}",
                error_code="DATA_INVALID_STT"
            )

        # 3. SLM 서비스 로드
        slm_service = await get_slm_service()

        # 4. 빠른 요약 생성
        logger.info(f"[실시간 API] SLM 요약 시작 - 대화 길이: {len(conversation_text)}자")
        result = slm_service.summarize_consultation(conversation_text)

        # 5. 결과 검증
        if not result.get('success'):
            error_msg = result.get('error', '알 수 없는 오류')
            logger.error(f"[실시간 API] 요약 생성 실패: {error_msg}")
            return RealtimeAnalysisResponse(
                success=False,
                consultation_id=request.consultation_id,
                summary=None,
                processing_time=time.time() - start_time,
                model="Qwen3-1.7B",
                timestamp=datetime.now(timezone.utc).isoformat(),
                error=error_msg,
                error_code="AI_SUMMARIZATION_FAILED"
            )

        # 6. 성공 응답
        processing_time = time.time() - start_time
        summary = result.get('summary', '')

        logger.info(
            f"[실시간 API] 요약 완료 - ID: {request.consultation_id}, "
            f"처리시간: {processing_time:.2f}초, 요약길이: {len(summary)}자"
        )

        return RealtimeAnalysisResponse(
            success=True,
            consultation_id=request.consultation_id,
            summary=summary,
            processing_time=processing_time,
            model="Qwen3-1.7B",
            timestamp=datetime.now(timezone.utc).isoformat(),
            error=None,
            error_code=None
        )

    except HTTPException:
        # FastAPI HTTPException은 그대로 전파
        raise

    except Exception as e:
        # 예상하지 못한 오류
        processing_time = time.time() - start_time
        logger.error(f"[실시간 API] 예상치 못한 오류: {e}", exc_info=True)

        return RealtimeAnalysisResponse(
            success=False,
            consultation_id=request.consultation_id,
            summary=None,
            processing_time=processing_time,
            model="Qwen3-1.7B",
            timestamp=datetime.now(timezone.utc).isoformat(),
            error=f"서버 내부 오류: {str(e)}",
            error_code="SERVER_INTERNAL_ERROR"
        )


@router.get(
    "/realtime-status",
    summary="실시간 처리 시스템 상태",
    description="SLM 모델 로드 상태 및 처리 통계 조회"
)
async def get_realtime_status():
    """실시간 처리 시스템 상태 조회"""
    global _slm_service

    return {
        "status": "healthy" if _slm_service is not None else "not_initialized",
        "model_loaded": _slm_service is not None,
        "model_name": "Qwen3-1.7B" if _slm_service else None,
        "model_type": "SLM (Small Language Model)",
        "target_response_time": "1-3초",
        "average_response_time": "2.83초 (검증 완료)"
    }
