"""
개발 전용 API (인증 없음)

⚠️ 경고: 이 엔드포인트는 개발/테스트 전용입니다.
- DEBUG 모드에서만 활성화됩니다
- 운영 환경에서는 절대 사용하지 마세요
- 바운드 키 인증이 필요 없습니다

Author: AI 분석팀
Date: 2025-10-16
"""

import time
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from ...schemas.consultation import (
    RealtimeAnalysisRequest,
    RealtimeAnalysisResponse,
)
from ...core.logger import logger
from ...core.file_processor import extract_conversation_text

# 라우터 생성
router = APIRouter(prefix="/dev", tags=["개발 전용 (인증 없음)"])

# SLM 서비스 인스턴스 (지연 초기화)
_slm_service = None


async def get_slm_service():
    """SLM 서비스 인스턴스 반환 (지연 초기화)"""
    global _slm_service

    if _slm_service is None:
        logger.info("[개발 API] SLM 서비스 초기화 시작...")

        # Qwen3-1.7B 모델 로드
        from ...core.models.qwen3_1_7b.summarizer import Qwen17BSummarizer

        try:
            _slm_service = Qwen17BSummarizer(
                model_path=r"models\Qwen3-1.7B"
            )

            success = _slm_service.load_model()

            if not success:
                logger.error("[개발 API] SLM 모델 로드 실패")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="SLM 모델 초기화 실패"
                )

            logger.info(f"[개발 API] SLM 모델 로드 완료: Qwen3-1.7B")

        except Exception as e:
            logger.error(f"[개발 API] SLM 서비스 초기화 오류: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"SLM 서비스 초기화 실패: {str(e)}"
            )

    return _slm_service


@router.post(
    "/realtime-analyze-no-auth",
    response_model=RealtimeAnalysisResponse,
    summary="실시간 상담 분석 (인증 없음 - 개발 전용)",
    description="""
    ⚠️ **개발/테스트 전용 엔드포인트**

    - **인증 불필요**: 바운드 키 없이 사용 가능
    - **목표 응답시간**: 1-3초
    - **사용 모델**: Qwen3-1.7B (SLM)
    - **제공 기능**: 간략 요약 (3줄 구조)
    - **주의**: DEBUG 모드에서만 활성화됩니다

    **운영 환경에서는 절대 사용하지 마세요!**
    """
)
async def dev_realtime_analyze(
    request: RealtimeAnalysisRequest
) -> RealtimeAnalysisResponse:
    """
    개발 전용 - 실시간 상담 분석 API (인증 없음)

    Args:
        request: 실시간 분석 요청
            - consultation_id: 상담 ID
            - stt_data: STT 변환 데이터

    Returns:
        RealtimeAnalysisResponse: 간략 요약 결과 (1-3초 이내)
    """
    start_time = time.time()

    logger.warning(
        f"[개발 API] ⚠️ 인증 없는 요청 수신 - ID: {request.consultation_id} "
        "(개발 전용 엔드포인트 사용 중)"
    )

    try:
        # 1. STT 데이터 추출
        try:
            conversation_text = extract_conversation_text(request.stt_data.dict())
            if not conversation_text or len(conversation_text.strip()) < 10:
                raise ValueError("대화 내용이 너무 짧거나 없습니다")
        except Exception as e:
            logger.error(f"[개발 API] STT 데이터 추출 실패: {e}")
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

        # 2. SLM 서비스 로드
        slm_service = await get_slm_service()

        # 3. 빠른 요약 생성
        logger.info(f"[개발 API] SLM 요약 시작 - 대화 길이: {len(conversation_text)}자")
        result = slm_service.summarize_consultation(conversation_text)

        # 4. 결과 검증
        if not result.get('success'):
            error_msg = result.get('error', '알 수 없는 오류')
            logger.error(f"[개발 API] 요약 생성 실패: {error_msg}")
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

        # 5. 성공 응답
        processing_time = time.time() - start_time
        summary = result.get('summary', '')

        logger.info(
            f"[개발 API] 요약 완료 - ID: {request.consultation_id}, "
            f"처리시간: {processing_time:.2f}초, 요약길이: {len(summary)}자"
        )

        return RealtimeAnalysisResponse(
            success=True,
            consultation_id=request.consultation_id,
            summary=summary,
            processing_time=processing_time,
            model="Qwen3-1.7B (개발 모드)",
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
        logger.error(f"[개발 API] 예상치 못한 오류: {e}", exc_info=True)

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
    "/status",
    summary="개발 전용 API 상태",
    description="개발 전용 엔드포인트 활성화 상태 및 모델 정보 조회"
)
async def get_dev_status():
    """개발 전용 API 상태 조회"""
    global _slm_service

    return {
        "status": "active",
        "warning": "⚠️ 이 엔드포인트는 개발/테스트 전용입니다. 운영 환경에서는 사용하지 마세요.",
        "authentication": "disabled",
        "model_loaded": _slm_service is not None,
        "model_name": "Qwen3-1.7B" if _slm_service else None,
        "endpoints": {
            "realtime_no_auth": "/api/v1/dev/realtime-analyze-no-auth"
        },
        "usage_note": "바운드 키 없이 API를 호출할 수 있습니다 (테스트용)"
    }


@router.get(
    "/test",
    summary="개발 전용 테스트",
    description="개발 전용 엔드포인트 동작 확인"
)
async def dev_test():
    """개발 전용 테스트 엔드포인트"""
    return {
        "message": "개발 전용 API가 정상 작동 중입니다",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "warning": "⚠️ 이 엔드포인트는 개발/테스트 전용입니다",
        "authentication": "disabled"
    }
