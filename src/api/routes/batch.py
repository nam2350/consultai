"""
비동기 배치 처리 API (외부 시스템 연동용)

Batch 모델을 사용한 대량 상담 분석 (비동기 처리)
15분 단위 배치 처리를 위한 API
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from ...schemas.consultation import (
    AsyncBatchAnalysisRequest,
    AsyncBatchAnalysisResponse,
    BatchCallbackResult
)
from ...core.logger import logger
from ...core.batch_queue import get_batch_queue, BatchJob, BatchStatus
from ...core.batch_worker import BatchWorker, get_batch_worker, build_callback_payload
from ...core.config import get_application_settings
from ...core.url_utils import parse_allowed_hosts, validate_callback_url
from ...api.dependencies.auth import verify_batch_permission

# 라우터 생성
router = APIRouter(prefix="/consultation", tags=["비동기 배치 분석 (Batch)"])
settings = get_application_settings()


@router.post(
    "/batch-analyze-async",
    response_model=AsyncBatchAnalysisResponse,
    summary="비동기 배치 분석 (Batch)",
    description="""
    대량 상담 데이터를 비동기로 처리하는 API

    - **처리 방식**: 즉시 접수 후 백그라운드 처리
    - **사용 모델**: Batch (Qwen3-4B)
    - **배치 크기**: 최대 20개 통화
    - **완료 알림**: 콜백 URL로 결과 전송
    - **인증**: 바운드 키 필수 (batch 권한)
    """,
    status_code=status.HTTP_202_ACCEPTED
)
async def batch_analyze_async(
    request: AsyncBatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    key_info: Dict[str, Any] = Depends(verify_batch_permission)
) -> AsyncBatchAnalysisResponse:
    """
    비동기 배치 분석 API

    외부 시스템에서 15분 단위로 약 20개 통화를 배치로 전송하면
    즉시 접수 후 백그라운드에서 처리하고, 완료 시 콜백으로 결과 전송

    Args:
        request: 비동기 배치 분석 요청
            - bound_key: 바운드 키
            - batch_id: 배치 ID
            - consultations: 상담 데이터 목록
            - callback_url: 결과 전송 URL
            - batch_model: Batch 모델 선택
            - priority: 우선순위

    Returns:
        AsyncBatchAnalysisResponse: 접수 확인 응답 (즉시 반환)
    """
    try:
        logger.info(
            f"[배치 API] 비동기 배치 요청 수신 - "
            f"ID: {request.batch_id}, 통화수: {len(request.consultations)}, "
            f"키: {key_info['name']}"
        )

        # 1. 배치 크기 검증
        if len(request.consultations) > 20:
            return AsyncBatchAnalysisResponse(
                success=False,
                batch_id=request.batch_id,
                status=BatchStatus.FAILED.value,
                total_count=len(request.consultations),
                estimated_completion_time=None,
                callback_url=request.callback_url,
                error="배치 크기 초과 (최대 20개)",
                error_code="BATCH_SIZE_EXCEEDED"
            )

        if len(request.consultations) == 0:
            return AsyncBatchAnalysisResponse(
                success=False,
                batch_id=request.batch_id,
                status=BatchStatus.FAILED.value,
                total_count=0,
                estimated_completion_time=None,
                callback_url=request.callback_url,
                error="상담 데이터가 없습니다",
                error_code="BATCH_EMPTY"
            )

        allowed_hosts = parse_allowed_hosts(settings.BATCH_CALLBACK_ALLOWED_HOSTS)
        valid_callback, reason = validate_callback_url(
            request.callback_url,
            allowed_hosts,
            settings.BATCH_CALLBACK_BLOCK_PRIVATE_IPS,
        )
        if not valid_callback:
            return AsyncBatchAnalysisResponse(
                success=False,
                batch_id=request.batch_id,
                status=BatchStatus.FAILED.value,
                total_count=len(request.consultations),
                estimated_completion_time=None,
                callback_url=request.callback_url,
                error=f"콜백 URL이 허용되지 않습니다: {reason}",
                error_code="BATCH_CALLBACK_URL_INVALID",
            )

        # 3. 배치 작업 생성
        batch_job = BatchJob(
            batch_id=request.batch_id,
            consultations=request.consultations,
            callback_url=request.callback_url,
            batch_model=request.batch_model or "qwen3_4b",
            priority=request.priority or 1,
            bound_key=request.bound_key
        )

        # 4. 큐에 추가
        queue = get_batch_queue()
        success = await queue.add_job(batch_job)

        if not success:
            return AsyncBatchAnalysisResponse(
                success=False,
                batch_id=request.batch_id,
                status=BatchStatus.FAILED.value,
                total_count=len(request.consultations),
                estimated_completion_time=None,
                callback_url=request.callback_url,
                error="중복된 배치 ID입니다",
                error_code="BATCH_DUPLICATE_ID"
            )

        # 5. 예상 완료 시간 계산 (Batch 평균 처리 시간 기준)
        # Qwen3-4B: 7.85초/통화
        avg_time_per_call = 7.85
        estimated_time = avg_time_per_call * len(request.consultations)

        # 6. 성공 응답 (즉시 반환)
        logger.info(
            f"[배치 API] 배치 작업 큐 추가 완료 - "
            f"ID: {request.batch_id}, 예상시간: {estimated_time:.1f}초"
        )

        return AsyncBatchAnalysisResponse(
            success=True,
            batch_id=request.batch_id,
            status=BatchStatus.QUEUED.value,
            total_count=len(request.consultations),
            estimated_completion_time=estimated_time,
            callback_url=request.callback_url,
            error=None,
            error_code=None
        )

    except HTTPException:
        # FastAPI HTTPException은 그대로 전파
        raise

    except Exception as e:
        logger.error(f"[배치 API] 예상치 못한 오류: {e}", exc_info=True)

        return AsyncBatchAnalysisResponse(
            success=False,
            batch_id=request.batch_id,
            status=BatchStatus.FAILED.value,
            total_count=len(request.consultations) if request.consultations else 0,
            estimated_completion_time=None,
            callback_url=request.callback_url,
            error=f"서버 내부 오류: {str(e)}",
            error_code="SERVER_INTERNAL_ERROR"
        )


@router.get(
    "/batch-status/{batch_id}",
    summary="배치 작업 상태 조회",
    description="배치 ID로 작업 상태 및 진행률 조회"
)
async def get_batch_status(
    batch_id: str,
    key_info: Dict[str, Any] = Depends(verify_batch_permission)
):
    """
    배치 작업 상태 조회 API

    Args:
        batch_id: 배치 ID
        key_info: 인증 정보

    Returns:
        배치 작업 상태 정보
    """
    try:
        logger.info(f"[배치 API] 배치 상태 조회 - ID: {batch_id}")

        # 큐에서 작업 조회
        queue = get_batch_queue()
        job_status = await queue.get_job_status(batch_id)

        if not job_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "배치 작업을 찾을 수 없습니다",
                    "error_code": "BATCH_NOT_FOUND",
                    "batch_id": batch_id
                }
            )

        return {
            "success": True,
            "batch_id": batch_id,
            **job_status
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"[배치 API] 상태 조회 오류: {e}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"서버 내부 오류: {str(e)}",
                "error_code": "SERVER_INTERNAL_ERROR"
            }
        )


@router.delete(
    "/batch-cancel/{batch_id}",
    summary="배치 작업 취소",
    description="대기 중인 배치 작업 취소 (처리 중/완료된 작업은 취소 불가)"
)
async def cancel_batch(
    batch_id: str,
    key_info: Dict[str, Any] = Depends(verify_batch_permission)
):
    """
    배치 작업 취소 API

    Args:
        batch_id: 배치 ID
        key_info: 인증 정보

    Returns:
        취소 결과
    """
    try:
        logger.info(f"[배치 API] 배치 취소 요청 - ID: {batch_id}")

        # 큐에서 작업 취소
        queue = get_batch_queue()
        success = await queue.cancel_job(batch_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "배치 작업을 취소할 수 없습니다 (이미 처리 중이거나 완료됨)",
                    "error_code": "BATCH_CANCEL_FAILED",
                    "batch_id": batch_id
                }
            )

        return {
            "success": True,
            "batch_id": batch_id,
            "message": "배치 작업이 취소되었습니다"
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"[배치 API] 취소 오류: {e}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"서버 내부 오류: {str(e)}",
                "error_code": "SERVER_INTERNAL_ERROR"
            }
        )


@router.post(
    "/batch-callback-retry/{batch_id}",
    summary="배치 콜백 재전송",
    description="콜백 실패 시 수동으로 재전송을 요청합니다."
)
async def retry_batch_callback(
    batch_id: str,
    key_info: Dict[str, Any] = Depends(verify_batch_permission)
):
    """
    배치 콜백 재전송 API

    Args:
        batch_id: 배치 ID
        key_info: 인증 정보

    Returns:
        콜백 재전송 결과
    """
    try:
        logger.info(f"[배치 API] 콜백 재전송 요청 - ID: {batch_id}")

        queue = get_batch_queue()
        job = await queue.get_job(batch_id)

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "배치 작업을 찾을 수 없습니다",
                    "error_code": "BATCH_NOT_FOUND",
                    "batch_id": batch_id
                }
            )

        if job.status not in (BatchStatus.COMPLETED, BatchStatus.FAILED):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": "배치 작업이 아직 완료되지 않았습니다",
                    "error_code": "BATCH_NOT_READY",
                    "batch_id": batch_id,
                    "status": job.status.value,
                }
            )

        callback_data = build_callback_payload(job)
        worker = get_batch_worker() or BatchWorker()
        callback_success = await worker.send_callback(job.callback_url, callback_data)
        refreshed = await queue.get_job(batch_id)

        return {
            "success": callback_success,
            "batch_id": batch_id,
            "callback_url": job.callback_url,
            "callback_status": refreshed.callback_status if refreshed else None,
            "callback_attempts": refreshed.callback_attempts if refreshed else None,
            "callback_last_error": refreshed.callback_last_error if refreshed else None,
            "callback_errors": refreshed.callback_errors if refreshed else None,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"[배치 API] 콜백 재전송 오류: {e}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"서버 내부 오류: {str(e)}",
                "error_code": "SERVER_INTERNAL_ERROR"
            }
        )


@router.post(
    "/batch-recover-stale",
    summary="배치 중단 복구",
    description="오래된 processing 상태를 재큐잉/실패 처리합니다."
)
async def recover_stale_batches(
    key_info: Dict[str, Any] = Depends(verify_batch_permission)
):
    """
    배치 중단 복구 API

    Returns:
        복구 처리 결과
    """
    try:
        logger.info("[배치 API] 배치 중단 복구 요청")

        queue = get_batch_queue()
        stats = await queue.recover_stale_jobs()

        return {
            "success": True,
            **stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"[배치 API] 배치 중단 복구 오류: {e}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"서버 내부 오류: {str(e)}",
                "error_code": "SERVER_INTERNAL_ERROR"
            }
        )


@router.get(
    "/batch-queue-stats",
    summary="배치 큐 통계",
    description="배치 큐의 전체 통계 정보 조회 (관리자용)"
)
async def get_queue_stats(
    key_info: Dict[str, Any] = Depends(verify_batch_permission)
):
    """
    배치 큐 통계 조회 API

    Returns:
        큐 통계 정보
    """
    try:
        queue = get_batch_queue()
        stats = await queue.get_queue_stats()

        return {
            "success": True,
            **stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"[배치 API] 통계 조회 오류: {e}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"서버 내부 오류: {str(e)}",
                "error_code": "SERVER_INTERNAL_ERROR"
            }
        )
