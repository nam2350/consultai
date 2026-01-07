"""
배치 워커 시스템

배치 큐에서 작업을 가져와 배치용 모델로 처리하고 콜백 전송
"""

import asyncio
import time
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .logger import logger
from .batch_queue import get_batch_queue, BatchJob, BatchStatus
from .file_processor import extract_conversation_text
from .config import get_application_settings
from .url_utils import parse_allowed_hosts, validate_callback_url

settings = get_application_settings()


def build_callback_payload(
    job: BatchJob,
    results: Optional[List[Dict[str, Any]]] = None,
    processing_time: Optional[float] = None,
    status_override: Optional[str] = None,
    completed_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    resolved_results = results if results is not None else job.results
    success_count = sum(1 for r in resolved_results if r.get("success"))
    failed_count = len(resolved_results) - success_count
    status_value = (
        status_override
        if status_override is not None
        else (job.status.value if isinstance(job.status, BatchStatus) else str(job.status))
    )
    completed_at_value = completed_at or job.completed_at or datetime.now(timezone.utc)

    if processing_time is None:
        if job.started_at and completed_at_value:
            processing_time = (completed_at_value - job.started_at).total_seconds()
        else:
            processing_time = 0.0

    return {
        "batch_id": job.batch_id,
        "status": status_value,
        "total_count": job.total_count,
        "success_count": success_count,
        "failed_count": failed_count,
        "results": resolved_results,
        "total_processing_time": processing_time,
        "completed_at": completed_at_value.isoformat(),
    }


class BatchWorker:
    """배치 작업 처리 워커"""

    def __init__(self):
        """초기화"""
        self.is_running = False
        self.queue = get_batch_queue()
        self.batch_service = None  # 배치용 서비스 (지연 로드)

        # 콜백 설정
        self.callback_timeout = settings.BATCH_CALLBACK_TIMEOUT_SECONDS
        self.callback_retry_count = settings.BATCH_CALLBACK_RETRY_COUNT
        self.callback_retry_interval = settings.BATCH_CALLBACK_RETRY_INTERVAL_SECONDS
        self._callback_allowed_hosts = parse_allowed_hosts(settings.BATCH_CALLBACK_ALLOWED_HOSTS)
        self._callback_block_private = settings.BATCH_CALLBACK_BLOCK_PRIVATE_IPS

        logger.info("[배치워커] 배치 워커 초기화 완료")

    async def load_batch_service(self, model_name: str = "qwen3_4b") -> bool:
        """
        배치 분석 서비스 로드 (지연 초기화)

        Args:
            model_name: 모델 이름 (qwen3_4b)

        Returns:
            True if service loaded successfully, False otherwise
        """
        if self.batch_service is not None:
            # 이미 로드된 경우 정리
            if hasattr(self.batch_service, 'cleanup'):
                await asyncio.to_thread(self.batch_service.cleanup)

        logger.info(f"[배치워커] 배치 분석 서비스 로드 시작: {model_name}")

        try:
            if model_name == "qwen3_4b":
                from ..services.consultation_service import ConsultationService
                model_path = settings.BATCH_MODEL_PATH_QWEN3_4B or settings.MODEL_PATH
                self.batch_service = ConsultationService(model_path=model_path)
                success = await asyncio.to_thread(self.batch_service.initialize)

            else:
                logger.error(f"[배치워커] 지원하지 않는 모델: {model_name}")
                return False

            if success:
                logger.info(f"[배치워커] 배치 분석 서비스 로드 완료: {model_name}")
                return True
            else:
                logger.error(f"[배치워커] 배치 분석 서비스 로드 실패: {model_name}")
                return False

        except Exception as e:
            logger.error(f"[배치워커] 배치 분석 서비스 로드 오류: {e}", exc_info=True)
            return False

    async def process_consultation(
        self,
        consultation: Dict[str, Any],
        model_name: str
    ) -> Dict[str, Any]:
        """
        개별 상담 처리

        Args:
            consultation: 상담 데이터
            model_name: 모델 이름

        Returns:
            처리 결과
        """
        consultation_id = consultation.get("consultation_id", "UNKNOWN")
        start_time = time.time()

        try:
            # 1. STT 데이터 추출
            stt_data = consultation.get("stt_data", {})
            conversation_text = extract_conversation_text(stt_data)

            if not conversation_text or len(conversation_text.strip()) < 10:
                return {
                    "consultation_id": consultation_id,
                    "success": False,
                    "error": "대화 내용이 없거나 너무 짧습니다",
                    "processing_time": time.time() - start_time
                }

            # 2. 배치 모델로 분석
            if model_name == "qwen3_4b":
                # ConsultationService 사용 (요약+키워드+제목)
                result = await asyncio.to_thread(
                    self.batch_service.analyze_consultation_text,
                    conversation_text,
                    {"batch_model": model_name},
                )

                if result.get('success'):
                    return {
                        "consultation_id": consultation_id,
                        "success": True,
                        "summary": result.get('summary', ''),
                        "categories": result.get('recommended_categories', []),
                        "titles": result.get('generated_titles', []),
                        "processing_time": time.time() - start_time,
                        "model": "Qwen3-4B-2507"
                    }
                else:
                    return {
                        "consultation_id": consultation_id,
                        "success": False,
                        "error": result.get('error', '알 수 없는 오류'),
                        "processing_time": time.time() - start_time
                    }

            else:
                return {
                    "consultation_id": consultation_id,
                    "success": False,
                    "error": f"지원하지 않는 모델: {model_name}",
                    "processing_time": time.time() - start_time
                }

        except Exception as e:
            logger.error(f"[배치워커] 상담 처리 오류 - {consultation_id}: {e}", exc_info=True)
            return {
                "consultation_id": consultation_id,
                "success": False,
                "error": f"처리 오류: {str(e)}",
                "processing_time": time.time() - start_time
            }

    async def send_callback(
        self,
        callback_url: str,
        result_data: Dict[str, Any]
    ) -> bool:
        """
        센터링크로 콜백 전송

        Args:
            callback_url: 콜백 URL
            result_data: 전송할 결과 데이터

        Returns:
            전송 성공 여부
        """
        batch_id = result_data.get("batch_id")
        valid_callback, reason = validate_callback_url(
            callback_url,
            self._callback_allowed_hosts,
            self._callback_block_private,
        )
        if not valid_callback:
            logger.error(
                "[배치워커] 콜백 URL 차단됨: %s (reason=%s)",
                callback_url,
                reason,
            )
            if batch_id:
                await self.queue.update_callback_status(
                    batch_id,
                    "failed",
                    0,
                    f"Callback URL blocked: {reason}",
                )
            return False

        for attempt in range(1, self.callback_retry_count + 1):
            try:
                logger.info(
                    f"[배치워커] 콜백 전송 시도 {attempt}/{self.callback_retry_count} "
                    f"- URL: {callback_url}"
                )

                async with httpx.AsyncClient(timeout=self.callback_timeout) as client:
                    response = await client.post(
                        callback_url,
                        json=result_data,
                        headers={"Content-Type": "application/json"}
                    )

                    if response.status_code == 200:
                        logger.info(f"[배치워커] 콜백 전송 성공 - Batch: {result_data['batch_id']}")
                        if batch_id:
                            await self.queue.update_callback_status(
                                batch_id,
                                "success",
                                attempt,
                                None,
                            )
                        return True
                    else:
                        error_message = (
                            f"HTTP {response.status_code}: {response.text[:200]}"
                        )
                        logger.warning(
                            f"[배치워커] 콜백 응답 오류 - 상태코드: {response.status_code}, "
                            f"응답: {response.text[:200]}"
                        )

            except httpx.TimeoutException:
                error_message = "Timeout while sending callback"
                logger.warning(f"[배치워커] 콜백 타임아웃 - 시도 {attempt}/{self.callback_retry_count}")

            except Exception as e:
                error_message = f"Callback error: {e}"
                logger.error(f"[배치워커] 콜백 전송 오류 - 시도 {attempt}/{self.callback_retry_count}: {e}")

            if batch_id:
                await self.queue.update_callback_status(
                    batch_id,
                    "retrying" if attempt < self.callback_retry_count else "failed",
                    attempt,
                    error_message,
                )

            # 재시도 대기
            if attempt < self.callback_retry_count:
                await asyncio.sleep(self.callback_retry_interval)

        logger.error(f"[배치워커] 콜백 전송 최종 실패 - Batch: {result_data['batch_id']}")
        return False

    async def process_batch(self, job: BatchJob):
        """
        배치 작업 처리

        Args:
            job: 배치 작업 객체
        """
        batch_id = job.batch_id
        logger.info(
            f"[배치워커] 배치 처리 시작 - ID: {batch_id}, "
            f"통화수: {job.total_count}, 모델: {job.batch_model}"
        )

        batch_start_time = time.time()

        try:
            # 1. 배치 분석 서비스 로드
            success = await self.load_batch_service(job.batch_model)
            if not success:
                await self.queue.update_job_status(
                    batch_id,
                    BatchStatus.FAILED,
                    error="배치 분석 서비스 로드 실패"
                )
                return

            # 2. 각 상담 처리
            results = []
            for idx, consultation in enumerate(job.consultations, 1):
                logger.info(f"[배치워커] 처리 중 [{idx}/{job.total_count}] - {consultation.get('consultation_id')}")

                result = await self.process_consultation(consultation, job.batch_model)
                results.append(result)

            # 3. 배치 상태 업데이트
            await self.queue.update_job_status(
                batch_id,
                BatchStatus.COMPLETED,
                results=results
            )

            # 4. 콜백 데이터 준비
            callback_data = build_callback_payload(
                job,
                results=results,
                processing_time=time.time() - batch_start_time,
                status_override=BatchStatus.COMPLETED.value,
                completed_at=datetime.now(timezone.utc),
            )

            # 5. 콜백 전송
            callback_success = await self.send_callback(job.callback_url, callback_data)

            if callback_success:
                logger.info(
                    f"[배치워커] 배치 완료 - ID: {batch_id}, "
                    f"성공: {callback_data['success_count']}/{job.total_count}, "
                    f"처리시간: {callback_data['total_processing_time']:.1f}초"
                )
            else:
                logger.warning(
                    f"[배치워커] 배치 처리는 완료되었으나 콜백 전송 실패 - ID: {batch_id}"
                )

        except Exception as e:
            logger.error(f"[배치워커] 배치 처리 오류 - {batch_id}: {e}", exc_info=True)
            await self.queue.update_job_status(
                batch_id,
                BatchStatus.FAILED,
                error=str(e)
            )

        finally:
            # 배치 분석 서비스 정리
            if self.batch_service and hasattr(self.batch_service, 'cleanup'):
                await asyncio.to_thread(self.batch_service.cleanup)
                self.batch_service = None

    async def run(self):
        """
        워커 실행 (무한 루프)
        """
        logger.info("[배치워커] 워커 시작")
        self.is_running = True

        while self.is_running:
            try:
                # 큐에서 다음 작업 가져오기
                job = await self.queue.get_next_job()

                if job:
                    # 작업 처리
                    await self.process_batch(job)
                else:
                    # 큐가 비어있으면 대기
                    await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"[배치워커] 워커 루프 오류: {e}", exc_info=True)
                await asyncio.sleep(10)

        logger.info("[배치워커] 워커 종료")

    async def stop(self):
        """워커 중지"""
        logger.info("[배치워커] 워커 중지 요청")
        self.is_running = False


# 전역 워커 인스턴스
_worker_instance: Optional[BatchWorker] = None
_worker_task: Optional[asyncio.Task] = None
_maintenance_task: Optional[asyncio.Task] = None
_maintenance_stop_event: Optional[asyncio.Event] = None


async def _run_batch_maintenance(stop_event: asyncio.Event) -> None:
    queue = get_batch_queue()
    interval = settings.BATCH_STALE_RECOVERY_INTERVAL_SECONDS

    while not stop_event.is_set():
        try:
            await queue.recover_stale_jobs()
        except Exception as exc:
            logger.error(f"[배치워커] 배치 복구 루프 오류: {exc}", exc_info=True)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


async def start_batch_worker():
    """배치 워커 시작"""
    global _worker_instance, _worker_task, _maintenance_task, _maintenance_stop_event

    if _worker_instance is not None:
        logger.warning("[배치워커] 워커가 이미 실행 중입니다")
        return

    _worker_instance = BatchWorker()
    _worker_task = asyncio.create_task(_worker_instance.run())
    if _maintenance_task is None:
        _maintenance_stop_event = asyncio.Event()
        _maintenance_task = asyncio.create_task(_run_batch_maintenance(_maintenance_stop_event))

    logger.info("[배치워커] 배치 워커가 백그라운드에서 시작되었습니다")


async def stop_batch_worker():
    """배치 워커 중지"""
    global _worker_instance, _worker_task, _maintenance_task, _maintenance_stop_event

    if _worker_instance is None:
        return

    await _worker_instance.stop()

    if _worker_task:
        await _worker_task
    if _maintenance_stop_event:
        _maintenance_stop_event.set()
    if _maintenance_task:
        await _maintenance_task

    _worker_instance = None
    _worker_task = None
    _maintenance_task = None
    _maintenance_stop_event = None

    logger.info("[배치워커] 배치 워커가 중지되었습니다")


def get_batch_worker() -> Optional[BatchWorker]:
    """배치 워커 인스턴스 반환"""
    return _worker_instance
