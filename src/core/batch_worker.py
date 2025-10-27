"""
배치 워커 시스템

배치 큐에서 작업을 가져와 LLM으로 처리하고 콜백 전송
"""

import asyncio
import time
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .logger import logger
from .batch_queue import get_batch_queue, BatchJob, BatchStatus
from .file_processor import extract_conversation_text


class BatchWorker:
    """배치 작업 처리 워커"""

    def __init__(self):
        """초기화"""
        self.is_running = False
        self.queue = get_batch_queue()
        self.llm_service = None  # LLM 서비스 (지연 로드)

        # 콜백 설정
        self.callback_timeout = 30  # 콜백 타임아웃 (초)
        self.callback_retry_count = 3  # 재시도 횟수
        self.callback_retry_interval = 5  # 재시도 간격 (초)

        logger.info("[배치워커] 배치 워커 초기화 완료")

    async def load_llm_service(self, model_name: str = "qwen3_4b"):
        """
        LLM 서비스 로드 (지연 초기화)

        Args:
            model_name: 모델 이름 (qwen3_4b, ax_light)
        """
        if self.llm_service is not None:
            # 이미 로드된 경우 정리
            if hasattr(self.llm_service, 'cleanup'):
                self.llm_service.cleanup()

        logger.info(f"[배치워커] LLM 서비스 로드 시작: {model_name}")

        try:
            if model_name == "qwen3_4b":
                from ..services.consultation_service import ConsultationService
                self.llm_service = ConsultationService(model_path=r"models\Qwen3-4B")
                success = self.llm_service.initialize()

            elif model_name == "ax_light":
                from .models.ax_light.summarizer import AXLightSummarizer
                self.llm_service = AXLightSummarizer(model_path=r"models\A.X-4.0-Light")
                success = self.llm_service.load_model()

            else:
                logger.error(f"[배치워커] 지원하지 않는 모델: {model_name}")
                return False

            if success:
                logger.info(f"[배치워커] LLM 서비스 로드 완료: {model_name}")
                return True
            else:
                logger.error(f"[배치워커] LLM 서비스 로드 실패: {model_name}")
                return False

        except Exception as e:
            logger.error(f"[배치워커] LLM 서비스 로드 오류: {e}", exc_info=True)
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

            # 2. LLM으로 분석
            if model_name == "qwen3_4b":
                # ConsultationService 사용 (요약+키워드+제목)
                result = self.llm_service.analyze_consultation_text(conversation_text)

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

            elif model_name == "ax_light":
                # A.X-4.0-Light 사용 (요약만)
                result = self.llm_service.summarize_consultation(conversation_text)

                if result.get('success'):
                    return {
                        "consultation_id": consultation_id,
                        "success": True,
                        "summary": result.get('summary', ''),
                        "categories": [],  # A.X는 요약만 지원
                        "titles": [],
                        "processing_time": time.time() - start_time,
                        "model": "A.X-4.0-Light"
                    }
                else:
                    return {
                        "consultation_id": consultation_id,
                        "success": False,
                        "error": result.get('error', '알 수 없는 오류'),
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
                        return True
                    else:
                        logger.warning(
                            f"[배치워커] 콜백 응답 오류 - 상태코드: {response.status_code}, "
                            f"응답: {response.text[:200]}"
                        )

            except httpx.TimeoutException:
                logger.warning(f"[배치워커] 콜백 타임아웃 - 시도 {attempt}/{self.callback_retry_count}")

            except Exception as e:
                logger.error(f"[배치워커] 콜백 전송 오류 - 시도 {attempt}/{self.callback_retry_count}: {e}")

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
            f"통화수: {job.total_count}, 모델: {job.llm_model}"
        )

        batch_start_time = time.time()

        try:
            # 1. LLM 서비스 로드
            success = await self.load_llm_service(job.llm_model)
            if not success:
                await self.queue.update_job_status(
                    batch_id,
                    BatchStatus.FAILED,
                    error="LLM 서비스 로드 실패"
                )
                return

            # 2. 각 상담 처리
            results = []
            for idx, consultation in enumerate(job.consultations, 1):
                logger.info(f"[배치워커] 처리 중 [{idx}/{job.total_count}] - {consultation.get('consultation_id')}")

                result = await self.process_consultation(consultation, job.llm_model)
                results.append(result)

            # 3. 배치 상태 업데이트
            await self.queue.update_job_status(
                batch_id,
                BatchStatus.COMPLETED,
                results=results
            )

            # 4. 콜백 데이터 준비
            callback_data = {
                "batch_id": batch_id,
                "status": "completed",
                "total_count": job.total_count,
                "success_count": sum(1 for r in results if r.get('success')),
                "failed_count": sum(1 for r in results if not r.get('success')),
                "results": results,
                "total_processing_time": time.time() - batch_start_time,
                "completed_at": datetime.now(timezone.utc).isoformat()
            }

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
            # LLM 서비스 정리
            if self.llm_service and hasattr(self.llm_service, 'cleanup'):
                self.llm_service.cleanup()
                self.llm_service = None

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


async def start_batch_worker():
    """배치 워커 시작"""
    global _worker_instance, _worker_task

    if _worker_instance is not None:
        logger.warning("[배치워커] 워커가 이미 실행 중입니다")
        return

    _worker_instance = BatchWorker()
    _worker_task = asyncio.create_task(_worker_instance.run())

    logger.info("[배치워커] 배치 워커가 백그라운드에서 시작되었습니다")


async def stop_batch_worker():
    """배치 워커 중지"""
    global _worker_instance, _worker_task

    if _worker_instance is None:
        return

    await _worker_instance.stop()

    if _worker_task:
        await _worker_task

    _worker_instance = None
    _worker_task = None

    logger.info("[배치워커] 배치 워커가 중지되었습니다")


def get_batch_worker() -> Optional[BatchWorker]:
    """배치 워커 인스턴스 반환"""
    return _worker_instance
