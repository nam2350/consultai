"""
배치 작업 큐 시스템

LLM 비동기 배치 처리를 위한 인메모리 큐 시스템
"""

import time
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from enum import Enum

from .logger import logger


class BatchStatus(str, Enum):
    """배치 작업 상태"""
    QUEUED = "queued"           # 대기 중
    PROCESSING = "processing"    # 처리 중
    COMPLETED = "completed"      # 완료
    FAILED = "failed"           # 실패
    CANCELLED = "cancelled"      # 취소됨


class BatchJob:
    """배치 작업 정보"""

    def __init__(
        self,
        batch_id: str,
        consultations: List[Dict[str, Any]],
        callback_url: str,
        llm_model: str = "qwen3_4b",
        priority: int = 1,
        bound_key: str = ""
    ):
        self.batch_id = batch_id
        self.consultations = consultations
        self.callback_url = callback_url
        self.llm_model = llm_model
        self.priority = priority
        self.bound_key = bound_key

        # 상태 관리
        self.status = BatchStatus.QUEUED
        self.total_count = len(consultations)
        self.processed_count = 0
        self.success_count = 0
        self.failed_count = 0

        # 결과 저장
        self.results: List[Dict[str, Any]] = []
        self.errors: List[str] = []

        # 시간 정보
        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """배치 작업 정보를 딕셔너리로 변환"""
        return {
            "batch_id": self.batch_id,
            "status": self.status.value,
            "total_count": self.total_count,
            "processed_count": self.processed_count,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "llm_model": self.llm_model,
            "priority": self.priority,
            "callback_url": self.callback_url,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at and self.started_at
                else None
            )
        }


class BatchQueue:
    """배치 작업 큐 관리자 (인메모리)"""

    def __init__(self):
        """초기화"""
        self.jobs: Dict[str, BatchJob] = {}  # batch_id -> BatchJob
        self.queue: List[str] = []  # batch_id 큐 (FIFO)
        self._lock = asyncio.Lock()

        logger.info("[배치큐] 배치 큐 시스템 초기화 완료")

    async def add_job(self, job: BatchJob) -> bool:
        """
        배치 작업 추가

        Args:
            job: 배치 작업 객체

        Returns:
            추가 성공 여부
        """
        async with self._lock:
            if job.batch_id in self.jobs:
                logger.warning(f"[배치큐] 중복된 배치 ID: {job.batch_id}")
                return False

            # 작업 저장
            self.jobs[job.batch_id] = job

            # 우선순위에 따라 큐에 추가
            if job.priority == 1:  # 높은 우선순위
                self.queue.insert(0, job.batch_id)
            else:
                self.queue.append(job.batch_id)

            logger.info(
                f"[배치큐] 배치 작업 추가 - ID: {job.batch_id}, "
                f"통화수: {job.total_count}, 우선순위: {job.priority}"
            )

            return True

    async def get_next_job(self) -> Optional[BatchJob]:
        """
        다음 처리할 작업 가져오기

        Returns:
            배치 작업 객체 (없으면 None)
        """
        async with self._lock:
            while self.queue:
                batch_id = self.queue.pop(0)

                # 작업 존재 확인
                if batch_id not in self.jobs:
                    continue

                job = self.jobs[batch_id]

                # 상태 확인 (QUEUED만 처리)
                if job.status != BatchStatus.QUEUED:
                    continue

                # 상태 변경
                job.status = BatchStatus.PROCESSING
                job.started_at = datetime.now(timezone.utc)

                logger.info(f"[배치큐] 배치 작업 시작 - ID: {job.batch_id}")

                return job

            return None

    async def update_job_status(
        self,
        batch_id: str,
        status: BatchStatus,
        results: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None
    ):
        """
        배치 작업 상태 업데이트

        Args:
            batch_id: 배치 ID
            status: 새 상태
            results: 처리 결과 (완료 시)
            error: 에러 메시지 (실패 시)
        """
        async with self._lock:
            if batch_id not in self.jobs:
                logger.warning(f"[배치큐] 존재하지 않는 배치 ID: {batch_id}")
                return

            job = self.jobs[batch_id]
            job.status = status

            if status == BatchStatus.COMPLETED:
                job.completed_at = datetime.now(timezone.utc)
                if results:
                    job.results = results
                    job.success_count = sum(1 for r in results if r.get('success'))
                    job.failed_count = len(results) - job.success_count
                    job.processed_count = len(results)

                logger.info(
                    f"[배치큐] 배치 작업 완료 - ID: {batch_id}, "
                    f"성공: {job.success_count}/{job.total_count}"
                )

            elif status == BatchStatus.FAILED:
                job.completed_at = datetime.now(timezone.utc)
                if error:
                    job.errors.append(error)

                logger.error(f"[배치큐] 배치 작업 실패 - ID: {batch_id}, 에러: {error}")

    async def get_job_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        배치 작업 상태 조회

        Args:
            batch_id: 배치 ID

        Returns:
            작업 정보 딕셔너리 (없으면 None)
        """
        async with self._lock:
            if batch_id not in self.jobs:
                return None

            job = self.jobs[batch_id]
            return job.to_dict()

    async def get_job(self, batch_id: str) -> Optional[BatchJob]:
        """
        배치 작업 객체 가져오기

        Args:
            batch_id: 배치 ID

        Returns:
            배치 작업 객체 (없으면 None)
        """
        async with self._lock:
            return self.jobs.get(batch_id)

    async def cancel_job(self, batch_id: str) -> bool:
        """
        배치 작업 취소

        Args:
            batch_id: 배치 ID

        Returns:
            취소 성공 여부
        """
        async with self._lock:
            if batch_id not in self.jobs:
                return False

            job = self.jobs[batch_id]

            # QUEUED 상태만 취소 가능
            if job.status != BatchStatus.QUEUED:
                return False

            job.status = BatchStatus.CANCELLED
            job.completed_at = datetime.now(timezone.utc)

            # 큐에서 제거
            if batch_id in self.queue:
                self.queue.remove(batch_id)

            logger.info(f"[배치큐] 배치 작업 취소 - ID: {batch_id}")

            return True

    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        큐 통계 정보 조회

        Returns:
            통계 정보 딕셔너리
        """
        async with self._lock:
            stats = {
                "total_jobs": len(self.jobs),
                "queued": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0
            }

            for job in self.jobs.values():
                stats[job.status.value] += 1

            stats["queue_length"] = len(self.queue)

            return stats


# 전역 인스턴스 (싱글톤)
_queue_instance: Optional[BatchQueue] = None


def get_batch_queue() -> BatchQueue:
    """배치 큐 인스턴스 반환 (싱글톤)"""
    global _queue_instance

    if _queue_instance is None:
        _queue_instance = BatchQueue()

    return _queue_instance
