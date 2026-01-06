"""
배치 작업 큐

배치 모델 비동기 처리를 위한 큐/상태 저장소.
SQLite 기반으로 프로세스 재시작/멀티 워커에서도 상태를 유지합니다.
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .config import get_application_settings
from .logger import logger

settings = get_application_settings()


class BatchStatus(str, Enum):
    """배치 작업 상태"""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchJob:
    """배치 작업 정보"""

    def __init__(
        self,
        batch_id: str,
        consultations: List[Dict[str, Any]],
        callback_url: str,
        batch_model: str = "qwen3_4b",
        priority: int = 1,
        bound_key: str = "",
    ):
        self.batch_id = batch_id
        self.consultations = consultations
        self.callback_url = callback_url
        self.batch_model = batch_model
        self.priority = priority
        self.bound_key = bound_key

        self.status = BatchStatus.QUEUED
        self.total_count = len(consultations)
        self.processed_count = 0
        self.success_count = 0
        self.failed_count = 0

        self.results: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.processing_attempts = 0
        self.callback_status: Optional[str] = None
        self.callback_attempts = 0
        self.callback_last_error: Optional[str] = None
        self.callback_last_attempt_at: Optional[datetime] = None
        self.callback_errors: List[str] = []

        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    @classmethod
    def from_record(cls, record: Union[sqlite3.Row, Dict[str, Any]]) -> "BatchJob":
        consultations = json.loads(record["consultations_json"] or "[]")
        job = cls(
            batch_id=record["batch_id"],
            consultations=consultations,
            callback_url=record["callback_url"],
            batch_model=record["batch_model"],
            priority=record["priority"],
            bound_key=record["bound_key"] or "",
        )
        job.status = BatchStatus(record["status"])
        job.total_count = record["total_count"]
        job.processed_count = record["processed_count"]
        job.success_count = record["success_count"]
        job.failed_count = record["failed_count"]
        job.results = json.loads(record["results_json"] or "[]")
        job.errors = json.loads(record["errors_json"] or "[]")
        job.processing_attempts = record["processing_attempts"] or 0
        job.callback_status = record["callback_status"]
        job.callback_attempts = record["callback_attempts"] or 0
        job.callback_last_error = record["callback_last_error"]
        job.callback_last_attempt_at = _parse_datetime(record["callback_last_attempt_at"])
        job.callback_errors = json.loads(record["callback_errors_json"] or "[]")
        job.created_at = _parse_datetime(record["created_at"]) or job.created_at
        job.started_at = _parse_datetime(record["started_at"])
        job.completed_at = _parse_datetime(record["completed_at"])
        return job

    def to_dict(self) -> Dict[str, Any]:
        """배치 작업 정보를 딕셔너리로 변환"""
        return {
            "batch_id": self.batch_id,
            "status": self.status.value,
            "total_count": self.total_count,
            "processed_count": self.processed_count,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "batch_model": self.batch_model,
            "priority": self.priority,
            "callback_url": self.callback_url,
            "processing_attempts": self.processing_attempts,
            "callback_status": self.callback_status,
            "callback_attempts": self.callback_attempts,
            "callback_last_error": self.callback_last_error,
            "callback_last_attempt_at": (
                self.callback_last_attempt_at.isoformat()
                if self.callback_last_attempt_at
                else None
            ),
            "callback_errors": self.callback_errors,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at and self.started_at
                else None
            ),
        }


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BatchQueue:
    """배치 작업 큐 관리자 (SQLite 기반)"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._db_path = Path(settings.BATCH_QUEUE_DB_PATH)
        self._max_retained_jobs = settings.BATCH_QUEUE_MAX_RETAINED
        self._processing_timeout_seconds = settings.BATCH_PROCESSING_TIMEOUT_SECONDS
        self._processing_max_attempts = settings.BATCH_PROCESSING_MAX_ATTEMPTS
        self._init_db()
        logger.info("[배치큐] 배치 큐 초기화 완료 (%s)", self._db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_columns(self, conn: sqlite3.Connection, columns: Dict[str, str]) -> None:
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(batch_jobs)").fetchall()
        }
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE batch_jobs ADD COLUMN {name} {definition}")

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    batch_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    consultations_json TEXT NOT NULL,
                    callback_url TEXT NOT NULL,
                    batch_model TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    bound_key TEXT,
                    total_count INTEGER NOT NULL,
                    processed_count INTEGER NOT NULL,
                    success_count INTEGER NOT NULL,
                    failed_count INTEGER NOT NULL,
                    results_json TEXT,
                    errors_json TEXT,
                    processing_attempts INTEGER NOT NULL DEFAULT 0,
                    callback_status TEXT,
                    callback_attempts INTEGER NOT NULL DEFAULT 0,
                    callback_last_error TEXT,
                    callback_last_attempt_at TEXT,
                    callback_errors_json TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_batch_status_created ON batch_jobs(status, created_at)"
            )
            self._ensure_columns(
                conn,
                {
                    "processing_attempts": "INTEGER NOT NULL DEFAULT 0",
                    "callback_status": "TEXT",
                    "callback_attempts": "INTEGER NOT NULL DEFAULT 0",
                    "callback_last_error": "TEXT",
                    "callback_last_attempt_at": "TEXT",
                    "callback_errors_json": "TEXT",
                },
            )
            conn.commit()

    async def add_job(self, job: BatchJob) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self._add_job_sync, job)

    def _add_job_sync(self, job: BatchJob) -> bool:
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO batch_jobs (
                        batch_id,
                        status,
                        consultations_json,
                        callback_url,
                        batch_model,
                        priority,
                        bound_key,
                        total_count,
                        processed_count,
                        success_count,
                        failed_count,
                        results_json,
                        errors_json,
                        processing_attempts,
                        callback_status,
                        callback_attempts,
                        callback_last_error,
                        callback_last_attempt_at,
                        callback_errors_json,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.batch_id,
                        job.status.value,
                        json.dumps(job.consultations, ensure_ascii=False),
                        job.callback_url,
                        job.batch_model,
                        job.priority,
                        job.bound_key,
                        job.total_count,
                        job.processed_count,
                        job.success_count,
                        job.failed_count,
                        json.dumps(job.results, ensure_ascii=False),
                        json.dumps(job.errors, ensure_ascii=False),
                        job.processing_attempts,
                        job.callback_status,
                        job.callback_attempts,
                        job.callback_last_error,
                        job.callback_last_attempt_at.isoformat()
                        if job.callback_last_attempt_at
                        else None,
                        json.dumps(job.callback_errors, ensure_ascii=False),
                        job.created_at.isoformat(),
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                logger.warning("[배치큐] 중복 배치 ID: %s", job.batch_id)
                return False

        self._prune_finished_jobs_sync()
        return True

    async def get_next_job(self) -> Optional[BatchJob]:
        async with self._lock:
            return await asyncio.to_thread(self._claim_next_job_sync)

    def _recover_stale_jobs_sync(self, conn: sqlite3.Connection) -> Dict[str, int]:
        if self._processing_timeout_seconds <= 0:
            return {"requeued": 0, "failed": 0}

        now = datetime.now(timezone.utc)
        rows = conn.execute(
            "SELECT batch_id, started_at, processing_attempts, errors_json FROM batch_jobs WHERE status = ?",
            (BatchStatus.PROCESSING.value,),
        ).fetchall()

        requeued = 0
        failed = 0
        for row in rows:
            started_at = _parse_datetime(row["started_at"])
            if not started_at:
                continue

            elapsed = (now - started_at).total_seconds()
            if elapsed < self._processing_timeout_seconds:
                continue

            attempts = row["processing_attempts"] or 0
            errors = json.loads(row["errors_json"] or "[]")

            if attempts >= self._processing_max_attempts:
                errors.append(
                    f"Processing timeout exceeded ({int(elapsed)}s). Max attempts reached."
                )
                conn.execute(
                    """
                    UPDATE batch_jobs
                    SET status = ?, completed_at = ?, errors_json = ?
                    WHERE batch_id = ?
                    """,
                    (
                        BatchStatus.FAILED.value,
                        _utc_now(),
                        json.dumps(errors, ensure_ascii=False),
                        row["batch_id"],
                    ),
                )
                logger.warning(
                    "[배치큐] 처리 타임아웃으로 실패 처리 - ID: %s, attempts=%s",
                    row["batch_id"],
                    attempts,
                )
                failed += 1
            else:
                errors.append(
                    f"Processing timeout exceeded ({int(elapsed)}s). Requeued."
                )
                conn.execute(
                    """
                    UPDATE batch_jobs
                    SET status = ?, started_at = ?, completed_at = ?, errors_json = ?
                    WHERE batch_id = ?
                    """,
                    (
                        BatchStatus.QUEUED.value,
                        None,
                        None,
                        json.dumps(errors, ensure_ascii=False),
                        row["batch_id"],
                    ),
                )
                logger.warning(
                    "[배치큐] 처리 타임아웃으로 재큐잉 - ID: %s, attempts=%s",
                    row["batch_id"],
                    attempts,
                )
                requeued += 1

        return {"requeued": requeued, "failed": failed}

    def _recover_stale_jobs_once_sync(self) -> Dict[str, int]:
        with self._connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                stats = self._recover_stale_jobs_sync(conn)
                conn.commit()
                return stats
            except Exception:
                conn.rollback()
                raise

    async def recover_stale_jobs(self) -> Dict[str, int]:
        async with self._lock:
            return await asyncio.to_thread(self._recover_stale_jobs_once_sync)

    def _claim_next_job_sync(self) -> Optional[BatchJob]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._recover_stale_jobs_sync(conn)
            row = conn.execute(
                """
                SELECT * FROM batch_jobs
                WHERE status = ?
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
                """,
                (BatchStatus.QUEUED.value,),
            ).fetchone()

            if not row:
                conn.commit()
                return None

            started_at = _utc_now()
            processing_attempts = (row["processing_attempts"] or 0) + 1
            updated = conn.execute(
                """
                UPDATE batch_jobs
                SET status = ?, started_at = ?, processing_attempts = ?
                WHERE batch_id = ? AND status = ?
                """,
                (
                    BatchStatus.PROCESSING.value,
                    started_at,
                    processing_attempts,
                    row["batch_id"],
                    BatchStatus.QUEUED.value,
                ),
            )

            if updated.rowcount == 0:
                conn.rollback()
                return None

            conn.commit()
            refreshed = conn.execute(
                "SELECT * FROM batch_jobs WHERE batch_id = ?",
                (row["batch_id"],),
            ).fetchone()

        return BatchJob.from_record(refreshed) if refreshed else None

    async def update_job_status(
        self,
        batch_id: str,
        status: BatchStatus,
        results: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(self._update_job_status_sync, batch_id, status, results, error)

    def _update_job_status_sync(
        self,
        batch_id: str,
        status: BatchStatus,
        results: Optional[List[Dict[str, Any]]],
        error: Optional[str],
    ) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM batch_jobs WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if not row:
                logger.warning("[배치큐] 존재하지 않는 배치 ID: %s", batch_id)
                return

            updates: Dict[str, Any] = {"status": status.value}

            if status in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED):
                updates["completed_at"] = _utc_now()

            if results is not None:
                success_count = sum(1 for r in results if r.get("success"))
                failed_count = len(results) - success_count
                updates.update(
                    {
                        "results_json": json.dumps(results, ensure_ascii=False),
                        "processed_count": len(results),
                        "success_count": success_count,
                        "failed_count": failed_count,
                    }
                )

            if error:
                errors = json.loads(row["errors_json"] or "[]")
                errors.append(error)
                updates["errors_json"] = json.dumps(errors, ensure_ascii=False)

            columns = ", ".join(f"{key} = ?" for key in updates.keys())
            values = list(updates.values())
            values.append(batch_id)
            conn.execute(
                f"UPDATE batch_jobs SET {columns} WHERE batch_id = ?",
                values,
            )
            conn.commit()

        self._prune_finished_jobs_sync()

    async def update_callback_status(
        self,
        batch_id: str,
        status: str,
        attempt: int,
        error: Optional[str] = None,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._update_callback_status_sync,
                batch_id,
                status,
                attempt,
                error,
            )

    def _update_callback_status_sync(
        self,
        batch_id: str,
        status: str,
        attempt: int,
        error: Optional[str],
    ) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT callback_errors_json FROM batch_jobs WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if not row:
                logger.warning("[배치큐] 존재하지 않는 배치 ID: %s", batch_id)
                return

            updates: Dict[str, Any] = {
                "callback_status": status,
                "callback_attempts": attempt,
                "callback_last_attempt_at": _utc_now(),
            }

            if error is not None:
                errors = json.loads(row["callback_errors_json"] or "[]")
                errors.append(error)
                updates["callback_errors_json"] = json.dumps(errors, ensure_ascii=False)
                updates["callback_last_error"] = error
            else:
                updates["callback_last_error"] = None

            columns = ", ".join(f"{key} = ?" for key in updates.keys())
            values = list(updates.values())
            values.append(batch_id)
            conn.execute(
                f"UPDATE batch_jobs SET {columns} WHERE batch_id = ?",
                values,
            )
            conn.commit()

    async def get_job_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return await asyncio.to_thread(self._get_job_status_sync, batch_id)

    def _get_job_status_sync(self, batch_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM batch_jobs WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if not row:
                return None

        job = BatchJob.from_record(row)
        return job.to_dict()

    async def get_job(self, batch_id: str) -> Optional[BatchJob]:
        async with self._lock:
            return await asyncio.to_thread(self._get_job_sync, batch_id)

    def _get_job_sync(self, batch_id: str) -> Optional[BatchJob]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM batch_jobs WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if not row:
                return None
        return BatchJob.from_record(row)

    async def cancel_job(self, batch_id: str) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self._cancel_job_sync, batch_id)

    def _cancel_job_sync(self, batch_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM batch_jobs WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if not row:
                return False

            if row["status"] != BatchStatus.QUEUED.value:
                return False

            conn.execute(
                """
                UPDATE batch_jobs
                SET status = ?, completed_at = ?
                WHERE batch_id = ?
                """,
                (BatchStatus.CANCELLED.value, _utc_now(), batch_id),
            )
            conn.commit()
            return True

    async def get_queue_stats(self) -> Dict[str, Any]:
        async with self._lock:
            return await asyncio.to_thread(self._get_queue_stats_sync)

    def _get_queue_stats_sync(self) -> Dict[str, Any]:
        stats = {
            "total_jobs": 0,
            "queued": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
        }

        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM batch_jobs").fetchone()
            stats["total_jobs"] = total[0] if total else 0

            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM batch_jobs GROUP BY status"
            ).fetchall()

        for row in rows:
            stats[row["status"]] = row["cnt"]

        stats["queue_length"] = stats["queued"]
        return stats

    def _prune_finished_jobs_sync(self) -> None:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM batch_jobs").fetchone()
            total_count = total[0] if total else 0
            if total_count <= self._max_retained_jobs:
                return

            excess = total_count - self._max_retained_jobs
            rows = conn.execute(
                """
                SELECT batch_id FROM batch_jobs
                WHERE status IN (?, ?, ?)
                ORDER BY completed_at ASC, created_at ASC
                LIMIT ?
                """,
                (
                    BatchStatus.COMPLETED.value,
                    BatchStatus.FAILED.value,
                    BatchStatus.CANCELLED.value,
                    excess,
                ),
            ).fetchall()

            batch_ids = [row["batch_id"] for row in rows]
            if not batch_ids:
                return

            conn.executemany(
                "DELETE FROM batch_jobs WHERE batch_id = ?",
                [(batch_id,) for batch_id in batch_ids],
            )
            conn.commit()


_queue_instance: Optional[BatchQueue] = None


def get_batch_queue() -> BatchQueue:
    """배치 큐 인스턴스 반환 (싱글톤)"""
    global _queue_instance

    if _queue_instance is None:
        _queue_instance = BatchQueue()

    return _queue_instance
