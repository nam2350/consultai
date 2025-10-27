"""
로깅 시스템 설정 모듈

애플리케이션의 로깅을 위한 loguru 기반 설정을 제공합니다.
콘솔, 파일, 에러, 성능 로그를 분리하여 관리합니다.
"""
import sys
import os
from loguru import logger
from src.core.config import settings

# 로그 저장을 위한 디렉토리 생성
os.makedirs("logs", exist_ok=True)

# loguru의 기본 로거 제거 (사용자 정의 설정 적용을 위해)
logger.remove()

# 콘솔 출력용 로거 설정 (컬러 포맷 적용)
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL,
    colorize=True
)

# 일반 애플리케이션 로그 파일 설정
logger.add(
    settings.LOG_FILE,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level=settings.LOG_LEVEL,
    rotation="10 MB",  # 파일 크기 10MB 도달 시 로테이션
    retention="7 days",  # 7일간 로그 파일 보관
    compression="zip"  # 로테이션된 파일 압축 저장
)

# 에러 전용 로그 파일 설정 (ERROR 레벨 이상만 기록)
logger.add(
    "logs/error.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",
    rotation="10 MB",
    retention="30 days",  # 에러 로그는 30일간 보관
    compression="zip"
)

# 성능 측정 전용 로그 파일 설정 (performance 태그가 있는 로그만 기록)
logger.add(
    "logs/performance.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
    filter=lambda record: "performance" in record["extra"],
    rotation="10 MB",
    retention="7 days",
    compression="zip"
)

# 개발 환경에서 디버그 로그를 stderr로 추가 출력
if settings.DEBUG:
    logger.add(
        sys.stderr,
        format="<red>{time:YYYY-MM-DD HH:mm:ss}</red> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG",
        colorize=True
    )

# 성능 측정 전용 로거 (performance 태그 바인딩)
application_performance_logger = logger.bind(performance=True)

def log_operation_performance(operation_name: str, execution_duration: float, **additional_context):
    """
    작업 성능 로그 기록
    
    특정 작업의 실행 시간과 추가 컨텍스트 정보를 성능 로그에 기록합니다.
    
    Args:
        operation_name: 작업 이름
        execution_duration: 실행 시간 (초)
        **additional_context: 추가 컨텍스트 정보
    """
    application_performance_logger.info(f"{operation_name} 완료 - {execution_duration:.3f}초", **additional_context)

def log_api_request_details(http_method: str, request_path: str, response_duration: float, http_status_code: int):
    """
    API 요청 상세 로그 기록
    
    HTTP 요청의 메소드, 경로, 응답 시간, 상태 코드를 로그에 기록합니다.
    
    Args:
        http_method: HTTP 메소드 (GET, POST, etc.)
        request_path: 요청 경로
        response_duration: 응답 시간 (초)
        http_status_code: HTTP 상태 코드
    """
    logger.info(f"API 요청: {http_method} {request_path} - {http_status_code} ({response_duration:.3f}초)")

def log_application_error(error_exception: Exception, error_context: str = ""):
    """
    애플리케이션 에러 로그 기록
    
    예외 정보와 컨텍스트를 포함하여 에러 로그에 상세히 기록합니다.
    
    Args:
        error_exception: 발생한 예외 객체
        error_context: 에러 발생 컨텍스트 설명
    """
    logger.error(f"{error_context}: {str(error_exception)}", exc_info=True) 