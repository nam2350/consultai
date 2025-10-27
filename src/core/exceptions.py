"""
커스텀 예외 클래스 및 에러 핸들러 모듈

애플리케이션 전반에서 사용되는 비즈니스 예외와 HTTP 예외 핸들러를 정의합니다.
체계적인 에러 코드와 상세한 로깅을 통해 디버깅과 모니터링을 지원합니다.
"""
from typing import Any, Dict, Optional
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
import traceback
from enum import Enum

class ApplicationErrorCode(Enum):
    """
    애플리케이션 에러 코드 열거형
    
    카테고리별로 구분된 에러 코드를 정의합니다.
    - 1000번대: 일반 시스템 에러
    - 2000번대: AI 모델 관련 에러  
    - 3000번대: 외부 API 연동 에러
    - 4000번대: 파일 처리 에러
    """
    
    # 일반 에러 (1000번대)
    INTERNAL_SERVER_ERROR = "E1000"
    VALIDATION_ERROR = "E1001"
    CONFIGURATION_ERROR = "E1002"
    
    # 모델 관련 에러 (2000번대)
    MODEL_NOT_FOUND = "E2000"
    MODEL_LOAD_FAILED = "E2001"
    MODEL_INFERENCE_FAILED = "E2002"
    MODEL_SWITCH_FAILED = "E2003"
    
    # API 관련 에러 (3000번대)
    EXTERNAL_API_ERROR = "E3000"
    RATE_LIMIT_EXCEEDED = "E3001"
    TIMEOUT_ERROR = "E3002"
    
    # 파일 관련 에러 (4000번대)
    FILE_NOT_FOUND = "E4000"
    FILE_TOO_LARGE = "E4001"
    INVALID_FILE_FORMAT = "E4002"
    FILE_UPLOAD_FAILED = "E4003"

class ApplicationBusinessException(Exception):
    """
    애플리케이션 비즈니스 로직 예외
    
    비즈니스 규칙 위반이나 예상된 에러 상황에서 발생하는 예외입니다.
    체계적인 에러 코드와 상세 정보를 포함합니다.
    """
    
    def __init__(
        self, 
        error_message: str, 
        error_code: ApplicationErrorCode, 
        error_details: Optional[Dict[str, Any]] = None,
        http_status_code: int = 500
    ):
        self.message = error_message
        self.error_code = error_code
        self.details = error_details or {}
        self.status_code = http_status_code
        super().__init__(error_message)

class AIModelException(ApplicationBusinessException):
    """
    AI 모델 관련 예외
    
    모델 로딩, 추론, 전환 등 AI 모델 작업 중 발생하는 예외입니다.
    """
    
    def __init__(self, error_message: str, error_code: ApplicationErrorCode, model_name: str = None):
        model_details = {"model_name": model_name} if model_name else {}
        super().__init__(error_message, error_code, model_details, 500)

class ExternalAPIConnectionException(ApplicationBusinessException):
    """
    외부 API 연동 예외
    
    외부 서비스와의 통신 중 발생하는 예외입니다.
    """
    
    def __init__(self, error_message: str, error_code: ApplicationErrorCode, api_endpoint_url: str = None):
        api_details = {"api_url": api_endpoint_url} if api_endpoint_url else {}
        super().__init__(error_message, error_code, api_details, 502)

class FileProcessingException(ApplicationBusinessException):
    """
    파일 처리 예외
    
    파일 업로드, 읽기, 파싱 등 파일 관련 작업 중 발생하는 예외입니다.
    """
    
    def __init__(self, error_message: str, error_code: ApplicationErrorCode, file_path: str = None):
        file_details = {"file_path": file_path} if file_path else {}
        super().__init__(error_message, error_code, file_details, 400)

# HTTP 예외 핸들러 함수들
async def handle_business_exception(request: Request, exception: ApplicationBusinessException):
    """
    비즈니스 예외 처리 핸들러
    
    애플리케이션 비즈니스 로직에서 발생한 예외를 처리하고
    적절한 HTTP 응답과 로그를 생성합니다.
    """
    from ..core.logger import logger
    
    logger.error(
        f"비즈니스 예외 발생: {exception.error_code.value} - {exception.message}",
        extra={
            "error_code": exception.error_code.value,
            "details": exception.details,
            "path": str(request.url),
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=exception.status_code,
        content={
            "success": False,
            "error": {
                "code": exception.error_code.value,
                "message": exception.message,
                "details": exception.details
            },
            "timestamp": str(request.state.start_time) if hasattr(request.state, 'start_time') else None
        }
    )

async def handle_general_exception(request: Request, exception: Exception):
    """
    일반 예외 처리 핸들러
    
    예상하지 못한 시스템 에러를 처리하고 고유한 에러 ID를 생성합니다.
    """
    from ..core.logger import logger
    
    unique_error_id = f"ERR_{request.state.start_time}_{hash(str(exception))}"
    
    logger.error(
        f"예상치 못한 에러 발생 [ID: {unique_error_id}]: {str(exception)}",
        extra={
            "error_id": unique_error_id,
            "path": str(request.url),
            "method": request.method,
            "traceback": traceback.format_exc()
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": ApplicationErrorCode.INTERNAL_SERVER_ERROR.value,
                "message": "내부 서버 오류가 발생했습니다",
                "error_id": unique_error_id
            }
        }
    )

async def handle_http_exception(request: Request, http_exception: HTTPException):
    """
    HTTP 예외 처리 핸들러
    
    FastAPI에서 발생하는 HTTP 예외를 처리하고 일관된 응답 형식을 제공합니다.
    """
    from ..core.logger import logger
    
    logger.warning(
        f"HTTP 예외: {http_exception.status_code} - {http_exception.detail}",
        extra={
            "status_code": http_exception.status_code,
            "path": str(request.url),
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=http_exception.status_code,
        content={
            "success": False,
            "error": {
                "code": f"HTTP_{http_exception.status_code}",
                "message": http_exception.detail
            }
        }
    )

def register_application_exception_handlers(fastapi_app):
    """
    FastAPI 애플리케이션에 예외 핸들러 등록
    
    비즈니스 예외, HTTP 예외, 일반 예외에 대한 핸들러를 등록합니다.
    """
    fastapi_app.add_exception_handler(ApplicationBusinessException, handle_business_exception)
    fastapi_app.add_exception_handler(HTTPException, handle_http_exception)
    fastapi_app.add_exception_handler(Exception, handle_general_exception)