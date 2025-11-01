"""
상담 분석 API 엔드포인트

센터링크 연동을 위한 완전한 상담 분석 API를 제공합니다.
- 단일 상담 분석
- 배치 상담 분석  
- 시스템 상태 조회
- 서비스 관리
"""

import asyncio
import time
from typing import Dict, Any
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse

from ...schemas.consultation import (
    ConsultationAnalysisRequest,
    ConsultationAnalysisResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    SystemStatus,
    ErrorResponse,
    ErrorDetail
)
from ...services.consultation_service import ConsultationService
from ...core.logger import logger
from ...core.config import get_application_settings

# 설정 로드
settings = get_application_settings()

# 라우터 생성
router = APIRouter(prefix="/consultation", tags=["상담 분석"])

# 전역 서비스 인스턴스 (싱글톤 패턴)
_consultation_service: ConsultationService = None
_service_start_time = None

async def get_consultation_service() -> ConsultationService:
    """상담 서비스 인스턴스 반환 (지연 초기화)"""
    global _consultation_service, _service_start_time
    
    if _consultation_service is None:
        logger.info("[API] 상담 서비스 초기화 시작...")
        _service_start_time = time.time()
        
        # 서비스 인스턴스 생성 및 초기화
        logger.info(f"[API] 모델 경로: {settings.MODEL_PATH}")
        _consultation_service = ConsultationService(
            model_path=settings.MODEL_PATH
        )
        
        # 동기 초기화 (AsyncIO 제거)
        logger.info(f"[API] 초기화 전 상태: is_initialized={_consultation_service.is_initialized}")
        success = _consultation_service.initialize()
        logger.info(f"[API] 초기화 결과: success={success}, is_initialized={_consultation_service.is_initialized}")
        
        if not success:
            logger.error("[API] 상담 서비스 초기화 실패")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI 모델 초기화에 실패했습니다"
            )
        
        logger.info("[API] 상담 서비스 초기화 완료")
    
    return _consultation_service

@router.post(
    "/analyze",
    response_model=ConsultationAnalysisResponse,
    summary="단일 상담 분석",
    description="STT 데이터를 받아 AI 기반 상담 분석을 수행합니다"
)
async def analyze_consultation(
    request: ConsultationAnalysisRequest
) -> ConsultationAnalysisResponse:
    """
    단일 상담 분석 API
    
    - **consultation_id**: 상담 고유 ID
    - **consultation_content**: 전체 상담 내용
    - **stt_data**: STT 변환된 대화 데이터 (6가지 형식 지원)
    - **options**: 분석 옵션 (요약, 카테고리 추천, 제목 생성)
    
    Returns:
        ConsultationAnalysisResponse: 분석 결과 포함 응답
    """
    try:
        logger.info(f"[API] 상담 분석 요청 - ID: {request.consultation_id}")
        
        # 서비스 인스턴스 획득
        service = await get_consultation_service()
        
        # 분석 실행 (동기 호출)
        result = service.analyze_consultation(request)
        
        # 성공/실패에 따른 로그
        if result.success:
            logger.info(f"[API] 상담 분석 성공 - ID: {request.consultation_id}")
        else:
            logger.warning(f"[API] 상담 분석 실패 - ID: {request.consultation_id}, Error: {result.error}")
        
        return result
        
    except HTTPException:
        # HTTPException은 그대로 전파
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"[API] 상담 분석 API 오류: {e}")
        logger.error(f"[API] 상세 오류: {error_details}")
        
        # 예외 발생시 실패 응답 반환
        return ConsultationAnalysisResponse(
            consultation_id=request.consultation_id,
            success=False,
            error=f"API 처리 오류: {str(e) or type(e).__name__} - {error_details[:200]}",
            error_code="API_ERROR"
        )

@router.post(
    "/batch-analyze", 
    response_model=BatchAnalysisResponse,
    summary="배치 상담 분석",
    description="여러 상담을 동시에 분석합니다"
)
async def batch_analyze_consultations(
    request: BatchAnalysisRequest
) -> BatchAnalysisResponse:
    """
    배치 상담 분석 API
    
    - **consultation_requests**: 분석할 상담 요청 목록
    - **batch_options**: 배치 처리 옵션 (동시 처리 수 등)
    
    Returns:
        BatchAnalysisResponse: 배치 분석 결과
    """
    try:
        batch_size = len(request.consultation_requests)
        logger.info(f"[API] 배치 분석 요청 - {batch_size}개 상담")
        
        if batch_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="분석할 상담이 없습니다"
            )
        
        if batch_size > 100:  # 배치 크기 제한
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="배치 크기는 100개를 초과할 수 없습니다"
            )
        
        # 서비스 인스턴스 획득
        service = await get_consultation_service()
        
        # 배치 분석 실행 (동기 호출)
        result = service.batch_analyze(request)
        
        logger.info(f"[API] 배치 분석 완료 - {result.success_count}/{result.total_count} 성공")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] 배치 분석 API 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"배치 분석 처리 중 오류가 발생했습니다: {str(e)}"
        )

@router.get(
    "/status",
    summary="시스템 상태 조회",
    description="상담 분석 시스템의 현재 상태를 조회합니다"
)
async def get_system_status():
    """
    시스템 상태 조회 API
    
    Returns:
        SystemStatus: 시스템 상태 정보
    """
    try:
        global _service_start_time

        # 서비스 인스턴스 획득 (지연 초기화)
        try:
            service = await get_consultation_service()
        except Exception as e:
            logger.error(f"[API] 시스템 상태 조회 중 서비스 초기화 실패: {e}")
            # 서비스 초기화 실패 시 기본 응답
            uninitialized_response = SystemStatus(
                status="error",
                model_loaded=False,
                model_name="N/A",
                uptime=0.0,
                processed_consultations=0,
                average_processing_time=0.0
            )

            enhanced_uninitialized = uninitialized_response.model_dump()
            enhanced_uninitialized.update({
                "service_initialized": False,
                "statistics": {
                    "processed_consultations": 0,
                    "successful_analyses": 0,
                    "failed_analyses": 0,
                    "success_rate": 0.0,
                    "average_processing_time": 0.0
                },
                "ai_analyzer_status": {
                    "model_loaded": False,
                    "model_name": "N/A"
                }
            })

            return enhanced_uninitialized

        # 서비스 상태 정보 수집 (초기화된 서비스 인스턴스 사용)
        service_status = service.get_service_status()
        uptime = time.time() - _service_start_time if _service_start_time else 0.0

        logger.info(f"[API] 시스템 상태 조회 - 처리된 건수: {service_status.get('statistics', {}).get('processed_consultations', 0)}")
        
        # 상태 결정
        if service_status.get("service_initialized", False):
            system_status = "healthy"
        else:
            system_status = "degraded"
        
        # SystemStatus 기본 응답에 추가 정보 포함
        base_response = SystemStatus(
            status=system_status,
            model_loaded=service_status.get("service_initialized", False),
            model_name=service_status.get("ai_analyzer_status", {}).get("model_name", "Qwen3-4B-Instruct-2507"),
            uptime=uptime,
            processed_consultations=service_status.get("statistics", {}).get("processed_consultations", 0),
            average_processing_time=service_status.get("statistics", {}).get("average_processing_time", 0.0)
        )

        # JavaScript가 기대하는 추가 필드들을 포함한 딕셔너리 생성
        enhanced_response = base_response.model_dump()
        enhanced_response.update({
            "service_initialized": service_status.get("service_initialized", False),
            "statistics": service_status.get("statistics", {}),
            "ai_analyzer_status": service_status.get("ai_analyzer_status", {})
        })

        return enhanced_response
        
    except Exception as e:
        logger.error(f"[API] 시스템 상태 조회 오류: {e}")
        error_response = SystemStatus(
            status="error",
            model_loaded=False,
            model_name="Unknown",
            uptime=0.0,
            processed_consultations=0,
            average_processing_time=0.0
        )

        enhanced_error = error_response.model_dump()
        enhanced_error.update({
            "service_initialized": False,
            "statistics": {
                "processed_consultations": 0,
                "successful_analyses": 0,
                "failed_analyses": 0,
                "success_rate": 0.0,
                "average_processing_time": 0.0
            },
            "ai_analyzer_status": {
                "model_loaded": False,
                "model_name": "Unknown"
            }
        })

        return enhanced_error

@router.get(
    "/health",
    summary="헬스 체크",
    description="상담 분석 서비스의 헬스 체크를 수행합니다"
)
async def health_check():
    """
    상담 분석 서비스 헬스 체크
    
    Returns:
        Dict: 헬스 체크 결과
    """
    try:
        service_healthy = _consultation_service is not None and _consultation_service.is_initialized
        
        return {
            "status": "healthy" if service_healthy else "degraded",
            "service": "상담 분석 서비스",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {
                "service_initialized": service_healthy,
                "model_loaded": service_healthy
            }
        }
        
    except Exception as e:
        logger.error(f"[API] 헬스 체크 오류: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": "상담 분석 서비스",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

@router.post(
    "/initialize",
    summary="서비스 초기화",
    description="상담 분석 서비스를 강제로 초기화합니다 (관리용)"
)
async def initialize_service(background_tasks: BackgroundTasks):
    """
    서비스 강제 초기화 API (관리용)
    
    Returns:
        Dict: 초기화 결과
    """
    try:
        global _consultation_service
        
        logger.info("[API] 서비스 강제 초기화 요청")
        
        # 기존 서비스 정리
        if _consultation_service:
            await _consultation_service.cleanup()
            _consultation_service = None
        
        # 백그라운드에서 새로운 서비스 초기화
        def init_service():
            asyncio.create_task(get_consultation_service())
        
        background_tasks.add_task(init_service)
        
        return {
            "status": "initialization_started",
            "message": "서비스 초기화가 백그라운드에서 시작되었습니다",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"[API] 서비스 초기화 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"서비스 초기화 중 오류가 발생했습니다: {str(e)}"
        )

@router.post(
    "/force-reset",
    summary="강제 시스템 리셋",
    description="GPU 디바이스 충돌 등 심각한 문제 발생 시 강제로 시스템을 완전히 초기화합니다"
)
async def force_reset_service():
    """
    강제 시스템 리셋 API (비상용)
    
    GPU 메모리 문제, 디바이스 충돌 등으로 시스템이 먹통된 경우 사용
    
    Returns:
        Dict: 리셋 결과
    """
    try:
        global _consultation_service
        
        logger.warning("[API] 🚨 강제 시스템 리셋 요청")
        
        # 기존 서비스가 있으면 강제 리셋
        reset_success = False
        if _consultation_service:
            reset_success = _consultation_service.force_reset()
        
        # 서비스 인스턴스 완전 재생성
        _consultation_service = None
        
        # 추가 GPU 메모리 정리
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        logger.info("[API] 강제 시스템 리셋 완료")
        
        return {
            "success": True,
            "message": "시스템이 완전히 초기화되었습니다. 다음 요청 시 모델이 새로 로드됩니다.",
            "force_reset_applied": reset_success,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"[API] 강제 리셋 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"강제 리셋 중 오류가 발생했습니다: {str(e)}"
        )

# ========================================
# 센터링크 호환 API 엔드포인트
# ========================================

@router.post(
    "/centerlink/analyze",
    response_model=Dict[str, Any],
    summary="센터링크 호환 상담 분석",
    description="센터링크 시스템과 호환되는 상담 분석 API"
)
async def centerlink_analyze(
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """
    센터링크 호환 상담 분석 API
    
    센터링크에서 전송하는 형식에 맞춰 데이터를 처리하고
    결과를 센터링크가 기대하는 형식으로 반환합니다.
    """
    try:
        # 1. 센터링크 요청을 내부 스키마로 변환
        consultation_request = _convert_centerlink_request(request)
        
        # 2. 내부 분석 서비스 호출 (동기 호출)
        service = await get_consultation_service()
        result = service.analyze_consultation(consultation_request)
        
        # 3. 결과를 센터링크 형식으로 변환
        centerlink_response = _convert_to_centerlink_response(result, request)
        
        return centerlink_response
        
    except Exception as e:
        logger.error(f"[API] 센터링크 분석 오류: {e}")
        return {
            "consultation_id": request.get("consultation_id", "unknown"),
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

def _convert_centerlink_request(centerlink_data: Dict[str, Any]) -> ConsultationAnalysisRequest:
    """센터링크 요청을 내부 스키마로 변환"""
    from ...schemas.consultation import STTData, AnalysisOptions
    
    # STT 데이터 변환
    conversation_data = centerlink_data.get("conversation_data", {})
    stt_data = STTData(
        conversation_text=conversation_data.get("conversation_text"),
        segments=conversation_data.get("segments"),
        utterances=conversation_data.get("utterances"),
        raw_data=conversation_data.get("raw_data") or conversation_data
    )
    
    # 분석 옵션 설정
    options = AnalysisOptions(
        include_summary=True,
        include_category_recommendation=centerlink_data.get("include_categories", True),
        include_title_generation=centerlink_data.get("include_titles", True),
        max_summary_length=centerlink_data.get("max_summary_length", 300)
    )
    
    return ConsultationAnalysisRequest(
        consultation_id=centerlink_data.get("consultation_id", f"CL_{int(time.time())}"),
        consultation_content=centerlink_data.get("consultation_content", ""),
        stt_data=stt_data,
        options=options
    )

def _convert_to_centerlink_response(
    result: ConsultationAnalysisResponse, 
    original_request: Dict[str, Any]
) -> Dict[str, Any]:
    """내부 결과를 센터링크 형식으로 변환"""
    
    if result.success:
        response = {
            "consultation_id": result.consultation_id,
            "success": True,
            "analysis": {
                "summary": result.results.summary,
                "categories": [
                    {
                        "rank": cat.rank,
                        "name": cat.name,
                        "code": cat.code,
                        "confidence": cat.confidence
                    }
                    for cat in result.results.recommended_categories
                ] if result.results.recommended_categories else [],
                "titles": [
                    {
                        "title": title.title,
                        "type": title.type,
                        "confidence": title.confidence
                    }
                    for title in result.results.generated_titles
                ] if result.results.generated_titles else []
            },
            "quality": {
                "score": result.quality_metrics.quality_score if result.quality_metrics else 0.0,
                "warnings": result.quality_metrics.warnings if result.quality_metrics else []
            },
            "metadata": {
                "processing_time": result.metadata.processing_time if result.metadata else 0.0,
                "model_used": result.metadata.model_used if result.metadata else "Qwen3-4B-Instruct-2507",
                "timestamp": result.metadata.timestamp if result.metadata else datetime.now(timezone.utc).isoformat()
            }
        }
    else:
        response = {
            "consultation_id": result.consultation_id,
            "success": False,
            "error": result.error,
            "error_code": result.error_code,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    return response

# ========================================
# 로컬 파일 브라우저 API 엔드포인트
# ========================================

@router.get(
    "/local-files",
    response_model=Dict[str, Any],
    summary="로컬 call_data 폴더 파일 목록",
    description="call_data 폴더의 날짜별 폴더와 JSON 파일 목록을 반환합니다"
)
async def list_local_files():
    """call_data 폴더의 파일 목록 반환"""
    import os
    from pathlib import Path
    
    try:
        call_data_path = Path("call_data")
        
        if not call_data_path.exists():
            raise HTTPException(404, "call_data 폴더를 찾을 수 없습니다")
        
        folders = []
        
        # 날짜 폴더들 탐색
        for date_folder in sorted(call_data_path.iterdir()):
            if date_folder.is_dir() and date_folder.name.startswith('202'):
                json_files = [f.name for f in date_folder.iterdir() if f.suffix == '.json' and 'rename_map' not in f.name]
                
                folders.append({
                    "date": date_folder.name,
                    "path": str(date_folder),
                    "file_count": len(json_files),
                    "files": sorted(json_files)[:100]  # 최대 100개만 표시
                })
        
        return {
            "success": True,
            "folders": folders,
            "total_folders": len(folders)
        }
        
    except Exception as e:
        logger.error(f"[API] 로컬 파일 목록 조회 실패: {e}")
        raise HTTPException(500, f"파일 목록 조회 중 오류 발생: {str(e)}")

@router.get(
    "/local-files/{date}",
    response_model=Dict[str, Any],
    summary="날짜별 파일 목록 조회",
    description="call_data 폴더의 특정 날짜 폴더 내 JSON 파일 목록을 반환합니다"
)
async def list_files_by_date(date: str):
    """특정 날짜 폴더의 파일 목록 반환"""
    import os
    import re
    from pathlib import Path

    try:
        # 날짜 형식 검증 (YYYY-MM-DD)
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
            return {
                "success": False,
                "error": "올바른 날짜 형식이 아닙니다 (YYYY-MM-DD 형식 필요)",
                "files": []
            }
        
        # 경로 순회 공격 방지
        if '..' in date:
            return {
                "success": False,
                "error": "경로 순회 공격 시도 감지됨",
                "files": []
            }
        
        # 절대 경로로 변환하여 상대 경로 공격 방지
        base_path = Path("call_data").resolve()
        date_path = (base_path / date).resolve()
        
        # base_path 내부인지 확인
        try:
            date_path.relative_to(base_path)
        except ValueError:
            return {
                "success": False,
                "error": "접근할 수 없는 경로입니다",
                "files": []
            }

        if not date_path.exists():
            return {
                "success": False,
                "error": f"날짜 폴더를 찾을 수 없습니다: {date}",
                "files": []
            }

        if not date_path.is_dir():
            return {
                "success": False,
                "error": f"올바른 폴더가 아닙니다: {date}",
                "files": []
            }

        # JSON 파일들 찾기
        json_files = []
        for file_path in sorted(date_path.glob("*.json")):
            if file_path.is_file():
                json_files.append(file_path.name)

        logger.info(f"[API] 날짜 {date}에서 {len(json_files)}개 파일 찾음")

        return {
            "success": True,
            "date": date,
            "files": json_files,
            "total_count": len(json_files)
        }

    except Exception as e:
        logger.error(f"[API] 날짜별 파일 목록 조회 실패 ({date}): {e}")
        return {
            "success": False,
            "error": f"파일 목록 조회 중 오류 발생: {str(e)}",
            "files": []
        }

@router.get(
    "/local-files/{date}/{filename}",
    response_model=Dict[str, Any],
    summary="로컬 파일 내용 조회",
    description="call_data 폴더의 특정 JSON 파일 내용을 반환합니다"
)
async def get_local_file(date: str, filename: str):
    """특정 로컬 파일 내용 반환"""
    import os
    import json
    import re
    from pathlib import Path
    
    try:
        # 엄격한 경로 검증 함수
        def validate_file_path(date_str: str, filename_str: str) -> Path:
            """안전한 파일 경로 검증 및 생성"""
            # 날짜 형식 검증 (YYYY-MM-DD)
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="올바른 날짜 형식이 아닙니다 (YYYY-MM-DD 형식 필요)"
                )
            
            # 파일명 검증 (알파벳, 숫자, 하이픈, 언더스코어, 점만 허용)
            if not re.match(r'^[a-zA-Z0-9_\-\.]+\.json$', filename_str):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="올바른 파일명 형식이 아닙니다 (알파벳, 숫자, 하이픈, 언더스코어만 허용)"
                )
            
            # 경로 순회 공격 방지 (.. 확인)
            if '..' in date_str or '..' in filename_str:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="경로 순회 공격 시도 감지됨"
                )
            
            # 절대 경로로 변환하여 상대 경로 공격 방지
            base_path = Path("call_data").resolve()
            file_path = (base_path / date_str / filename_str).resolve()
            
            # base_path 내부인지 확인 (경로 순회 공격 차단)
            try:
                file_path.relative_to(base_path)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="접근할 수 없는 경로입니다"
                )
            
            return file_path
        
        # 경로 검증 및 파일 경로 생성
        file_path = validate_file_path(date, filename)
        
        # 파일 존재 확인
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="파일을 찾을 수 없습니다"
            )
        
        # 파일이 아닌 경우 (디렉토리 등)
        if not file_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="파일이 아닙니다"
            )
        
        # 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 응답 형태로 변환
        metadata = data.get("metadata", {})
        return {
            "success": True,
            "data": {
                "consultation_id": metadata.get("call_id", filename.replace('.json', '')),
                "consultation_content": f"로컬 파일: {filename}",
                "conversation_text": data.get("conversation_text", ""),
                "file_name": filename,
                "file_path": f"{date}/{filename}",
                "category": metadata.get("full_category_name", ""),
                "duration": metadata.get("call_duration", 0),
                "extraction_date": metadata.get("extraction_date", date)
            }
        }
        
    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="JSON 파일 형식이 올바르지 않습니다"
        )
    except Exception as e:
        logger.error(f"[API] 로컬 파일 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="파일 조회 중 오류가 발생했습니다"
        )

# ========================================
# 테스트 데이터 제공 API 엔드포인트 
# ========================================

@router.get(
    "/test-data",
    response_model=Dict[str, Any],
    summary="실제 통화 데이터 조회",
    description="call_data 폴더의 실제 통화 데이터를 랜덤으로 반환합니다"
)
async def get_test_data():
    """실제 통화 데이터를 랜덤으로 반환 - 고성능 버전"""
    import os, json, random
    
    # 직접 경로 구성 (최적화)
    call_data_dir = os.path.join("call_data", "2025-07-15")
    
    # 파일 목록 가져오기 (최소한의 검증)
    json_files = [f for f in os.listdir(call_data_dir) if f.endswith('.json')]
    if not json_files:
        raise HTTPException(404, "No data files")
    
    # 파일 읽기 및 변환 (단일 try-catch)
    selected_file = random.choice(json_files)
    with open(os.path.join(call_data_dir, selected_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 직접 매핑 (성능 최적화)
    metadata = data.get("metadata", {})
    return {
        "success": True,
        "data": {
            "consultation_id": metadata.get("call_id", "unknown"),
            "consultation_content": metadata.get("question", "실제 통화 데이터"),
            "conversation_text": data.get("conversation_text", ""),
            "file_name": selected_file,
            "category": metadata.get("full_category_name", ""),
            "duration": metadata.get("call_duration", 0)
        }
    }

# 애플리케이션 종료시 정리 작업
async def cleanup_consultation_service():
    """애플리케이션 종료시 서비스 정리"""
    global _consultation_service
    
    if _consultation_service:
        logger.info("[API] 상담 서비스 정리 시작")
        await _consultation_service.cleanup()
        _consultation_service = None
        logger.info("[API] 상담 서비스 정리 완료")