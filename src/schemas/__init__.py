"""
API 스키마 모듈
"""

from .consultation import (
    # 요청 스키마
    ConsultationAnalysisRequest,
    BatchAnalysisRequest, 
    STTData,
    AnalysisOptions,
    
    # 응답 스키마
    ConsultationAnalysisResponse,
    BatchAnalysisResponse,
    AnalysisResults,
    CategoryRecommendation,
    GeneratedTitle,
    QualityMetrics,
    ProcessingMetadata,
    
    # 상태 및 에러
    SystemStatus,
    ErrorResponse,
    ErrorDetail
)

__all__ = [
    # 요청 스키마
    "ConsultationAnalysisRequest",
    "BatchAnalysisRequest", 
    "STTData",
    "AnalysisOptions",
    
    # 응답 스키마
    "ConsultationAnalysisResponse",
    "BatchAnalysisResponse", 
    "AnalysisResults",
    "CategoryRecommendation",
    "GeneratedTitle",
    "QualityMetrics",
    "ProcessingMetadata",
    
    # 상태 및 에러
    "SystemStatus",
    "ErrorResponse",
    "ErrorDetail"
]