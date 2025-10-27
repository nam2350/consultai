"""
상담 분석 API용 Pydantic 스키마 정의
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

# ========================
# STT 데이터 관련 스키마
# ========================

class STTSegment(BaseModel):
    """STT 세그먼트 데이터"""
    speaker: str = Field(..., description="화자 (고객/상담사)")
    text: str = Field(..., description="발화 내용")
    start: Optional[float] = Field(None, description="시작 시간(초)")
    end: Optional[float] = Field(None, description="종료 시간(초)")
    confidence: Optional[float] = Field(None, description="신뢰도 점수")

class STTUtterance(BaseModel):
    """STT 발화 데이터"""
    speaker: str = Field(..., description="화자 ID (SPEAKER_00, SPEAKER_01)")
    text: str = Field(..., description="발화 내용")
    confidence: Optional[float] = Field(None, description="신뢰도 점수")

class STTData(BaseModel):
    """STT 데이터 (6가지 형식 지원)"""
    conversation_text: Optional[str] = Field(None, description="완성된 대화 텍스트")
    segments: Optional[List[STTSegment]] = Field(None, description="세그먼트 형식 데이터")
    utterances: Optional[List[STTUtterance]] = Field(None, description="발화 형식 데이터")
    raw_data: Optional[Dict[str, Any]] = Field(None, description="기타 STT 형식 데이터")

# ========================
# 분석 옵션 스키마
# ========================

class AnalysisOptions(BaseModel):
    """분석 옵션 설정"""
    include_summary: bool = Field(True, description="요약 생성 포함")
    include_category_recommendation: bool = Field(True, description="카테고리 추천 포함")
    include_title_generation: bool = Field(True, description="제목 생성 포함")
    max_summary_length: int = Field(300, description="요약 최대 길이")

# ========================
# 요청 스키마
# ========================

class ConsultationAnalysisRequest(BaseModel):
    """상담 분석 요청"""
    consultation_id: str = Field(..., description="상담 고유 ID")
    consultation_content: str = Field(..., description="전체 상담 내용")
    stt_data: STTData = Field(..., description="STT 변환 데이터")

    # 선택적 메타데이터
    consultation_category_name: Optional[str] = Field(None, description="기존 상담분류명")
    consultation_category_code: Optional[str] = Field(None, description="기존 상담분류번호")
    existing_title: Optional[str] = Field(None, description="기존 제목")

    # 모델 티어 선택 (2025-09-17 추가)
    ai_tier: Optional[str] = Field("llm", description="AI 티어 (slm: 실시간용, llm: 배치용)")
    slm_model: Optional[str] = Field("qwen3", description="SLM 모델 선택 (qwen3: Qwen3-1.7B, midm: Midm-2.0-Mini)")
    llm_model: Optional[str] = Field("qwen3_4b", description="LLM 모델 선택 (qwen3_4b, midm_base, ax_light)")

    # 분석 옵션
    options: AnalysisOptions = Field(default_factory=AnalysisOptions, description="분석 옵션")

# ========================
# 응답 관련 스키마  
# ========================

class CategoryRecommendation(BaseModel):
    """카테고리 추천 결과"""
    rank: int = Field(..., description="순위 (1, 2, 3)")
    name: str = Field(..., description="추천 카테고리명")
    code: Optional[str] = Field(None, description="카테고리 코드")
    confidence: float = Field(..., description="신뢰도 점수 (0.0~1.0)")
    reason: Optional[str] = Field(None, description="추천 이유")

class GeneratedTitle(BaseModel):
    """생성된 제목"""
    title: str = Field(..., description="생성된 제목")
    type: str = Field(..., description="제목 유형 (keyword/descriptive)")
    confidence: float = Field(..., description="신뢰도 점수 (0.0~1.0)")

class AnalysisResults(BaseModel):
    """분석 결과"""
    # 기존 Qwen 요약기 결과 활용
    summary: str = Field(..., description="3줄 구조 요약 (**고객**/**상담사**/**상담결과**)")
    
    # 새로 추가되는 AI 분석 결과
    recommended_categories: List[CategoryRecommendation] = Field(default_factory=list, description="추천 카테고리 (1~3순위)")
    generated_titles: List[GeneratedTitle] = Field(default_factory=list, description="생성된 제목들")

class QualityMetrics(BaseModel):
    """품질 평가 지표"""
    quality_score: float = Field(..., description="전체 품질 점수 (0.0~1.0)")
    summary_completeness: Optional[float] = Field(None, description="요약 완성도")
    category_accuracy: Optional[float] = Field(None, description="카테고리 정확도") 
    title_relevance: Optional[float] = Field(None, description="제목 관련성")
    warnings: List[str] = Field(default_factory=list, description="품질 경고사항")

class ProcessingMetadata(BaseModel):
    """처리 메타데이터"""
    processing_time: float = Field(..., description="처리 시간(초)")
    model_used: str = Field("Qwen3-4B-Instruct-2507", description="사용된 모델")
    timestamp: str = Field(..., description="처리 완료 시각 (ISO 8601)")
    
    model_config = {"protected_namespaces": ()}

# ========================
# 메인 응답 스키마
# ========================

class ConsultationAnalysisResponse(BaseModel):
    """상담 분석 응답"""
    consultation_id: str = Field(..., description="상담 고유 ID")
    success: bool = Field(..., description="분석 성공 여부")
    
    # 분석 결과 (성공시에만)
    results: Optional[AnalysisResults] = Field(None, description="분석 결과")
    quality_metrics: Optional[QualityMetrics] = Field(None, description="품질 평가")
    metadata: Optional[ProcessingMetadata] = Field(None, description="처리 정보")
    
    # 에러 정보 (실패시에만)
    error: Optional[str] = Field(None, description="에러 메시지")
    error_code: Optional[str] = Field(None, description="에러 코드")

# ========================
# 배치 처리 스키마
# ========================

class BatchAnalysisRequest(BaseModel):
    """배치 분석 요청"""
    consultation_requests: List[ConsultationAnalysisRequest] = Field(..., description="분석할 상담 목록")
    batch_options: Optional[Dict[str, Any]] = Field(None, description="배치 처리 옵션")

class BatchAnalysisResponse(BaseModel):
    """배치 분석 응답"""
    batch_id: Optional[str] = Field(None, description="배치 ID")
    total_count: int = Field(..., description="총 요청 수")
    success_count: int = Field(..., description="성공 처리 수")
    failed_count: int = Field(..., description="실패 처리 수")
    results: List[ConsultationAnalysisResponse] = Field(..., description="개별 분석 결과")
    total_processing_time: float = Field(..., description="전체 처리 시간(초)")

# ========================
# 상태 조회 스키마
# ========================

class SystemStatus(BaseModel):
    """시스템 상태 정보"""
    status: str = Field(..., description="시스템 상태 (healthy/degraded/down)")
    model_loaded: bool = Field(..., description="모델 로드 상태")
    model_name: str = Field(..., description="현재 로드된 모델명")
    uptime: float = Field(..., description="가동 시간(초)")
    processed_consultations: int = Field(0, description="처리된 상담 수")
    average_processing_time: float = Field(0.0, description="평균 처리 시간(초)")
    
    model_config = {"protected_namespaces": ()}

# ========================
# 실시간 처리 스키마 (센터링크 연동용)
# ========================

class RealtimeAnalysisRequest(BaseModel):
    """실시간 상담 분석 요청 (SLM 전용)"""
    bound_key: str = Field(..., description="센터링크 바운드 키")
    consultation_id: str = Field(..., description="상담 고유 ID")
    stt_data: STTData = Field(..., description="STT 변환 데이터")

    # 선택적 메타데이터
    metadata: Optional[Dict[str, Any]] = Field(None, description="추가 메타데이터")

class RealtimeAnalysisResponse(BaseModel):
    """실시간 상담 분석 응답 (1-3초 목표)"""
    success: bool = Field(..., description="분석 성공 여부")
    consultation_id: str = Field(..., description="상담 고유 ID")

    # 간략 요약 (SLM 생성)
    summary: Optional[str] = Field(None, description="3줄 구조 간략 요약")

    # 처리 정보
    processing_time: float = Field(..., description="처리 시간(초)")
    model: str = Field(..., description="사용된 SLM 모델")
    timestamp: str = Field(..., description="처리 완료 시각 (ISO 8601)")

    # 에러 정보 (실패시)
    error: Optional[str] = Field(None, description="에러 메시지")
    error_code: Optional[str] = Field(None, description="에러 코드")

# ========================
# 비동기 배치 처리 스키마 (센터링크 연동용)
# ========================

class AsyncBatchAnalysisRequest(BaseModel):
    """비동기 배치 분석 요청 (LLM 전용)"""
    bound_key: str = Field(..., description="센터링크 바운드 키")
    batch_id: str = Field(..., description="배치 고유 ID")
    consultations: List[Dict[str, Any]] = Field(..., description="상담 데이터 목록 (최대 20개)")
    callback_url: str = Field(..., description="결과 전송할 콜백 URL")

    # 선택적 설정
    llm_model: Optional[str] = Field("qwen3_4b", description="LLM 모델 (qwen3_4b, ax_light)")
    priority: Optional[int] = Field(1, description="우선순위 (1: 높음, 2: 중간, 3: 낮음)")

class AsyncBatchAnalysisResponse(BaseModel):
    """비동기 배치 분석 응답 (즉시 반환)"""
    success: bool = Field(..., description="요청 접수 성공 여부")
    batch_id: str = Field(..., description="배치 고유 ID")
    status: str = Field(..., description="배치 상태 (queued/processing/completed/failed)")
    total_count: int = Field(..., description="총 상담 수")
    estimated_completion_time: Optional[float] = Field(None, description="예상 완료 시간(초)")
    callback_url: str = Field(..., description="결과 전송 URL")

    # 에러 정보
    error: Optional[str] = Field(None, description="에러 메시지")
    error_code: Optional[str] = Field(None, description="에러 코드")

class BatchCallbackResult(BaseModel):
    """배치 처리 완료 후 센터링크로 전송할 결과"""
    batch_id: str = Field(..., description="배치 고유 ID")
    status: str = Field(..., description="배치 상태 (completed/failed)")
    total_count: int = Field(..., description="총 상담 수")
    success_count: int = Field(..., description="성공 처리 수")
    failed_count: int = Field(..., description="실패 처리 수")

    # 개별 결과
    results: List[Dict[str, Any]] = Field(..., description="상담별 분석 결과")

    # 처리 정보
    total_processing_time: float = Field(..., description="전체 처리 시간(초)")
    completed_at: str = Field(..., description="완료 시각 (ISO 8601)")

# ========================
# 에러 응답 스키마
# ========================

class ErrorDetail(BaseModel):
    """에러 상세 정보"""
    code: str = Field(..., description="에러 코드")
    message: str = Field(..., description="에러 메시지")
    details: Optional[Dict[str, Any]] = Field(None, description="추가 에러 정보")

class ErrorResponse(BaseModel):
    """에러 응답"""
    success: bool = Field(False, description="처리 성공 여부")
    error: ErrorDetail = Field(..., description="에러 정보")
    timestamp: str = Field(..., description="에러 발생 시각")

# ========================
# 모델 Export (FastAPI에서 사용)
# ========================

__all__ = [
    # 요청 스키마
    "ConsultationAnalysisRequest",
    "BatchAnalysisRequest",
    "RealtimeAnalysisRequest",
    "AsyncBatchAnalysisRequest",
    "STTData",
    "AnalysisOptions",

    # 응답 스키마
    "ConsultationAnalysisResponse",
    "BatchAnalysisResponse",
    "RealtimeAnalysisResponse",
    "AsyncBatchAnalysisResponse",
    "BatchCallbackResult",
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
