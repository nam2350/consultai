from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
from pydantic import Field, AliasChoices, field_validator
import os

class ApplicationSettings(BaseSettings):
    """
    애플리케이션 설정 관리 클래스
    
    환경 변수와 기본값을 통해 애플리케이션의 모든 설정을 관리합니다.
    Pydantic BaseSettings를 상속하여 유효성 검사와 자동 타입 변환을 지원합니다.
    """
    
    # 기본 설정
    APP_NAME: str = Field(default="상담 데이터 분석 시스템", description="애플리케이션 이름")
    VERSION: str = Field(default="1.0.0", description="애플리케이션 버전")
    DEBUG: bool = Field(default=False, description="디버그 모드")
    
    # 서버 설정
    HOST: str = Field(default="0.0.0.0", description="서버 호스트")
    PORT: int = Field(default=8000, ge=1024, le=65535, description="서버 포트")
    
    # Redis 설정
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis 연결 URL")
    
    # CORS 설정
    ALLOWED_ORIGINS: List[str] = Field(default=["*"], description="허용된 오리진 목록")
    
    # UI 설정
    ENABLE_WEB_UI: bool = Field(default=True, description="웹 UI 활성화 여부")
    ENABLE_ADMIN_UI: bool = Field(default=True, description="관리자 UI 활성화 여부")
    
    # 보안 설정 - 프로덕션에서는 반드시 환경변수 사용
    SECRET_KEY: str = Field(default="dev-secret-key-for-testing-only-32chars", min_length=32, description="JWT 시크릿 키")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=1, description="액세스 토큰 만료 시간(분)")
    
    # AI 모델 설정 (이중 모델 시스템)
    MODEL_PATH: str = Field(default=r"models\Qwen3-4B", description="기본 모델 저장 경로 (하위 호환)")
    MAX_INPUT_LENGTH: int = Field(default=2048, ge=1, description="최대 입력 길이 (최적화: 4096->2048)")
    MAX_SUMMARY_LENGTH: int = Field(default=600, ge=1, description="최대 요약 길이 (상세 분석을 위해 확장: 256->600)")
    
    # Realtime processing settings
    REALTIME_MODEL_TYPE: str = Field(
        default="qwen3",
        description="실시간용 모델 타입 (qwen3)",
    )
    REALTIME_MODEL_PATH_QWEN3: str = Field(
        default=r"models\Qwen3-1.7B",
        description="Qwen3-1.7B 모델 경로",
    )
    REALTIME_TARGET_RESPONSE_TIME: float = Field(
        default=3.0,
        description="실시간 목표 응답시간(초)",
    )
    REALTIME_MAX_INPUT_LENGTH: int = Field(
        default=12000,
        description="실시간 최대 입력 길이",
    )

    # Batch processing settings (multi-model)
    BATCH_INTERVAL_MINUTES: int = Field(default=15, description="배치 처리 간격(분)")
    BATCH_PRIMARY_MODEL: str = Field(default="qwen3_4b", description="우선 배치 모델")
    BATCH_QUEUE_DB_PATH: str = Field(
        default="logs/batch_queue.db",
        description="배치 큐 상태 저장 DB 경로",
    )
    BATCH_QUEUE_MAX_RETAINED: int = Field(
        default=1000,
        ge=1,
        description="배치 큐에 보관할 최대 작업 수",
    )
    BATCH_PROCESSING_TIMEOUT_SECONDS: int = Field(
        default=3600,
        ge=60,
        description="배치 처리 중단 감지 타임아웃(초)",
    )
    BATCH_PROCESSING_MAX_ATTEMPTS: int = Field(
        default=3,
        ge=1,
        description="배치 처리 재시도 허용 횟수",
    )
    BATCH_STALE_RECOVERY_INTERVAL_SECONDS: int = Field(
        default=60,
        ge=5,
        description="배치 중단 복구 주기(초)",
    )

    # Batch model path
    BATCH_MODEL_PATH_QWEN3_4B: str = Field(
        default=r"models\Qwen3-4B",
        description="Qwen3-4B batch model path",
        validation_alias=AliasChoices("BATCH_MODEL_PATH_QWEN3_4B", "MLM_MODEL_PATH_QWEN3_4B"),
    )

    @property
    def MLM_MODEL_PATH_QWEN3_4B(self) -> str:  # Legacy alias
        return self.BATCH_MODEL_PATH_QWEN3_4B
    
    # 모델 선택 전략
    ENABLE_DYNAMIC_MODEL_SELECTION: bool = Field(default=True, description="동적 모델 선택 활성화")
    MODEL_SELECTION_STRATEGY: str = Field(default="adaptive", description="모델 선택 전략 (adaptive/round_robin/quality_first)")
    BATCH_QUALITY_THRESHOLD: float = Field(default=0.8, description="배치 품질 임계값")
    
    # 파일 업로드 설정
    MAX_FILE_SIZE: int = Field(default=100 * 1024 * 1024, ge=1024, description="최대 파일 크기(바이트)")
    UPLOAD_DIR: str = Field(default="uploads/", description="업로드 디렉토리")
    
    # 로깅 설정
    LOG_LEVEL: str = Field(default="INFO", description="로그 레벨")
    LOG_FILE: str = Field(default="logs/app.log", description="로그 파일 경로")
    
    # 성능 설정
    WORKER_PROCESSES: int = Field(default=1, ge=1, description="워커 프로세스 수")
    MAX_CONCURRENT_REQUESTS: int = Field(default=100, ge=1, description="최대 동시 요청 수")
    
    # API 설정
    API_PREFIX: str = Field(default="/api/v1", description="API 프리픽스")
    
    # 외부 AI 분석 API 설정
    EXTERNAL_API_BASE_URL: Optional[str] = Field(default=None, description="외부 API 기본 URL")
    EXTERNAL_API_KEY: Optional[str] = Field(default=None, description="외부 API 키")
    
    # 환경 설정
    NODE_ENV: str = Field(default="development", description="실행 환경")
    HF_TOKEN: Optional[str] = Field(default=None, description="HuggingFace 토큰")
    BATCH_CALLBACK_ALLOWED_HOSTS: str = Field(
        default="",
        description="배치 콜백 허용 호스트 목록(쉼표 구분, 예: api.example.com,.example.org)",
    )
    BATCH_CALLBACK_BLOCK_PRIVATE_IPS: bool = Field(
        default=True,
        description="배치 콜백에서 사설/로컬 IP 접근 차단 여부",
    )
    BATCH_CALLBACK_TIMEOUT_SECONDS: int = Field(
        default=30,
        ge=1,
        description="배치 콜백 타임아웃(초)",
        validation_alias=AliasChoices("BATCH_CALLBACK_TIMEOUT_SECONDS", "CALLBACK_TIMEOUT"),
    )
    BATCH_CALLBACK_RETRY_COUNT: int = Field(
        default=3,
        ge=0,
        description="배치 콜백 재시도 횟수",
        validation_alias=AliasChoices("BATCH_CALLBACK_RETRY_COUNT", "CALLBACK_RETRY_COUNT"),
    )
    BATCH_CALLBACK_RETRY_INTERVAL_SECONDS: int = Field(
        default=5,
        ge=0,
        description="배치 콜백 재시도 간격(초)",
        validation_alias=AliasChoices(
            "BATCH_CALLBACK_RETRY_INTERVAL_SECONDS",
            "CALLBACK_RETRY_INTERVAL",
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_assignment=True,
    )

    @field_validator('SECRET_KEY')
    def validate_secret_key_security(cls, v):
        """
        시크릿 키 보안 유효성 검사
        
        프로덕션 환경에서 기본 시크릿 키 사용을 방지합니다.
        """
        # 프로덕션 환경에서 개발용 키 사용 금지
        if os.getenv("NODE_ENV") == "production" and "dev-secret-key" in v:
            raise ValueError("프로덕션에서는 개발용 시크릿 키를 사용하지 마세요")
        return v

    @field_validator('LOG_LEVEL')
    def validate_log_level_format(cls, v):
        """
        로그 레벨 형식 유효성 검사
        
        허용된 로그 레벨인지 확인하고 대문자로 변환합니다.
        """
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f"유효하지 않은 로그 레벨: {v}. 사용 가능한 값: {valid_levels}")
        return v.upper()

    @field_validator('ALLOWED_ORIGINS')
    def validate_cors_origins_configuration(cls, v):
        """
        CORS 오리진 설정 유효성 검사
        
        와일드카드(*)와 특정 오리진을 함께 사용하는 것을 방지합니다.
        """
        if "*" in v and len(v) > 1:
            raise ValueError("와일드카드(*)와 다른 오리진을 함께 사용할 수 없습니다")
        return v

    def create_required_directories(self):
        """
        애플리케이션 실행에 필요한 디렉토리 생성
        
        로그, 업로드, 모델 저장용 디렉토리를 생성합니다.
        """
        required_directories = [
            os.path.dirname(self.LOG_FILE),
            self.UPLOAD_DIR,
            self.MODEL_PATH,
            os.path.dirname(self.BATCH_QUEUE_DB_PATH),
        ]
        for directory_path in required_directories:
            if directory_path:
                os.makedirs(directory_path, exist_ok=True)

# 싱글톤 패턴을 위한 전역 설정 인스턴스
_application_settings_instance = None

def get_application_settings() -> ApplicationSettings:
    """
    애플리케이션 설정 인스턴스 반환 (싱글톤 패턴)
    
    전역적으로 단일 설정 인스턴스를 유지하고 필요한 디렉토리를 생성합니다.
    """
    global _application_settings_instance
    if _application_settings_instance is None:
        _application_settings_instance = ApplicationSettings()
        _application_settings_instance.create_required_directories()
    return _application_settings_instance

# 하위 호환성을 위한 기본 설정 인스턴스
settings = get_application_settings()
