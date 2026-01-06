"""
Redis 기반 캐싱 시스템 - 분산 캐싱 및 영구 저장
"""
import json
import hashlib
import time
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
try:
    import redis
    from redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    Redis = None
from src.core.config import get_application_settings

logger = logging.getLogger(__name__)
settings = get_application_settings()

@dataclass
class CacheEntry:
    """캐시 엔트리 데이터 구조"""
    data: Any
    created_at: float
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = 0

class CacheManager:
    """Redis 기반 분산 캐싱 관리자"""
    
    def __init__(self, redis_url: str = None):
        """캐시 매니저 초기화"""
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis_client: Optional[Redis] = None
        self._local_cache: Dict[str, CacheEntry] = {}
        self._max_local_cache_size = 1000
        
        # 기본 TTL 설정 (초)
        self.default_ttl = {
            'analysis_result': 3600,      # 1시간
            'model_output': 1800,         # 30분  
            'system_prompt': 86400,       # 24시간
            'tokenizer_cache': 86400      # 24시간
        }
        
        self._setup_redis_connection()
    
    def _setup_redis_connection(self):
        """Redis 연결 설정"""
        if not REDIS_AVAILABLE:
            logger.info("[Cache] Redis 모듈 없음, 로컬 캐시만 사용")
            self._redis_client = None
            return
            
        try:
            self._redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # 연결 테스트
            self._redis_client.ping()
            logger.info("[Redis] Redis 캐시 연결 성공")
            
        except Exception as e:
            logger.warning(f"[Cache] Redis 연결 실패, 로컬 캐시 사용: {str(e)}")
            self._redis_client = None
    
    def _generate_cache_key(self, category: str, identifier: str) -> str:
        """캐시 키 생성"""
        # 해시 기반 키 생성으로 충돌 방지
        hash_obj = hashlib.md5(f"{category}:{identifier}".encode('utf-8'))
        return f"consultation_cache:{category}:{hash_obj.hexdigest()[:16]}"
    
    def get(self, category: str, identifier: str) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        cache_key = self._generate_cache_key(category, identifier)
        
        # 1. Redis에서 조회 시도
        if self._redis_client:
            try:
                cached_data = self._redis_client.get(cache_key)
                if cached_data:
                    entry_dict = json.loads(cached_data)
                    entry = CacheEntry(**entry_dict)
                    
                    # 만료 시간 확인
                    if entry.expires_at and time.time() > entry.expires_at:
                        self._redis_client.delete(cache_key)
                        return None
                    
                    # 액세스 통계 업데이트
                    entry.access_count += 1
                    entry.last_accessed = time.time()
                    self._redis_client.setex(
                        cache_key,
                        self.default_ttl.get(category, 3600),
                        json.dumps(asdict(entry))
                    )
                    
                    logger.debug(f"[Redis 캐시 히트] {category}:{identifier[:16]}...")
                    return entry.data
                    
            except Exception as e:
                logger.warning(f"Redis 조회 실패: {str(e)}")
        
        # 2. 로컬 캐시에서 조회
        if cache_key in self._local_cache:
            entry = self._local_cache[cache_key]
            
            # 만료 시간 확인
            if entry.expires_at and time.time() > entry.expires_at:
                del self._local_cache[cache_key]
                return None
            
            entry.access_count += 1
            entry.last_accessed = time.time()
            logger.debug(f"[로컬 캐시 히트] {category}:{identifier[:16]}...")
            return entry.data
        
        return None
    
    def set(self, category: str, identifier: str, data: Any, ttl: Optional[int] = None) -> bool:
        """캐시에 데이터 저장"""
        cache_key = self._generate_cache_key(category, identifier)
        ttl = ttl or self.default_ttl.get(category, 3600)
        
        current_time = time.time()
        entry = CacheEntry(
            data=data,
            created_at=current_time,
            expires_at=current_time + ttl if ttl > 0 else None,
            access_count=1,
            last_accessed=current_time
        )
        
        # 1. Redis에 저장 시도
        if self._redis_client:
            try:
                success = self._redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(asdict(entry))
                )
                if success:
                    logger.debug(f"[Redis 캐시 저장] {category}:{identifier[:16]}... (TTL: {ttl}s)")
                    return True
            except Exception as e:
                logger.warning(f"Redis 저장 실패: {str(e)}")
        
        # 2. 로컬 캐시에 저장
        self._local_cache[cache_key] = entry
        self._cleanup_local_cache()
        logger.debug(f"[로컬 캐시 저장] {category}:{identifier[:16]}... (TTL: {ttl}s)")
        return True
    
    def _cleanup_local_cache(self):
        """로컬 캐시 정리"""
        if len(self._local_cache) <= self._max_local_cache_size:
            return
        
        # LRU 방식으로 오래된 항목 제거
        sorted_items = sorted(
            self._local_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # 절반 제거
        for key, _ in sorted_items[:len(sorted_items) // 2]:
            del self._local_cache[key]
        
        logger.debug(f"[로컬 캐시 정리] 크기: {len(self._local_cache)}")
    
    def delete(self, category: str, identifier: str) -> bool:
        """캐시에서 데이터 삭제"""
        cache_key = self._generate_cache_key(category, identifier)
        
        deleted = False
        
        # Redis에서 삭제
        if self._redis_client:
            try:
                deleted = bool(self._redis_client.delete(cache_key))
            except Exception as e:
                logger.warning(f"Redis 삭제 실패: {str(e)}")
        
        # 로컬 캐시에서 삭제
        if cache_key in self._local_cache:
            del self._local_cache[cache_key]
            deleted = True
        
        return deleted
    
    def clear_category(self, category: str) -> int:
        """특정 카테고리의 모든 캐시 삭제"""
        pattern = f"consultation_cache:{category}:*"
        deleted_count = 0
        
        # Redis에서 패턴 매칭 삭제
        if self._redis_client:
            try:
                keys_to_delete = []
                for key in self._redis_client.scan_iter(pattern, count=500):
                    keys_to_delete.append(key)

                for idx in range(0, len(keys_to_delete), 500):
                    batch = keys_to_delete[idx:idx + 500]
                    deleted_count += self._redis_client.delete(*batch)
            except Exception as e:
                logger.warning(f"Redis 카테고리 삭제 실패: {str(e)}")
        
        # 로컬 캐시에서 패턴 매칭 삭제
        keys_to_delete = [
            key for key in self._local_cache.keys()
            if key.startswith(f"consultation_cache:{category}:")
        ]
        for key in keys_to_delete:
            del self._local_cache[key]
            deleted_count += 1
        
        logger.info(f"[캐시 정리] {category} 카테고리 {deleted_count}개 항목 삭제")
        return deleted_count
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        stats = {
            "redis_connected": self._redis_client is not None,
            "local_cache_size": len(self._local_cache),
            "max_local_cache_size": self._max_local_cache_size,
            "categories": {}
        }
        
        if self._redis_client:
            try:
                redis_info = self._redis_client.info('memory')
                stats["redis_memory_used"] = redis_info.get('used_memory_human', 'N/A')
                stats["redis_keys"] = self._redis_client.dbsize()
            except Exception as e:
                logger.warning(f"Redis 통계 조회 실패: {str(e)}")
                stats["redis_error"] = str(e)
        
        # 로컬 캐시 카테고리별 통계
        for key, entry in self._local_cache.items():
            category = key.split(':')[1] if ':' in key else 'unknown'
            if category not in stats["categories"]:
                stats["categories"][category] = {"count": 0, "total_access": 0}
            
            stats["categories"][category]["count"] += 1
            stats["categories"][category]["total_access"] += entry.access_count
        
        return stats

# 전역 캐시 매니저 인스턴스
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """캐시 매니저 싱글톤 인스턴스 반환"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def cache_analysis_result(text: str, max_length: int, result: Dict[str, Any]) -> bool:
    """분석 결과 캐싱"""
    identifier = f"{hashlib.md5(text.encode()).hexdigest()}:{max_length}"
    return get_cache_manager().set('analysis_result', identifier, result)

def get_cached_analysis_result(text: str, max_length: int) -> Optional[Dict[str, Any]]:
    """캐시된 분석 결과 조회"""
    identifier = f"{hashlib.md5(text.encode()).hexdigest()}:{max_length}"
    return get_cache_manager().get('analysis_result', identifier)
