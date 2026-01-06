"""
바운드 키 인증 시스템

외부 시스템 연동을 위한 바운드 키 기반 인증 시스템
"""

import os
import hashlib
import secrets
from typing import Optional, Dict, List
from datetime import datetime, timedelta

from .logger import logger


class BoundKeyValidator:
    """바운드 키 검증기"""

    def __init__(self):
        """초기화: 환경변수에서 허용된 키 목록 로드"""
        self._load_valid_keys()

    def _load_valid_keys(self):
        """환경변수에서 유효한 바운드 키 목록 로드"""
        # 환경변수에서 쉼표로 구분된 키 목록 읽기
        keys_str = os.getenv("BOUND_KEYS", "")
        node_env = os.getenv("NODE_ENV", "").lower()

        if not keys_str:
            if node_env == "production":
                logger.error("[인증] 프로덕션 환경에서 BOUND_KEYS가 설정되지 않았습니다.")
                raise RuntimeError("BOUND_KEYS must be set in production")

            # 기본 테스트 키 (개발 환경용)
            logger.warning("[인증] BOUND_KEYS 환경변수가 설정되지 않았습니다. 기본 테스트 키를 사용합니다.")
            self.valid_keys = {
                "test_key_external_2025": {
                    "name": "외부 시스템 테스트 키",
                    "created_at": "2025-01-01",
                    "expires_at": None,
                    "permissions": ["realtime", "batch"]
                },
                "dev_key_12345678901234": {
                    "name": "개발용 테스트 키",
                    "created_at": "2025-01-01",
                    "expires_at": None,
                    "permissions": ["realtime", "batch"]
                }
            }
        else:
            # 환경변수에서 키 파싱
            self.valid_keys = {}
            for key in keys_str.split(","):
                key = key.strip()
                if len(key) >= 20:  # 최소 20자 이상
                    self.valid_keys[key] = {
                        "name": f"Key-{key[:8]}",
                        "created_at": datetime.now().isoformat(),
                        "expires_at": None,
                        "permissions": ["realtime", "batch"]
                    }

        logger.info(f"[인증] 총 {len(self.valid_keys)}개의 바운드 키 로드됨")

    def validate(self, bound_key: str) -> Dict[str, any]:
        """
        바운드 키 검증

        Args:
            bound_key: 검증할 바운드 키

        Returns:
            검증 결과 딕셔너리
            - valid: 유효성 여부
            - key_info: 키 정보 (유효한 경우)
            - error: 에러 메시지 (유효하지 않은 경우)
        """
        # 1. 기본 검증: 키 존재 여부
        if not bound_key:
            return {
                "valid": False,
                "error": "바운드 키가 제공되지 않았습니다",
                "error_code": "AUTH_KEY_MISSING"
            }

        # 2. 길이 검증: 최소 20자
        if len(bound_key) < 20:
            return {
                "valid": False,
                "error": "바운드 키 형식이 올바르지 않습니다 (최소 20자)",
                "error_code": "AUTH_KEY_INVALID_FORMAT"
            }

        # 3. 키 존재 여부 검증
        if bound_key not in self.valid_keys:
            logger.warning(f"[인증] 유효하지 않은 바운드 키 시도: {bound_key[:10]}...")
            return {
                "valid": False,
                "error": "유효하지 않은 바운드 키입니다",
                "error_code": "AUTH_KEY_INVALID"
            }

        # 4. 키 정보 조회
        key_info = self.valid_keys[bound_key]

        # 5. 만료 여부 검증
        if key_info.get("expires_at"):
            expires_at = datetime.fromisoformat(key_info["expires_at"])
            if datetime.now() > expires_at:
                return {
                    "valid": False,
                    "error": "만료된 바운드 키입니다",
                    "error_code": "AUTH_KEY_EXPIRED"
                }

        # 6. 검증 성공
        logger.info(f"[인증] 바운드 키 검증 성공: {key_info['name']}")
        return {
            "valid": True,
            "key_info": key_info
        }

    def check_permission(self, bound_key: str, permission: str) -> bool:
        """
        바운드 키의 특정 권한 확인

        Args:
            bound_key: 검증할 바운드 키
            permission: 확인할 권한 (realtime, batch, admin)

        Returns:
            권한 보유 여부
        """
        validation = self.validate(bound_key)

        if not validation["valid"]:
            return False

        key_info = validation["key_info"]
        permissions = key_info.get("permissions", [])

        return permission in permissions

    def generate_new_key(self, name: str, permissions: List[str] = None) -> str:
        """
        새로운 바운드 키 생성 (관리자용)

        Args:
            name: 키 이름
            permissions: 권한 목록

        Returns:
            생성된 바운드 키
        """
        if permissions is None:
            permissions = ["realtime", "batch"]

        # 안전한 랜덤 키 생성 (32자)
        new_key = secrets.token_urlsafe(24)  # 32자 base64 문자열

        # 키 정보 저장
        self.valid_keys[new_key] = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "expires_at": None,
            "permissions": permissions
        }

        logger.info(f"[인증] 새 바운드 키 생성: {name}")

        return new_key

    def revoke_key(self, bound_key: str) -> bool:
        """
        바운드 키 폐기 (관리자용)

        Args:
            bound_key: 폐기할 바운드 키

        Returns:
            폐기 성공 여부
        """
        if bound_key in self.valid_keys:
            del self.valid_keys[bound_key]
            logger.info(f"[인증] 바운드 키 폐기됨: {bound_key[:10]}...")
            return True

        return False

    def list_keys(self) -> List[Dict]:
        """
        모든 바운드 키 목록 조회 (관리자용)

        Returns:
            키 정보 목록
        """
        return [
            {
                "key_preview": f"{key[:10]}...",
                "name": info["name"],
                "created_at": info["created_at"],
                "expires_at": info.get("expires_at"),
                "permissions": info.get("permissions", [])
            }
            for key, info in self.valid_keys.items()
        ]


# 전역 인스턴스 (싱글톤)
_validator_instance: Optional[BoundKeyValidator] = None


def get_bound_key_validator() -> BoundKeyValidator:
    """바운드 키 검증기 인스턴스 반환 (싱글톤)"""
    global _validator_instance

    if _validator_instance is None:
        _validator_instance = BoundKeyValidator()

    return _validator_instance


def validate_bound_key(bound_key: str) -> Dict[str, any]:
    """
    바운드 키 검증 (편의 함수)

    Args:
        bound_key: 검증할 바운드 키

    Returns:
        검증 결과
    """
    validator = get_bound_key_validator()
    return validator.validate(bound_key)


def check_permission(bound_key: str, permission: str) -> bool:
    """
    바운드 키 권한 확인 (편의 함수)

    Args:
        bound_key: 검증할 바운드 키
        permission: 확인할 권한

    Returns:
        권한 보유 여부
    """
    validator = get_bound_key_validator()
    return validator.check_permission(bound_key, permission)
