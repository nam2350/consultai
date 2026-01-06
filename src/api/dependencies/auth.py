"""
인증 관련 FastAPI 의존성
"""

from typing import Optional, Dict, Any
from fastapi import Header, HTTPException, status, Depends

from ...core.auth import validate_bound_key, check_permission
from ...core.logger import logger


async def verify_bound_key(
    x_bound_key: Optional[str] = Header(None, alias="X-Bound-Key"),
    authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    바운드 키 검증 의존성

    헤더에서 바운드 키를 추출하고 검증합니다.
    다음 2가지 방식을 지원합니다:
    1. X-Bound-Key 헤더
    2. Authorization 헤더 (Bearer 방식)

    Args:
        x_bound_key: X-Bound-Key 헤더 값
        authorization: Authorization 헤더 값

    Returns:
        키 정보 딕셔너리

    Raises:
        HTTPException: 인증 실패 시
    """
    bound_key = None

    # 1. X-Bound-Key 헤더에서 추출
    if x_bound_key:
        bound_key = x_bound_key

    # 2. Authorization 헤더에서 추출 (Bearer 방식)
    elif authorization:
        if authorization.startswith("Bearer "):
            bound_key = authorization[7:]  # "Bearer " 제거
        else:
            bound_key = authorization

    # 3. 키가 없는 경우
    if not bound_key:
        logger.warning("[인증] 바운드 키가 제공되지 않음")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "바운드 키가 제공되지 않았습니다",
                "error_code": "AUTH_KEY_MISSING",
                "hint": "X-Bound-Key 헤더 또는 Authorization 헤더를 사용하세요"
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    # 4. 바운드 키 검증
    validation = validate_bound_key(bound_key)

    if not validation["valid"]:
        logger.warning(f"[인증] 바운드 키 검증 실패: {validation.get('error')}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": validation.get("error", "인증 실패"),
                "error_code": validation.get("error_code", "AUTH_FAILED")
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    # 5. 검증 성공
    logger.debug(f"[인증] 바운드 키 검증 성공: {validation['key_info']['name']}")
    return validation["key_info"]


async def verify_realtime_permission(
    key_info: Dict[str, Any] = Depends(verify_bound_key)
) -> Dict[str, Any]:
    """
    실시간 처리 권한 검증 의존성

    Args:
        key_info: verify_bound_key에서 반환된 키 정보

    Returns:
        키 정보 딕셔너리

    Raises:
        HTTPException: 권한 없음
    """
    if key_info is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "인증 정보가 없습니다",
                "error_code": "AUTH_REQUIRED"
            }
        )

    permissions = key_info.get("permissions", [])

    if "realtime" not in permissions:
        logger.warning(f"[인증] 실시간 처리 권한 없음: {key_info['name']}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "실시간 처리 권한이 없습니다",
                "error_code": "AUTH_PERMISSION_DENIED",
                "required_permission": "realtime"
            }
        )

    return key_info


async def verify_batch_permission(
    key_info: Dict[str, Any] = Depends(verify_bound_key)
) -> Dict[str, Any]:
    """
    배치 처리 권한 검증 의존성

    Args:
        key_info: verify_bound_key에서 반환된 키 정보

    Returns:
        키 정보 딕셔너리

    Raises:
        HTTPException: 권한 없음
    """
    if key_info is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "인증 정보가 없습니다",
                "error_code": "AUTH_REQUIRED"
            }
        )

    permissions = key_info.get("permissions", [])

    if "batch" not in permissions:
        logger.warning(f"[인증] 배치 처리 권한 없음: {key_info['name']}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "배치 처리 권한이 없습니다",
                "error_code": "AUTH_PERMISSION_DENIED",
                "required_permission": "batch"
            }
        )

    return key_info


async def get_current_bound_key(
    x_bound_key: Optional[str] = Header(None, alias="X-Bound-Key"),
    authorization: Optional[str] = Header(None)
) -> str:
    """
    현재 바운드 키 문자열 반환 (검증 없음)

    Args:
        x_bound_key: X-Bound-Key 헤더 값
        authorization: Authorization 헤더 값

    Returns:
        바운드 키 문자열 (없으면 빈 문자열)
    """
    if x_bound_key:
        return x_bound_key

    if authorization:
        if authorization.startswith("Bearer "):
            return authorization[7:]
        return authorization

    return ""
