#!/usr/bin/env python3
"""
캐싱 시스템 롤백 스크립트

캐싱 시스템에 문제가 발생한 경우 즉시 원래 상태로 복구합니다.
Git 없이도 안전하게 롤백 가능합니다.
"""

import os
import shutil
from pathlib import Path

def rollback_cache_system():
    """캐싱 시스템을 제거하고 원래 상태로 롤백"""
    
    project_root = Path(__file__).parent.parent
    service_file = project_root / "src" / "services" / "consultation_service.py"
    backup_file = project_root / "src" / "services" / "consultation_service_backup_before_cache.py"
    
    print("캐싱 시스템 롤백 도구")
    print("=" * 40)
    
    try:
        # 백업 파일 존재 확인
        if not backup_file.exists():
            print("ERROR: 백업 파일을 찾을 수 없습니다!")
            print(f"   경로: {backup_file}")
            return False
        
        # 현재 파일을 실패 버전으로 백업
        failed_file = project_root / "src" / "services" / "consultation_service_failed_cache.py"
        if service_file.exists():
            shutil.copy2(service_file, failed_file)
            print(f"SUCCESS: 실패한 버전 백업: {failed_file.name}")
        
        # 원본 복구
        shutil.copy2(backup_file, service_file)
        print(f"SUCCESS: 원본 파일 복구 완료: {service_file.name}")
        
        # 캐시 관련 import 제거 확인
        with open(service_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'cache_manager' in content:
            print("WARNING: 파일에 캐시 관련 코드가 여전히 남아있을 수 있습니다.")
            print("   수동으로 확인해주세요.")
        else:
            print("SUCCESS: 캐시 관련 코드 완전 제거 확인")
        
        print("\n롤백 완료!")
        print("다음 단계:")
        print("1. 서버 재시작: python -m uvicorn main:app --reload")
        print("2. 기능 테스트: 웹 대시보드에서 분석 실행")
        print("3. 정상 작동 확인")
        
        return True
        
    except Exception as e:
        print(f"ERROR: 롤백 실패: {e}")
        return False

def show_cache_status():
    """현재 캐싱 시스템 상태 확인"""
    
    project_root = Path(__file__).parent.parent
    service_file = project_root / "src" / "services" / "consultation_service.py"
    
    print("캐싱 시스템 상태 확인")
    print("=" * 40)
    
    if not service_file.exists():
        print("ERROR: 서비스 파일이 없습니다!")
        return
    
    with open(service_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cache_indicators = [
        ('cache_manager import', 'cache_manager' in content),
        ('캐시 조회 코드', 'get_cached_analysis_result' in content),
        ('캐시 저장 코드', 'cache_analysis_result' in content),
        ('캐시 히트 로그', '캐시 히트' in content),
        ('처리 방식 로그', '처리 방식' in content)
    ]
    
    cache_active = any(indicator[1] for indicator in cache_indicators)
    
    print(f"캐싱 시스템 상태: {'ACTIVE' if cache_active else 'INACTIVE'}")
    print()
    
    for name, present in cache_indicators:
        status = "YES" if present else "NO"
        print(f"  {name}: {status}")
    
    print(f"\n파일 크기: {len(content):,} 문자")
    print(f"마지막 수정: {os.path.getmtime(service_file)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="캐싱 시스템 관리 도구")
    parser.add_argument("--rollback", action="store_true", help="캐싱 시스템 롤백 실행")
    parser.add_argument("--status", action="store_true", help="현재 상태 확인")
    
    args = parser.parse_args()
    
    if args.rollback:
        rollback_cache_system()
    elif args.status:
        show_cache_status()
    else:
        print("사용법:")
        print("  python scripts/rollback_cache_system.py --status    # 상태 확인")
        print("  python scripts/rollback_cache_system.py --rollback  # 롤백 실행")
