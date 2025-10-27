#!/usr/bin/env python3
"""
지정한 call_data 디렉터리의 JSON 파일들을 시간순으로 정렬해
7월 15일 샘플처럼 call_00001.json, call_00002.json ... 형태로 일괄 변경.

정렬 키 우선순위:
  1) sttList[].regDate 최소값 (가능하면 가장 신뢰도 높음)
  2) metadata.call_id의 첫 정수(에포크 초로 간주) → datetime
  3) 파일 수정시각(mtime)

사용 예시:
  # 드라이런 (미리보기)
  python scripts/rename_calls_sequential.py --dir call_data/2025-08-01 --dry-run

  # 실제 적용
  python scripts/rename_calls_sequential.py --dir call_data/2025-08-01
"""
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

from dateutil import parser as dateparser


def parse_first_datetime_from_stt(stt_list: List[Dict[str, Any]]) -> datetime | None:
    if not stt_list:
        return None
    best: datetime | None = None
    for item in stt_list:
        v = item.get("regDate") or item.get("reg_date") or item.get("time")
        if not v:
            continue
        try:
            if isinstance(v, (int, float)):
                dt = datetime.fromtimestamp(float(v))
            else:
                # 다양한 포맷 허용 (예: "2025-07-15 10:00:25")
                dt = dateparser.parse(str(v))
            if best is None or dt < best:
                best = dt
        except Exception:
            continue
    return best


def parse_dt_from_call_id(call_id: str | None) -> datetime | None:
    if not call_id:
        return None
    try:
        first = str(call_id).split(".")[0]
        ts = int(first)
        # 10자리 이상이면 에포크 초로 판단
        if ts > 10**9:
            return datetime.fromtimestamp(ts)
    except Exception:
        return None
    return None


def compute_sort_key(p: Path) -> Tuple[datetime, str]:
    # 기본값: 파일 mtime
    fallback_dt = datetime.fromtimestamp(p.stat().st_mtime)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 1) sttList.regDate 최소값
        stt = data.get("sttList") or (data.get("raw_call_data") or {}).get("details") or []
        dt = parse_first_datetime_from_stt(stt)
        if dt is not None:
            return (dt, p.name)
        # 2) metadata.call_id 추정 시간
        call_id = (data.get("metadata") or {}).get("call_id")
        dt2 = parse_dt_from_call_id(call_id)
        if dt2 is not None:
            return (dt2, p.name)
        # 3) 파일 mtime
        return (fallback_dt, p.name)
    except Exception:
        return (fallback_dt, p.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="대상 디렉터리 (예: call_data/2025-08-01)")
    ap.add_argument("--prefix", default="call_", help="파일명 접두사 (기본: call_)")
    ap.add_argument("--start", type=int, default=1, help="시작 인덱스 (기본: 1)")
    ap.add_argument("--dry-run", action="store_true", help="미리보기만 (실제 변경 없음)")
    args = ap.parse_args()

    base = Path(args.dir)
    if not base.exists():
        print(f"디렉터리 없음: {base}")
        return

    files = sorted([p for p in base.glob("*.json")])
    if not files:
        print("대상 파일이 없습니다.")
        return

    # 정렬 키 계산
    items = [(compute_sort_key(p), p) for p in files]
    items.sort(key=lambda x: x[0])

    # rename 매핑 생성
    mapping: List[Tuple[Path, Path]] = []
    idx = args.start
    for (_, _oldname), p in items:
        new_name = f"{args.prefix}{idx:05d}.json"
        mapping.append((p, p.parent / new_name))
        idx += 1

    # 충돌 방지: 이미 같은 이름이 있다면 임시 이름으로 우회
    temp_suffix = ".tmp_renaming"
    if not args.dry_run:
        # 1차: 임시 이름으로 모두 이동(중복 회피)
        temp_paths: List[Tuple[Path, Path]] = []
        for old, new in mapping:
            if old.name == new.name:
                continue
            tmp = old.with_name(old.name + temp_suffix)
            os.replace(old, tmp)
            temp_paths.append((tmp, new))
        # 2차: 최종 이름으로 이동
        for tmp, new in temp_paths:
            if new.exists():
                # 혹시 남아있다면 백업 접미사
                backup = new.with_name(new.stem + "_bak" + new.suffix)
                os.replace(new, backup)
            os.replace(tmp, new)

    # 요약 출력
    print(f"총 {len(mapping)}건 정렬 및 {'미리보기' if args.dry_run else '이름 변경'} 완료")
    # 앞 몇개만 프린트
    for old, new in mapping[:10]:
        print(f"{old.name} -> {new.name}")


if __name__ == "__main__":
    main()

