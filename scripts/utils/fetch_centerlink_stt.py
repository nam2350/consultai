#!/usr/bin/env python3
"""
Centerlink Smart Data API에서 STT 데이터를 조회해 프로젝트의 call_data 포맷으로 저장.

사용 예시 (PowerShell):
  $env:CENTERLINK_BASE_URL = "http://062.centerlink.kr:9104"
  $env:CENTERLINK_JWT = "<JWT 토큰>"
  python scripts/fetch_centerlink_stt.py --center ECountry --date 2025-07-14 --page 1 --pages 1 --call-ss 60

출력: call_data/<date>/<call_id>.json 파일들 생성
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import httpx
from dotenv import load_dotenv
from dateutil import parser as dateparser

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.core.file_processor import file_processor, conversation_processor
except Exception:
    print("[오류] 프로젝트 모듈 임포트 실패. 작업 디렉터리를 프로젝트 루트로 맞추세요.")
    raise


def _read_env() -> Dict[str, str]:
    load_dotenv()
    base_url = os.getenv("CENTERLINK_BASE_URL", "").strip()
    token = os.getenv("CENTERLINK_JWT", "").strip()
    if not base_url:
        raise RuntimeError("CENTERLINK_BASE_URL 환경변수가 필요합니다 (예: http://062.centerlink.kr:9104)")
    # 토큰은 선택(일부 배포는 인증 미사용 또는 쿠키 기반일 수 있음)
    return {"base_url": base_url.rstrip("/"), "token": token}


def _unwrap_response(data: Any) -> Dict[str, Any]:
    """공통 래퍼(success/message/data) 유무를 처리하여 페이로드를 반환."""
    if isinstance(data, dict) and "success" in data and "data" in data:
        if not data.get("success", False):
            raise RuntimeError(f"API 실패: {data.get('message')}")
        return data["data"] or {}
    if isinstance(data, dict):
        return data
    raise RuntimeError("알 수 없는 응답 형식")


def build_conversation_text_from_stt_list(stt_list: List[Dict[str, Any]]) -> str:
    """기존 Fallback 파이프라인과 호환되도록 conversation_text 생성."""
    # file_processor.normalize_stt_json은 raw_call_data.details를 기대하므로
    # 동일 키로 감싼 뒤 기존 유틸을 재사용한다.
    data = {"raw_call_data": {"details": stt_list}}
    segments = file_processor.load_and_convert_stt_json(data) or []
    return conversation_processor.construct_conversation_text(segments)


def row_to_internal_json(row: Dict[str, Any], extraction_date: str) -> Dict[str, Any]:
    metadata = {
        "call_id": row.get("ctiCallId") or row.get("boundUuid"),
        "cti_call_id": row.get("ctiCallId"),
        "bound_uuid": row.get("boundUuid"),
        "center_divide": row.get("centerDivide"),
        "call_category_id": row.get("callCatId"),
        "full_category_name": row.get("fullCatNm"),
        "question": "",
        "answer": "",
        "call_duration": row.get("callSs"),
        "extraction_date": extraction_date,
    }
    stt_list = row.get("sttList") or []
    conversation_text = build_conversation_text_from_stt_list(stt_list)
    return {
        "metadata": metadata,
        "conversation_text": conversation_text,
        "sttList": stt_list,
        # Fallback 파서 호환을 위한 원본 보관
        "raw_call_data": {"details": stt_list},
    }


def fetch_page(client: httpx.Client, base_url: str, token: str, *, center: str, date: str, call_ss: int, page: int, totalpage: int) -> Dict[str, Any]:
    url = f"{base_url}/study/search"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {
        "center": center,
        "date": date,
        "callSs": call_ss,
        "page": page,
        "totalpage": totalpage,
    }
    resp = client.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return _unwrap_response(resp.json())


def save_call(obj: Dict[str, Any], date: str, idx: int, filename: Optional[str] = None) -> str:
    out_dir = PROJECT_ROOT / "call_data" / date
    out_dir.mkdir(parents=True, exist_ok=True)
    if filename:
        path = out_dir / filename
    else:
        call_id = obj.get("metadata", {}).get("call_id") or f"call_{idx:05d}"
        path = out_dir / f"{call_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(path)


def _parse_first_datetime_from_stt(stt_list: List[Dict[str, Any]]) -> Optional[datetime]:
    if not stt_list:
        return None
    best = None
    for item in stt_list:
        v = item.get("regDate") or item.get("reg_date") or item.get("time")
        if v is None:
            continue
        try:
            if isinstance(v, (int, float)):
                dt = datetime.fromtimestamp(float(v))
            else:
                dt = dateparser.parse(str(v))
            if best is None or dt < best:
                best = dt
        except Exception:
            continue
    return best


def _parse_dt_from_call_id(call_id: Optional[str]) -> Optional[datetime]:
    if not call_id:
        return None
    try:
        first = str(call_id).split(".")[0]
        ts = int(first)
        if ts > 10**9:  # epoch seconds heuristic
            return datetime.fromtimestamp(ts)
    except Exception:
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Centerlink STT를 가져와 call_data 포맷으로 저장")
    parser.add_argument("--center", default="ECountry", help="센터 구분 (예: ECountry)")
    parser.add_argument("--date", required=True, help="조회 기준일 (YYYY-MM-DD)")
    parser.add_argument("--call-ss", type=int, default=60, help="통화시간(초) 필터")
    parser.add_argument("--page", type=int, default=1, help="시작 페이지")
    parser.add_argument("--pages", type=int, default=1, help="가져올 페이지 수(안전 테스트 권장: 1)")
    parser.add_argument("--first-only", action="store_true", help="첫 번째 유효 통화 1건만 저장")
    parser.add_argument("--min-stt", type=int, default=1, help="최소 STT 발화 수(유효성 필터)")
    parser.add_argument("--all", action="store_true", help="빈 페이지가 나올 때까지 모든 페이지 수집 (최대 --max-pages)")
    parser.add_argument("--max-pages", type=int, default=100, help="--all 사용 시 최대 페이지 수 한도")
    # 순차 저장 옵션
    parser.add_argument("--sequential", action="store_true", help="수집 후 시간순으로 정렬해 call_00001.json 형식으로 저장")
    parser.add_argument("--prefix", default="call_", help="--sequential 사용 시 접두사 (기본: call_)\n")
    parser.add_argument("--start-index", type=int, default=-1, help="--sequential 시작 인덱스(기본: 자동-기존 파일 최대+1)")
    args = parser.parse_args()

    env = _read_env()
    base_url, token = env["base_url"], env["token"]

    print(f"[INFO] BASE_URL={base_url}, date={args.date}, center={args.center}")

    saved = []
    seen_ids = set()
    collected: List[Dict[str, Any]] = []
    with httpx.Client() as client:
        # 페이징 전략: --all 이면 최대 --max-pages까지 진행, 아니면 --pages 횟수만큼
        page_index = 0
        while True:
            if args.all:
                if page_index >= max(1, args.max_pages):
                    break
            else:
                if page_index >= max(1, args.pages):
                    break

            page_no = args.page + page_index
            data = fetch_page(
                client,
                base_url,
                token,
                center=args.center,
                date=args.date,
                call_ss=args.call_ss,
                page=page_no,
                totalpage=(args.max_pages if args.all else args.pages),
            )
            rows = data.get("rows", [])
            print(f"[INFO] page {page_no} rows={len(rows)}")
            for idx, row in enumerate(rows, 1):
                stt = row.get("sttList") or []
                if len(stt) < args.min_stt:
                    # 불완전/빈 통화 스킵
                    continue
                obj = row_to_internal_json(row, extraction_date=args.date)
                call_id = obj.get("metadata", {}).get("call_id")
                if call_id and call_id in seen_ids:
                    continue
                if args.sequential:
                    collected.append(obj)
                else:
                    path = save_call(obj, date=args.date, idx=idx)
                    saved.append(path)
                if call_id:
                    seen_ids.add(call_id)
                if args.first_only:
                    break
            if args.first_only and saved:
                break
            # --all 모드에서 빈 페이지면 종료
            if args.all and not rows:
                break
            page_index += 1

    # 순차 저장 모드: 수집 끝난 뒤 시간순 정렬하여 저장
    if args.sequential and collected:
        # 정렬 키 계산
        def sort_key(obj: Dict[str, Any]):
            stt = obj.get("sttList") or (obj.get("raw_call_data") or {}).get("details") or []
            dt = _parse_first_datetime_from_stt(stt)
            if dt:
                return dt
            call_id = (obj.get("metadata") or {}).get("call_id")
            dt2 = _parse_dt_from_call_id(call_id)
            return dt2 or datetime.now()

        collected.sort(key=sort_key)

        # 시작 인덱스 계산
        out_dir = PROJECT_ROOT / "call_data" / args.date
        out_dir.mkdir(parents=True, exist_ok=True)
        start_idx = args.start_index
        if start_idx < 0:
            import re
            max_idx = 0
            for p in out_dir.glob("*.json"):
                m = re.match(rf"^{args.prefix}(\d+)\.json$", p.name)
                if m:
                    try:
                        n = int(m.group(1))
                        if n > max_idx:
                            max_idx = n
                    except Exception:
                        pass
            start_idx = max_idx + 1

        for i, obj in enumerate(collected, start=start_idx):
            filename = f"{args.prefix}{i:05d}.json"
            path = save_call(obj, date=args.date, idx=i, filename=filename)
            saved.append(path)

    print("[DONE] 저장 파일:")
    for p in saved:
        print(f" - {p}")


if __name__ == "__main__":
    main()
