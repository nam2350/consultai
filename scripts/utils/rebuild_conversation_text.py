#!/usr/bin/env python3
"""
기존 call_data JSON들에서 conversation_text가 비어있는 경우
raw_call_data.details를 이용해 재구성하여 덮어쓰기.

사용 예시:
  python scripts/rebuild_conversation_text.py --dir call_data/2025-09-01 --dry-run
  python scripts/rebuild_conversation_text.py --dir call_data/2025-09-01
"""
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

from src.core.file_processor import file_processor, conversation_processor


def rebuild_file(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        convo = (data.get("conversation_text") or "").strip()
        if convo:
            return False
        details = (data.get("raw_call_data") or {}).get("details") or data.get("sttList") or []
        if not details:
            return False
        # 기존 파이프라인으로 세그먼트 변환
        segments = file_processor.load_and_convert_stt_json({"raw_call_data": {"details": details}}) or []
        if not segments:
            return False
        rebuilt = conversation_processor.construct_conversation_text(segments)
        rebuilt = (rebuilt or "").strip()
        if not rebuilt:
            return False
        data["conversation_text"] = rebuilt
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="대상 디렉터리 (예: call_data/2025-09-01)")
    parser.add_argument("--dry-run", action="store_true", help="실제 저장 없이 변경 필요 여부만 출력")
    args = parser.parse_args()

    base = Path(args.dir)
    if not base.exists():
        print(f"디렉터리 없음: {base}")
        return

    files = list(base.glob("*.json"))
    fixed = 0
    total = 0
    for p in files:
        total += 1
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            convo = (data.get("conversation_text") or "").strip()
            if convo:
                continue
            details = (data.get("raw_call_data") or {}).get("details") or data.get("sttList") or []
            if not details:
                continue
            if args.dry_run:
                fixed += 1
            else:
                if rebuild_file(p):
                    fixed += 1
        except Exception:
            continue

    mode = "(dry-run) " if args.dry_run else ""
    print(f"{mode}총 {total}건 중 재구성 대상 {fixed}건")


if __name__ == "__main__":
    main()

