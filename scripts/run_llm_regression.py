#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick regression runner for Midm-2.0-Base and A.X-4.0-Light."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.file_processor import load_conversation_from_json
from src.core.models.ax_light.summarizer import AXLightSummarizer
from src.core.models.midm_base.summarizer import MidmBaseSummarizer
from src.core.quality_validator import quality_validator

MODEL_FACTORIES: Tuple[Tuple[str, Callable[[], object]], ...] = (
    (
        "Midm-2.0-Base-Instruct",
        lambda: MidmBaseSummarizer(str(PROJECT_ROOT / "models" / "Midm-2.0-Base")),
    ),
    (
        "A.X-4.0-Light",
        lambda: AXLightSummarizer(str(PROJECT_ROOT / "models" / "A.X-4.0-Light")),
    ),
)


def collect_json_calls(root: Path, count: int) -> List[Path]:
    files: List[Path] = []
    if not root.exists():
        raise FileNotFoundError(f"call_data root not found: {root}")

    date_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith("202")]
    )
    for date_dir in date_dirs:
        for json_file in sorted(date_dir.glob("*.json")):
            if "rename_map" in json_file.name:
                continue
            files.append(json_file)
            if len(files) >= count:
                return files
    return files


def evaluate_model(model_name: str, factory: Callable[[], object], files: Iterable[Path]):
    summarizer = factory()
    if not summarizer.load_model():
        raise RuntimeError(f"Failed to load {model_name}")

    total_files = 0
    success = 0
    total_time = 0.0
    total_quality = 0.0
    warning_hist: dict[str, int] = {}
    per_file = []

    start = time.time()
    for json_path in files:
        total_files += 1
        conversation = load_conversation_from_json(str(json_path))
        file_entry = {
            "file": json_path.name,
            "path": str(json_path),
            "success": False,
            "processing_time": 0.0,
            "quality_score": None,
            "warnings": [],
            "error": "",
        }

        try:
            result = summarizer.summarize_consultation(conversation)
        except Exception as exc:  # pragma: no cover - defensive
            result = {
                "success": False,
                "summary": "",
                "processing_time": 0.0,
                "error": str(exc),
            }

        file_entry["processing_time"] = float(result.get("processing_time", 0.0))
        file_entry["error"] = result.get("error", "")
        summary_text = result.get("summary", "")
        is_success = bool(result.get("success") and summary_text)

        if is_success:
            qres = quality_validator.validate_summary(summary_text, conversation)
            file_entry["quality_score"] = qres.get("quality_score")
            file_entry["warnings"] = qres.get("warnings", [])
            file_entry["semantic"] = qres.get("semantic_check", {})

            if not qres.get("is_acceptable", True):
                is_success = False
                reason = "; ".join(file_entry["warnings"]) or "요약 품질 기준 미달"
                file_entry["error"] = reason

        file_entry["success"] = bool(is_success)

        if is_success:
            success += 1
            total_time += file_entry["processing_time"]
            if file_entry["quality_score"] is not None:
                total_quality += float(file_entry["quality_score"])
            for warn in file_entry["warnings"]:
                if warn:
                    warning_hist[warn] = warning_hist.get(warn, 0) + 1
        per_file.append(file_entry)

    duration = time.time() - start
    summarizer.cleanup()

    summary = {
        "model": model_name,
        "total_files": total_files,
        "success": success,
        "fail": total_files - success,
        "avg_quality": round(total_quality / success, 4) if success else 0.0,
        "avg_time": round(total_time / success, 2) if success else 0.0,
        "duration": round(duration, 2),
        "warnings": warning_hist,
    }
    return summary, per_file


def main():
    parser = argparse.ArgumentParser(description="Run regression checks for LLM models.")
    parser.add_argument("--count", type=int, default=50, help="Number of call records to process")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "outputs" / "LLM_regression",
        help="Directory to store regression artifacts",
    )
    args = parser.parse_args()

    data_root = PROJECT_ROOT / "call_data"
    files = collect_json_calls(data_root, args.count)
    if not files:
        raise RuntimeError("No call_data JSON files available for regression")
    if len(files) < args.count:
        print(f"[warning] only found {len(files)} files; requested {args.count}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    summaries = []
    print("Running regression across", len(files), "calls...")

    for name, factory in MODEL_FACTORIES:
        print(f"\n=== {name} ===")
        summary, detail = evaluate_model(name, factory, files)
        summaries.append(summary)
        results[name] = detail
        print(
            f"  success: {summary['success']}/{summary['total_files']}, "
            f"avg_quality={summary['avg_quality']}, avg_time={summary['avg_time']}s"
        )

    output_payload = {
        "timestamp": timestamp,
        "count_requested": args.count,
        "file_count": len(files),
        "files": [str(p) for p in files],
        "summaries": summaries,
        "details": results,
    }

    output_path = args.output_dir / f"llm_regression_{timestamp}.json"
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved regression report to {output_path}")


if __name__ == "__main__":
    main()
