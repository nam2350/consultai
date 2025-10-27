"""Unified model download utility for SLM/LLM tiers.

This consolidates the previous download_model.py, download_slm_models.py,
and download_mlm_models.py scripts. It supports downloading any combination
of real-time SLM models or batch MLM models while working completely offline
after the artefacts are cached locally.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from huggingface_hub import snapshot_download


# Normalise stdout/stderr to UTF-8 for Windows terminals.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    repo_id: str
    tier: str  # "slm", "mlm", or "core"
    description: str
    ignore_patterns: Iterable[str] | None = None

    @property
    def local_dir(self) -> Path:
        return MODELS_ROOT / self.name


PROJECT_ROOT = Path(__file__).parent.parent
MODELS_ROOT = PROJECT_ROOT / "models"


MODEL_REGISTRY: List[ModelSpec] = [
    # Core / batch models
    ModelSpec(
        key="qwen3-4b",
        name="Qwen3-4B",
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        tier="core",
        description="주요 배치 summarizer (Qwen3-4B-Instruct-2507)",
    ),
    ModelSpec(
        key="ax-4b",
        name="A.X-4.0-Light",
        repo_id="skt/A.X-4.0-Light",
        tier="mlm",
        description="SKT 경량 4B 모델",
    ),
    ModelSpec(
        key="midm-2.0-base",
        name="Midm-2.0-Base",
        repo_id="K-intelligence/Midm-2.0-Base-Instruct",
        tier="mlm",
        description="Midm 중급 베이스",
    ),
    # Real-time SLM models
    ModelSpec(
        key="qwen3-1.7b",
        name="Qwen3-1.7B",
        repo_id="Qwen/Qwen3-1.7B",
        tier="slm",
        description="실시간 상담 보조용 SLM",
    ),
    ModelSpec(
        key="midm-2.0-mini",
        name="Midm-2.0-Mini",
        repo_id="K-intelligence/Midm-2.0-Mini-Instruct",
        tier="slm",
        description="초저지연 Midm SLM",
    ),
]


def list_models(tier: str | None = None) -> None:
    print("\n가용 모델 목록")
    print("=" * 60)
    for spec in MODEL_REGISTRY:
        if tier and spec.tier != tier:
            continue
        print(f"[{spec.tier.upper()}] {spec.key:15s} -> {spec.name} ({spec.repo_id})")
        print(f"   설명: {spec.description}")
        print(f"   경로: {spec.local_dir}")
    print("=" * 60)


def ensure_models_root() -> None:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)


def download_model(spec: ModelSpec, force: bool) -> bool:
    destination = spec.local_dir
    print("=" * 60)
    print(f"{spec.name} 모델 다운로드")
    print("=" * 60)
    print(f"Repo ID   : {spec.repo_id}")
    print(f"설치 경로 : {destination}")

    if destination.exists() and not force:
        print("[SKIP] 이미 다운로드되어 있습니다. --force 로 재다운로드 하세요.")
        return True

    destination.mkdir(parents=True, exist_ok=True)

    try:
        start_time = time.time()
        print("[DOWNLOAD] 스냅샷 다운로드 시작 ...")
        downloaded_path = snapshot_download(
            repo_id=spec.repo_id,
            local_dir=str(destination),
            local_dir_use_symlinks=False,
            resume_download=True,
            cache_dir=str(MODELS_ROOT / ".cache"),
            ignore_patterns=list(spec.ignore_patterns) if spec.ignore_patterns else None,
        )
        elapsed = time.time() - start_time

        print(f"[SUCCESS] 다운로드 완료 ({elapsed:.1f}s)")
        print(f"          {downloaded_path}")

        total_size_mb = 0.0
        print("[FILES] 주요 파일")
        for file in destination.glob("*"):
            if not file.is_file():
                continue
            size_mb = file.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            marker = "*" if file.suffix in {".safetensors", ".bin"} or file.name in {
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            } else "-"
            print(f"   {marker} {file.name}: {size_mb:.1f}MB")

        print(f"총 용량: {total_size_mb/1024:.2f}GB")
        return True

    except Exception as exc:
        print(f"[ERROR] 다운로드 실패: {exc}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="통합 모델 다운로드 스크립트")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="다운로드할 모델 키 (예: qwen3-4b midm-2.0-mini). 기본값 all",
    )
    parser.add_argument(
        "--tier",
        choices=["slm", "mlm", "core", "all"],
        default="all",
        help="특정 티어만 선택해서 다운로드",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 파일이 있어도 재다운로드",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="모델 목록만 출력하고 종료",
    )
    return parser.parse_args()


def filter_models(args: argparse.Namespace) -> List[ModelSpec]:
    if args.list:
        list_models(None if args.tier == "all" else args.tier)
        sys.exit(0)

    selected: List[ModelSpec]

    if args.models == ["all"]:
        selected = [spec for spec in MODEL_REGISTRY if args.tier in {"all", spec.tier}]
    else:
        selected = []
        for key in args.models:
            match = next((spec for spec in MODEL_REGISTRY if spec.key == key), None)
            if not match:
                print(f"[WARNING] 알 수 없는 모델 키: {key}")
                continue
            if args.tier not in {"all", match.tier}:
                print(f"[SKIP] 티어 필터({args.tier})와 맞지 않아 건너뜀: {key}")
                continue
            selected.append(match)

    if not selected:
        print("[ERROR] 다운로드할 모델이 없습니다. --list 로 가용 목록을 확인하세요.")
        sys.exit(1)

    return selected


def print_environment_info() -> None:
    print("통합 모델 다운로드 스크립트")
    print("=" * 60)
    print(f"프로젝트 경로: {PROJECT_ROOT}")
    print(f"모델 경로   : {MODELS_ROOT}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU         : {torch.cuda.get_device_name(0)} ({props.total_memory / 1024**3:.1f}GB)")
    else:
        print("GPU         : 사용 불가 (CPU 모드)")
    print("=" * 60)


def main() -> int:
    args = parse_args()
    print_environment_info()

    ensure_models_root()
    targets = filter_models(args)

    print(f"총 {len(targets)}개 모델 다운로드를 시작합니다.")
    success_count = 0
    for index, spec in enumerate(targets, 1):
        print(f"\n[{index}/{len(targets)}] {spec.name}")
        if download_model(spec, args.force):
            success_count += 1

    print("\n다운로드 결과")
    print("=" * 60)
    print(f"성공: {success_count}/{len(targets)}")
    return 0 if success_count == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())

