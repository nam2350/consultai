"""Helper to switch batch model selections.

This refresh replaces the previous copy with a cleaner output and references
the unified download script (scripts/core/download_models.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.model_registry import BATCH_MODELS, MODEL_KEY_ALIASES, ModelSpec

ENV_FILE = PROJECT_ROOT / ".env"
MODELS_ROOT = PROJECT_ROOT / "models"
MODEL_REGISTRY = BATCH_MODELS


def normalize_model_key(value: str) -> str:
    raw = value.strip().lower()
    return MODEL_KEY_ALIASES.get(raw, raw)


def get_model_path(spec: ModelSpec) -> Path:
    return spec.local_dir(MODELS_ROOT)


def update_env_file(primary: str, fallback: str | None) -> None:
    if fallback is None:
        fallback = "qwen3_4b"

    updates = {
        "BATCH_PRIMARY_MODEL": primary,
        "BATCH_FALLBACK_MODEL": fallback,
    }

    existing_lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    new_lines = []
    handled = set()

    for line in existing_lines:
        if "=" in line and not line.strip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                handled.add(key)
                continue
        new_lines.append(line)

    for key, value in updates.items():
        if key not in handled:
            new_lines.append(f"{key}={value}")

    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    print(f".env 업데이트 완료 → {ENV_FILE}")
    print(f"  BATCH_PRIMARY_MODEL={primary}")
    print(f"  BATCH_FALLBACK_MODEL={fallback}")


def check_model_availability(model_key: str) -> bool:
    spec = MODEL_REGISTRY.get(model_key)
    if not spec:
        return False

    required_files = {"config.json", "tokenizer_config.json"}
    model_path = get_model_path(spec)
    if not model_path.exists():
        return False

    existing = {path.name for path in model_path.iterdir() if path.is_file()}
    if not required_files.issubset(existing):
        return False

    weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
    return bool(weight_files)


def list_available_models() -> None:
    print("\n가용 배치 모델")
    print("=" * 40)
    for spec in MODEL_REGISTRY.values():
        available = check_model_availability(spec.key)
        status = "사용 가능" if available else "다운로드 필요"
        description = spec.description or spec.display_name
        print(f"{spec.key:12s} : {description}")
        print(f"  경로: {get_model_path(spec)}")
        print(f"  상태: {status}\n")
    print("다운로드: python scripts/core/download_models.py --tier batch --models <model_key>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="배치 처리 모델 변경")
    parser.add_argument(
        "--model",
        type=normalize_model_key,
        choices=list(MODEL_REGISTRY.keys()),
        help="기본 모델 선택",
    )
    parser.add_argument(
        "--fallback",
        type=normalize_model_key,
        choices=list(MODEL_REGISTRY.keys()),
        help="백업 모델",
        default=None,
    )
    parser.add_argument("--list", action="store_true", help="가용 모델 목록 출력")
    parser.add_argument(
        "--check",
        type=normalize_model_key,
        choices=list(MODEL_REGISTRY.keys()),
        help="특정 모델 가용 여부 확인",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("배치 모델 설정 도우미")
    print("=" * 40)

    if args.list:
        list_available_models()
        return

    if args.check:
        available = check_model_availability(args.check)
        status = "사용 가능" if available else "다운로드 필요"
        print(f"{args.check}: {status}")
        if not available:
            print(f"다운로드 명령: python scripts/core/download_models.py --models {args.check}")
        return

    if not args.model:
        print("사용법 예시:")
        print("  python scripts/change_batch_model.py --model qwen3_4b")
        print("  python scripts/change_batch_model.py --list")
        print("  python scripts/change_batch_model.py --check qwen3_4b")
        return

    if not check_model_availability(args.model):
        print(f"모델 {args.model} 이(가) 설치되어 있지 않습니다.")
        print(f"다운로드 명령: python scripts/core/download_models.py --models {args.model}")
        return

    fallback = args.fallback or "qwen3_4b"
    if fallback and not check_model_availability(fallback):
        print(f"백업 모델 {fallback} 이(가) 없으므로 qwen3_4b로 대체합니다.")
        fallback = "qwen3_4b"

    update_env_file(args.model, fallback)

    print("\n설정 완료!")
    print("서버 재시작 후 새로운 배치 모델이 적용됩니다.")
    print("다운로드 스크립트: python scripts/core/download_models.py --tier batch")


if __name__ == "__main__":
    main()



