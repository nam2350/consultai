"""Helper to switch batch (MLM) model selections.

This refresh replaces the previous copy with a cleaner output and references
the unified download script (download_models.py).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class BatchModel:
    key: str
    folder: str
    description: str

    @property
    def path(self) -> Path:
        return PROJECT_ROOT / "models" / self.folder


MODEL_REGISTRY: Dict[str, BatchModel] = {
    "ax_4b": BatchModel("ax_4b", "A.X-4.0-Light", "SKT 효율 4B"),
    "midm_base": BatchModel("midm_base", "Midm-2.0-Base", "Midm 2.0 베이스"),
    "qwen3_4b": BatchModel("qwen3_4b", "Qwen3-4B", "기본 4B LLM"),
}


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
    if not spec.path.exists():
        return False

    existing = {path.name for path in spec.path.iterdir() if path.is_file()}
    if not required_files.issubset(existing):
        return False

    weight_files = list(spec.path.glob("*.safetensors")) + list(spec.path.glob("*.bin"))
    return bool(weight_files)


def list_available_models() -> None:
    print("\n가용 배치 모델")
    print("=" * 40)
    for spec in MODEL_REGISTRY.values():
        available = check_model_availability(spec.key)
        status = "사용 가능" if available else "다운로드 필요"
        print(f"{spec.key:12s} : {spec.description}")
        print(f"  경로: {spec.path}")
        print(f"  상태: {status}\n")
    print("다운로드: python scripts/download_models.py --tier mlm --models <model_key>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="배치 처리 모델 변경")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), help="기본 모델 선택")
    parser.add_argument("--fallback", choices=list(MODEL_REGISTRY.keys()), help="백업 모델", default=None)
    parser.add_argument("--list", action="store_true", help="가용 모델 목록 출력")
    parser.add_argument("--check", choices=list(MODEL_REGISTRY.keys()), help="특정 모델 가용 여부 확인")
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
            print(f"다운로드 명령: python scripts/download_models.py --models {args.check}")
        return

    if not args.model:
        print("사용법 예시:")
        print("  python scripts/change_batch_model.py --model qwen3_4b")
        print("  python scripts/change_batch_model.py --list")
        print("  python scripts/change_batch_model.py --check qwen3_4b")
        return

    if not check_model_availability(args.model):
        print(f"모델 {args.model} 이(가) 설치되어 있지 않습니다.")
        print(f"다운로드 명령: python scripts/download_models.py --models {args.model}")
        return

    fallback = args.fallback or "qwen3_4b"
    if fallback and not check_model_availability(fallback):
        print(f"백업 모델 {fallback} 이(가) 없으므로 qwen3_4b로 대체합니다.")
        fallback = "qwen3_4b"

    update_env_file(args.model, fallback)

    print("\n설정 완료!")
    print("서버 재시작 후 새로운 배치 모델이 적용됩니다.")
    print("다운로드 스크립트: python scripts/download_models.py --tier mlm")


if __name__ == "__main__":
    main()



