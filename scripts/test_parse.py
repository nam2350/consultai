import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse batch test output files")
    parser.add_argument("path", type=Path, help="Path to a batch_test TXT output file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.path.exists():
        raise FileNotFoundError(f"File not found: {args.path}")
    content = args.path.read_text(encoding="utf-8")

    sections = content.split('=' * 60)
    print(f"Total sections: {len(sections)}\n")
    
    for i, sec in enumerate(sections):
        has_qwen = "Qwen3-4B-2507" in sec
    
        if has_qwen:
            print(f"Section {i}: Qwen={has_qwen}")
    
            # Test time extraction
            time_match = re.search(r'전체 처리시간:\s*([\d.]+)\s*초', sec)
            if time_match:
                print(f"  Time: {time_match.group(1)}")
    
            # Test quality extraction
            quality_match = re.search(r'품질 평가:\s*([\d.]+)/[\d.]+', sec)
            if quality_match:
                print(f"  Quality: {quality_match.group(1)}")
    
            # Test summary extraction
            summary_match = re.search(r'요약 \((\d+)자\):', sec)
            if summary_match:
                print(f"  Summary length: {summary_match.group(1)}")
    
            print()


if __name__ == "__main__":
    main()
