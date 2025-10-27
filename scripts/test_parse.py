import re

content = open(r'C:\Workspace\product_test_app\scripts\outputs\LLM_test\2025-09-30\call_00001_ALLLLM_S_224652.txt', 'r', encoding='utf-8').read()

sections = content.split('=' * 60)
print(f"Total sections: {len(sections)}\n")

for i, sec in enumerate(sections):
    has_qwen = "Qwen3-4B-2507" in sec
    has_midm = "Midm" in sec
    has_ax = "A.X-4.0-Light" in sec

    if has_qwen or has_midm or has_ax:
        print(f"Section {i}: Qwen={has_qwen}, Midm={has_midm}, AX={has_ax}")

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
