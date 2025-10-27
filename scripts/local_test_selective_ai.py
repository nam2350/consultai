#!/usr/bin/env python3
"""
AI 분석 선택적 테스트 스크립트 - 웹 대시보드처럼 기능을 선택해서 테스트
요약, 키워드 추천, 제목 생성을 개별적으로 선택 가능
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# UTF-8 인코딩 설정 (Windows 콘솔 호환성)
import sys
if sys.platform == 'win32':
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
    except:
        pass
    # Windows에서 이모지 출력 문제를 해결하기 위한 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
else:
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 필수 모듈들 import
try:
    # LLM 모델들 (Large Language Models - 배치 처리용)
    from src.core.models.qwen3_4b.summarizer import Qwen2507Summarizer
    from src.core.models.qwen3_4b.classifier import CategoryClassifier
    from src.core.models.qwen3_4b.title_generator import TitleGenerator

    # 모든 LLM 모델들 (대용량 언어 모델)
    from src.core.models.midm_base.summarizer import MidmBaseSummarizer
    from src.core.models.midm_base.classifier import CategoryClassifier as Midm_Base_Classifier
    from src.core.models.midm_base.title_generator import TitleGenerator as Midm_Base_TitleGenerator
    from src.core.models.ax_light.summarizer import AXLightSummarizer
    from src.core.models.ax_light.classifier import CategoryClassifier as AX_Light_Classifier
    from src.core.models.ax_light.title_generator import TitleGenerator as AX_Light_TitleGenerator

    # SLM 모델들 (Small Language Models - 실시간 처리용)
    from src.core.models.qwen3_1_7b.summarizer import Qwen3Summarizer
    from src.core.models.midm_mini.summarizer import MidmSummarizer

    from src.core.file_processor import load_conversation_from_json
    from src.core.quality_validator import quality_validator
except ImportError as e:
    print(f"[오류] 필수 모듈 import 실패: {e}")
    print("모든 필수 모듈이 정상적으로 로드되어야 합니다.")
    sys.exit(1)

def print_banner(model_tier: str = 'llm'):
    """프로그램 배너 출력"""
    print("=" * 70)
    print("    AI 상담 분석 선택적 테스트 도구 v1.0.0")
    if model_tier == 'slm':
        print("    ?? SLM 모드: 실시간 처리용 (요약만 지원)")
    elif model_tier == 'all_llm':
        print("    ?? ALL-LLM 모드: 모든 LLM 모델 비교 테스트")
    else:
        print("    ?? LLM 모드: 배치 처리용 (요약, 키워드, 제목)")
    print("=" * 70)

# LLM 모델 설정 (모든 대용량 언어 모델)
LLM_MODELS = {
    'qwen3_4b': {
        'name': 'Qwen3-4B-2507',
        'path': 'models/Qwen3-4B',
        'summarizer_class': Qwen2507Summarizer,
        'classifier_class': CategoryClassifier,
        'title_generator_class': TitleGenerator
    },
    'midm_base': {
        'name': 'Midm-2.0-Base-Instruct',
        'path': 'models/Midm-2.0-Base',
        'summarizer_class': MidmBaseSummarizer,
        'classifier_class': Midm_Base_Classifier,
        'title_generator_class': Midm_Base_TitleGenerator
    },
    'ax_light': {
        'name': 'A.X-4.0-Light',
        'path': 'models/A.X-4.0-Light',
        'summarizer_class': AXLightSummarizer,
        'classifier_class': AX_Light_Classifier,
        'title_generator_class': AX_Light_TitleGenerator
    },
}

def get_date_folders(data_dir: Path) -> List[str]:
    """call_data 디렉토리에서 날짜 폴더 목록을 가져옴 (정렬된 상태)"""
    date_folders = []
    for folder in data_dir.iterdir():
        if folder.is_dir() and folder.name.startswith('202'):  # 2020년대 폴더만
            date_folders.append(folder.name)
    return sorted(date_folders)

def get_files_by_date_sequence(data_dir: Path, date_sequence: List[str], count: int = None) -> List[str]:
    """날짜 순서대로 파일 목록을 생성"""
    files_to_process = []
    
    for date in date_sequence:
        date_folder = data_dir / date
        if not date_folder.exists():
            print(f"[경고] 날짜 폴더를 찾을 수 없습니다: {date}")
            continue
        
        print(f"[처리중] {date} 폴더 처리 중...")
        date_files = []
        for json_file in sorted(date_folder.glob('*.json')):
            if 'rename_map' not in json_file.name:
                date_files.append(str(json_file))
        
        print(f"   → {len(date_files)}개 파일 발견")
        files_to_process.extend(date_files)
        
        # count 제한이 있으면 확인
        if count and len(files_to_process) >= count:
            files_to_process = files_to_process[:count]
            break
    
    return files_to_process

def parse_date_sequence(date_sequence_arg: str, data_dir: Path, start_date: str = None) -> List[str]:
    """날짜 순서 인자를 파싱하여 날짜 목록 반환"""
    if date_sequence_arg == 'auto':
        # 자동으로 모든 날짜 폴더를 순서대로
        dates = get_date_folders(data_dir)
        if start_date:
            try:
                start_idx = dates.index(start_date)
                dates = dates[start_idx:]
            except ValueError:
                print(f"[경고] 시작 날짜를 찾을 수 없습니다: {start_date}")
        return dates
    else:
        # 사용자 지정 날짜 순서
        return [date.strip() for date in date_sequence_arg.split(',')]

def get_user_selection() -> Dict[str, bool]:
    """사용자로부터 테스트할 기능 선택받기 (인터랙티브 모드)"""
    print("\n테스트할 기능을 선택하세요:")
    print("-" * 40)
    
    selections = {}
    
    # 요약 선택
    while True:
        answer = input("1. 요약 생성을 테스트하시겠습니까? (y/n) [기본: y]: ").strip().lower()
        if answer in ['', 'y', 'yes']:
            selections['summary'] = True
            break
        elif answer in ['n', 'no']:
            selections['summary'] = False
            break
        else:
            print("   ??  y 또는 n을 입력해주세요.")
    
    # 키워드 추천 선택
    while True:
        answer = input("2. 키워드 추천을 테스트하시겠습니까? (y/n) [기본: y]: ").strip().lower()
        if answer in ['', 'y', 'yes']:
            selections['keywords'] = True
            break
        elif answer in ['n', 'no']:
            selections['keywords'] = False
            break
        else:
            print("   ??  y 또는 n을 입력해주세요.")
    
    # 제목 생성 선택
    while True:
        answer = input("3. 제목 생성을 테스트하시겠습니까? (y/n) [기본: y]: ").strip().lower()
        if answer in ['', 'y', 'yes']:
            selections['titles'] = True
            break
        elif answer in ['n', 'no']:
            selections['titles'] = False
            break
        else:
            print("   ??  y 또는 n을 입력해주세요.")
    
    # 선택 확인
    print("\n" + "=" * 40)
    print("선택된 기능:")
    if selections['summary']:
        print("  + 요약 생성")
    if selections['keywords']:
        print("  + 키워드 추천")
    if selections['titles']:
        print("  + 제목 생성")
    
    if not any(selections.values()):
        print("  ??  아무 기능도 선택되지 않았습니다.")
        return selections
    
    print("=" * 40)
    return selections

def parse_feature_args(args) -> Dict[str, bool]:
    """커맨드라인 인자에서 기능 선택 파싱"""
    selections = {
        'summary': True,  # 기본값
        'keywords': True,  # 기본값
        'titles': True     # 기본값
    }
    
    # --features 인자 처리
    if args.features:
        # 모든 기능 비활성화 후 선택된 것만 활성화
        selections = {
            'summary': False,
            'keywords': False,
            'titles': False
        }
        
        features = args.features.lower().split(',')
        for feature in features:
            feature = feature.strip()
            if feature in ['summary', 's', '요약']:
                selections['summary'] = True
            elif feature in ['keywords', 'k', '키워드']:
                selections['keywords'] = True
            elif feature in ['titles', 't', '제목']:
                selections['titles'] = True
            elif feature in ['all', 'a', '전체']:
                selections = {'summary': True, 'keywords': True, 'titles': True}
                break
    
    # 개별 플래그 처리 (우선순위 높음)
    if args.no_summary:
        selections['summary'] = False
    if args.no_keywords:
        selections['keywords'] = False
    if args.no_titles:
        selections['titles'] = False
    
    if args.only_summary:
        selections = {'summary': True, 'keywords': False, 'titles': False}
    elif args.only_keywords:
        selections = {'summary': False, 'keywords': True, 'titles': False}
    elif args.only_titles:
        selections = {'summary': False, 'keywords': False, 'titles': True}
    
    return selections

def process_file_with_both_slm_models(file_path: str, selections: Dict[str, bool], output_dir: Path, idx: int, total: int) -> List[Dict[str, Any]]:
    """SLM 양쪽 모델로 파일 처리 (Qwen3-1.7B와 Midm-Mini)"""
    results = []

    # Qwen3-1.7B 모델로 처리
    print(f"  [1/2] Qwen3-1.7B 모델로 처리 중...")
    qwen3_model_path = str(Path(__file__).parent.parent / "models" / "Qwen3-1.7B")
    qwen3_summarizer = Qwen3Summarizer(qwen3_model_path)

    qwen3_result = None
    if qwen3_summarizer.load_model():
        qwen3_result = process_file(file_path, qwen3_summarizer, None, None, selections)
        qwen3_result['model_name'] = "Qwen3-1.7B"
        qwen3_result['model_type'] = "qwen3"
        results.append(qwen3_result)

        if qwen3_result['success']:
            print(f"     [OK] Qwen3-1.7B 처리 완료 ({qwen3_result['processing_time']:.2f}초)")
        else:
            print(f"     [FAIL] Qwen3-1.7B 처리 실패: {qwen3_result.get('error', 'Unknown error')}")

        qwen3_summarizer.cleanup()
    else:
        print(f"     [FAIL] Qwen3-1.7B 모델 로드 실패")
        qwen3_result = {'success': False, 'error': 'Qwen3-1.7B 모델 로드 실패', 'model_name': 'Qwen3-1.7B', 'model_type': 'qwen3'}
        results.append(qwen3_result)

    # Midm-Mini 모델로 처리
    print(f"  [2/2] Midm-Mini 모델로 처리 중...")
    midm_model_path = str(Path(__file__).parent.parent / "models" / "Midm-2.0-Mini")
    midm_summarizer = MidmSummarizer(midm_model_path)

    midm_result = None
    if midm_summarizer.load_model():
        midm_result = process_file(file_path, midm_summarizer, None, None, selections)
        midm_result['model_name'] = "Midm-2.0-Mini"
        midm_result['model_type'] = "midm"
        results.append(midm_result)

        if midm_result['success']:
            print(f"     [OK] Midm-Mini 처리 완료 ({midm_result['processing_time']:.2f}초)")
        else:
            print(f"     [FAIL] Midm-Mini 처리 실패: {midm_result.get('error', 'Unknown error')}")

        midm_summarizer.cleanup()
    else:
        print(f"     [FAIL] Midm-Mini 모델 로드 실패")
        midm_result = {'success': False, 'error': 'Midm-Mini 모델 로드 실패', 'model_name': 'Midm-2.0-Mini', 'model_type': 'midm'}
        results.append(midm_result)

    # 두 모델 결과를 하나의 파일로 저장
    if qwen3_result and midm_result:
        save_dual_model_result(qwen3_result, midm_result, selections, output_dir, idx, total)
        print(f"     >> 듀얼 모델 결과 저장 완료 (1개 통합 파일)")

    return results

def process_file_with_all_llm_models(file_path: str, selections: Dict[str, bool], output_dir: Path, idx: int, total: int) -> List[Dict[str, Any]]:
    """모든 LLM 모델로 파일 처리"""
    print(f"\n?? 파일 [{idx}/{total}]: {os.path.basename(file_path)}")
    print("=" * 60)

    all_results = []
    successful_results = []

    # 모든 LLM 모델로 순차 처리
    for model_id, model_config in LLM_MODELS.items():
        print(f"\n?? [{len(successful_results)+1}/{len(LLM_MODELS)}] {model_config['name']} 모델 처리 중...")

        try:
            # 모델 경로 설정
            model_path = str(Path(__file__).parent.parent / model_config['path'])

            # Summarizer 로드
            summarizer = model_config['summarizer_class'](model_path)

            if summarizer.load_model():
                # Classifier와 TitleGenerator 초기화 (공유 모델 사용)
                classifier = model_config['classifier_class'](shared_summarizer=summarizer)
                title_generator = model_config['title_generator_class'](shared_summarizer=summarizer)

                # 파일 처리
                result = process_file(file_path, summarizer, classifier, title_generator, selections)
                result['model_name'] = model_config['name']
                result['model_id'] = model_id

                if result['success']:
                    print(f"     ? {model_config['name']} 처리 완료 ({result['processing_time']:.2f}초)")
                    successful_results.append(result)
                else:
                    print(f"     ? {model_config['name']} 처리 실패: {result.get('error', 'Unknown error')}")

                all_results.append(result)

                # 메모리 정리
                summarizer.cleanup()

            else:
                print(f"     ? {model_config['name']} 모델 로드 실패")
                failed_result = {
                    'success': False,
                    'error': f'{model_config["name"]} 모델 로드 실패',
                    'model_name': model_config['name'],
                    'model_id': model_id,
                    'file': os.path.basename(file_path)
                }
                all_results.append(failed_result)

        except Exception as e:
            print(f"     ? {model_config['name']} 처리 중 오류: {e}")
            error_result = {
                'success': False,
                'error': f'{model_config["name"]} 처리 중 오류: {e}',
                'model_name': model_config['name'],
                'model_id': model_id,
                'file': os.path.basename(file_path)
            }
            all_results.append(error_result)

    # 모든 모델 결과를 하나의 파일로 저장
    if successful_results:
        save_all_llm_results(successful_results, selections, output_dir, idx, total)
        print(f"\n?? 모든 LLM 모델 결과 저장 완료 ({len(successful_results)}/{len(LLM_MODELS)}개 성공)")
    else:
        print(f"\n? 모든 LLM 모델 처리 실패")

    return all_results

def save_all_llm_results(results: List[Dict[str, Any]], selections: Dict[str, bool], output_dir: Path, idx: int, total: int):
    """모든 LLM 모델의 결과를 하나의 파일에 저장"""
    if not results:
        return

    # 선택된 기능 표시
    features = []
    if selections['summary']:
        features.append('S')
    if selections['keywords']:
        features.append('K')
    if selections['titles']:
        features.append('T')
    feature_str = ''.join(features) if features else 'NONE'

    # 파일명 생성
    base_name = results[0]['file'].replace('.json', '')
    file_timestamp = datetime.now().strftime('%H%M%S')
    all_llm_file = output_dir / f"{base_name}_ALLLLM_{feature_str}_{file_timestamp}.txt"

    # 파일 저장
    with open(all_llm_file, 'w', encoding='utf-8') as f:
        f.write(f"?? ALL-LLM 모드: 모든 LLM 모델 비교 결과\n")
        f.write(f"=" * 80 + "\n")
        f.write(f"?? 파일명: {results[0]['file']}\n")
        f.write(f"?? 처리 순서: [{idx}/{total}]\n")
        f.write(f"? 처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"?? 테스트 기능: {', '.join([k for k, v in selections.items() if v])}\n")
        f.write(f"? 성공 모델: {len(results)}개\n")
        f.write(f"\n" + "=" * 80 + "\n")

        # 각 모델별 결과 출력
        for i, result in enumerate(results, 1):
            f.write(f"\n?? [{i}/{len(results)}] {result['model_name']}\n")
            f.write("=" * 60 + "\n")
            f.write(f"??  전체 처리시간: {result['processing_time']:.2f}초\n")

            # AI 요약 시간 정보
            if 'ai_summary_time' in result and result['ai_summary_time'] > 0:
                f.write(f"?? AI 요약시간: {result['ai_summary_time']:.2f}초\n")

            # 대화 길이 정보
            if 'conversation_length' in result:
                f.write(f"?? 대화 길이: {result['conversation_length']}자\n")

            f.write(f"\n")

            # 요약 내용
            if 'summary' in result.get('results', {}):
                f.write(f"?? 요약 ({len(result['results']['summary'])}자):\n")
                f.write("-" * 40 + "\n")
                f.write(f"{result['results']['summary']}\n\n")

            # 키워드
            if 'keywords' in result.get('results', {}):
                f.write(f"???  키워드 추천 ({len(result['results']['keywords'])}개):\n")
                f.write("-" * 40 + "\n")
                for j, kw in enumerate(result['results']['keywords'], 1):
                    if isinstance(kw, dict):
                        f.write(f"{j}순위: {kw.get('name', 'N/A')}\n")
                        if 'reason' in kw:
                            f.write(f"   이유: {kw['reason']}\n")
                    else:
                        f.write(f"{j}순위: {kw}\n")
                f.write("\n")

            # 제목
            if 'titles' in result.get('results', {}):
                f.write(f"?? 제목 생성 ({len(result['results']['titles'])}개):\n")
                f.write("-" * 40 + "\n")
                for title in result['results']['titles']:
                    if isinstance(title, dict):
                        title_text = title.get('title', 'N/A')
                        title_type_raw = title.get('type', '')
                        title_type = str(title_type_raw or '').lower()
                        if title_type == 'keyword':
                            f.write(f"  ? [키워드형] {title_text}\n")
                        elif title_type and title_type != 'descriptive':
                            f.write(f"  ? [{title_type_raw}] {title_text}\n")
                        else:
                            f.write(f"  ? {title_text}\n")
                    else:
                        f.write(f"  ? {title}\n")
                f.write("\n")

            # 품질 점수
            if 'quality_score' in result:
                f.write(f"? 품질 평가: {result['quality_score']:.3f}/1.000\n")
                if 'warnings' in result and result['warnings']:
                    f.write(f"??  경고사항:\n")
                    for warning in result['warnings']:
                        f.write(f"  - {warning}\n")
                f.write("\n")

            f.write("=" * 60 + "\n")

def process_file(file_path: str, summarizer, classifier, generator, selections: Dict[str, bool]) -> Dict[str, Any]:
    """단일 파일 처리"""
    results = {
        'file': os.path.basename(file_path),
        'success': True,
        'processing_time': 0,
        'ai_summary_time': 0,  # AI 요약 시간 분리 측정
        'results': {}
    }
    
    start_time = time.time()
    
    try:
        # 파일에서 대화 내용 로드
        conversation_text = load_conversation_from_json(file_path)
        if not conversation_text:
            results['success'] = False
            results['error'] = "대화 내용 로드 실패"
            return results
        
        results['conversation_length'] = len(conversation_text)
        # Policy: decide route once (no retry) based on length/size
        try:
            file_size_bytes = os.path.getsize(file_path)
        except Exception:
            file_size_bytes = 0
        try:
            policy_force_long_chars = int(os.getenv('POLICY_FORCE_LONG_CHARS', '50000'))
        except Exception:
            policy_force_long_chars = 50000
        try:
            policy_force_long_bytes = int(os.getenv('POLICY_FORCE_LONG_BYTES', '614400'))  # ~600KB
        except Exception:
            policy_force_long_bytes = 614400
        try:
            policy_long_threshold = int(os.getenv('POLICY_LONG_THRESHOLD_CHARS', '30000'))
        except Exception:
            policy_long_threshold = 30000

        is_force_long = (results['conversation_length'] >= policy_force_long_chars) or (file_size_bytes >= policy_force_long_bytes)
        if is_force_long:
            # Force long-call mode and set safe defaults for long calls
            os.environ['LONG_TEXT_THRESHOLD_CHARS'] = '1'
            if not os.getenv('CHUNK_MAX_CHARS'):
                os.environ['CHUNK_MAX_CHARS'] = '2200'
            if not os.getenv('CHUNK_OVERLAP_CHARS'):
                os.environ['CHUNK_OVERLAP_CHARS'] = '100'
            if not os.getenv('SOFT_TIMEOUT_SEC'):
                os.environ['SOFT_TIMEOUT_SEC'] = '50'
            if not os.getenv('HARD_TIMEOUT_SEC'):
                os.environ['HARD_TIMEOUT_SEC'] = '60'
        else:
            # Prefer fast single-pass summary
            os.environ['LONG_TEXT_THRESHOLD_CHARS'] = str(policy_long_threshold)
        results['long_call_mode'] = bool(is_force_long)
        
        # 1. 요약 생성 (AI 처리 시간 분리 측정)
        if selections['summary']:
            print(f"\n  [1/3] 요약 생성 중...")
            ai_start = time.time()  # AI 요약 시작 시간
            summary_result = summarizer.summarize_consultation(conversation_text)
            ai_end = time.time()  # AI 요약 종료 시간
            results['ai_summary_time'] = ai_end - ai_start  # AI 요약 순수 시간
            
            if summary_result['success']:
                results['results']['summary'] = summary_result['summary']
                print(f"     >> 요약 완료 ({len(summary_result['summary'])}자) - AI시간: {results['ai_summary_time']:.2f}초")
            else:
                results['success'] = False
                results['error'] = summary_result.get('error', '요약 생성 실패')
                results['results']['summary'] = "요약 생성 실패"
                print(f"     ? 요약 실패 - AI시간: {results['ai_summary_time']:.2f}초")
        
        # 2. 키워드 추천
        if selections['keywords']:
            print(f"  ???  키워드 추천 중...")
            keywords = classifier.classify(conversation_text)
            
            if keywords:
                results['results']['keywords'] = keywords
                print(f"     >> 키워드 {len(keywords)}개 추출")
                for idx, kw in enumerate(keywords, 1):
                    print(f"        {idx}순위: {kw.get('name', 'N/A')}")
            else:
                results['results']['keywords'] = []
                print(f"     ? 키워드 추출 실패")
        
        # 3. 제목 생성
        if selections['titles']:
            print(f"  ?? 제목 생성 중...")
            titles = generator.generate(conversation_text)
            
            if titles:
                results['results']['titles'] = titles
                print(f"     >> 제목 {len(titles)}개 생성")
                for title in titles:
                    print(f"        - {title.get('title', 'N/A')}")
            else:
                results['results']['titles'] = []
                print(f"     ? 제목 생성 실패")
        
        # 품질 검증 (요약이 있는 경우만)
        if selections['summary'] and 'summary' in results['results']:
            if results['success']:
                quality_result = quality_validator.validate_summary(
                    results['results']['summary'],
                    conversation_text
                )
                results['quality_score'] = quality_result['quality_score']
                results['warnings'] = quality_result.get('warnings', [])
                results['semantic_check'] = quality_result.get('semantic_check', {})

                if not quality_result.get('is_acceptable', True):
                    results['success'] = False
                    reason = "요약 품질 기준 미달"
                    if results['warnings']:
                        reason += f" ({'; '.join(results['warnings'])})"
                    results['error'] = reason
            
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
        print(f"     ? 오류 발생: {e}")
    
    results['processing_time'] = time.time() - start_time
    return results

def save_checkpoint(processed_files: List[str], results: List[Dict], checkpoint_file: Path):
    """체크포인트 저장"""
    checkpoint_data = {
        'processed_files': processed_files,
        'results': results,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'count': len(results)
    }
    
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

def load_checkpoint(checkpoint_file: Path) -> tuple:
    """체크포인트 로드"""
    if not checkpoint_file.exists():
        return [], []
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        processed_files = checkpoint_data.get('processed_files', [])
        results = checkpoint_data.get('results', [])
        
        print(f"?? 체크포인트 발견: {len(results)}개 파일 이미 처리됨")
        print(f"   마지막 저장: {checkpoint_data.get('timestamp', 'N/A')}")
        
        return processed_files, results
        
    except Exception as e:
        print(f"??  체크포인트 로드 실패: {e}")
        return [], []

def save_dual_model_result(qwen3_result: Dict, midm_result: Dict, selections: Dict[str, bool], output_dir: Path, idx: int, total: int):
    """두 SLM 모델 결과를 하나의 파일에 저장"""

    # 선택된 기능 표시
    features = []
    if selections['summary']:
        features.append('S')
    if selections['keywords']:
        features.append('K')
    if selections['titles']:
        features.append('T')
    feature_str = ''.join(features) if features else 'NONE'

    # 파일명 생성 (both 모델 표시)
    base_name = qwen3_result['file'].replace('.json', '') if qwen3_result else midm_result['file'].replace('.json', '')
    file_timestamp = datetime.now().strftime('%H%M%S')
    individual_file = output_dir / f"{base_name}_SLM_both_{feature_str}_{file_timestamp}.txt"

    # 전체 처리 시간 계산
    total_processing_time = 0
    if qwen3_result.get('success', False):
        total_processing_time += qwen3_result.get('processing_time', 0)
    if midm_result.get('success', False):
        total_processing_time += midm_result.get('processing_time', 0)

    # 파일 저장
    with open(individual_file, 'w', encoding='utf-8') as f:
        f.write(f"AI 상담 분석 결과 (듀얼 SLM 모델) - {base_name}.json\n")
        f.write(f"=" * 70 + "\n")
        f.write(f"처리 순서: [{idx}/{total}]\n")
        f.write(f"처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"테스트 기능: {', '.join([k for k, v in selections.items() if v])}\n")
        f.write(f"사용 모델: Qwen3-1.7B + Midm-2.0-Mini\n")
        f.write(f"전체 처리시간: {total_processing_time:.2f}초\n")

        # 대화 길이 (공통)
        conversation_length = qwen3_result.get('conversation_length', 0) or midm_result.get('conversation_length', 0)
        f.write(f"대화 길이: {conversation_length}자\n")
        f.write("\n" + "=" * 70 + "\n\n")

        # Qwen3-1.7B 결과
        f.write(f"[1] Qwen3-1.7B 결과\n")
        f.write("-" * 70 + "\n")

        if qwen3_result.get('success', False):
            f.write(f"처리시간: {qwen3_result.get('processing_time', 0):.2f}초\n")
            f.write(f"AI 요약시간: {qwen3_result.get('ai_summary_time', 0):.2f}초\n")

            # Qwen3 요약
            if selections['summary'] and 'summary' in qwen3_result.get('results', {}):
                summary = qwen3_result['results']['summary']
                f.write(f"\n== 요약 ({len(summary)}자) ==\n")
                f.write(f"{summary}\n")

            # 품질 평가
            if 'quality_score' in qwen3_result:
                f.write(f"\n품질점수: {qwen3_result['quality_score']:.3f}/1.000\n")
                if 'warnings' in qwen3_result and qwen3_result['warnings']:
                    f.write(f"경고사항:\n")
                    for warning in qwen3_result['warnings']:
                        f.write(f"  - {warning}\n")
        else:
            f.write(f"처리 실패: {qwen3_result.get('error', 'Unknown error')}\n")

        f.write("\n\n")

        # Midm-Mini 결과
        f.write(f"[2] Midm-2.0-Mini 결과\n")
        f.write("-" * 70 + "\n")

        if midm_result.get('success', False):
            f.write(f"처리시간: {midm_result.get('processing_time', 0):.2f}초\n")
            f.write(f"AI 요약시간: {midm_result.get('ai_summary_time', 0):.2f}초\n")

            # Midm 요약
            if selections['summary'] and 'summary' in midm_result.get('results', {}):
                summary = midm_result['results']['summary']
                f.write(f"\n== 요약 ({len(summary)}자) ==\n")
                f.write(f"{summary}\n")

            # 품질 평가
            if 'quality_score' in midm_result:
                f.write(f"\n품질점수: {midm_result['quality_score']:.3f}/1.000\n")
                if 'warnings' in midm_result and midm_result['warnings']:
                    f.write(f"경고사항:\n")
                    for warning in midm_result['warnings']:
                        f.write(f"  - {warning}\n")
        else:
            f.write(f"처리 실패: {midm_result.get('error', 'Unknown error')}\n")

        f.write("\n\n")

        # 모델 비교 요약
        f.write(f"[비교] 모델 비교 요약\n")
        f.write("-" * 70 + "\n")

        if qwen3_result.get('success', False) and midm_result.get('success', False):
            qwen3_summary = qwen3_result.get('results', {}).get('summary', '')
            midm_summary = midm_result.get('results', {}).get('summary', '')

            f.write(f"Qwen3-1.7B 요약 길이: {len(qwen3_summary)}자\n")
            f.write(f"Midm-Mini 요약 길이: {len(midm_summary)}자\n")
            f.write(f"Qwen3 처리시간: {qwen3_result.get('processing_time', 0):.2f}초\n")
            f.write(f"Midm 처리시간: {midm_result.get('processing_time', 0):.2f}초\n")
            f.write(f"Qwen3 품질점수: {qwen3_result.get('quality_score', 0):.3f}\n")
            f.write(f"Midm 품질점수: {midm_result.get('quality_score', 0):.3f}\n")
        else:
            f.write(f"Qwen3 성공 여부: {qwen3_result.get('success', False)}\n")
            f.write(f"Midm 성공 여부: {midm_result.get('success', False)}\n")

        f.write("\n" + "=" * 70 + "\n")

def save_individual_result_with_model(result: Dict, selections: Dict[str, bool], output_dir: Path, idx: int, total: int, model_tier: str = 'llm', model_name: str = ''):
    """개별 통화 결과를 모델명과 함께 즉시 저장"""
    if not result['success']:
        return

    # 선택된 기능 표시
    features = []
    if selections['summary']:
        features.append('S')
    if selections['keywords']:
        features.append('K')
    if selections['titles']:
        features.append('T')
    feature_str = ''.join(features) if features else 'NONE'

    # 모델 티어 및 모델명 표시
    tier_str = model_tier.upper()  # SLM 또는 LLM
    model_suffix = f"_{model_name}" if model_name else ""

    # 파일명 생성
    base_name = result['file'].replace('.json', '')
    file_timestamp = datetime.now().strftime('%H%M%S')
    individual_file = output_dir / f"{base_name}_{tier_str}{model_suffix}_{feature_str}_{file_timestamp}.txt"

    # 파일 저장
    with open(individual_file, 'w', encoding='utf-8') as f:
        f.write(f"AI 상담 분석 결과 - {result['file']}\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"처리 순서: [{idx}/{total}]\n")
        f.write(f"처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"테스트 기능: {', '.join([k for k, v in selections.items() if v])}\n")
        f.write(f"전체 처리시간: {result['processing_time']:.2f}초\n")

        # 모델 정보 추가
        if 'model_name' in result:
            f.write(f"사용 모델: {result['model_name']}\n")
        if 'model_type' in result:
            f.write(f"모델 유형: {result['model_type']}\n")

        f.write(f"AI 요약시간: {result.get('ai_summary_time', 0):.2f}초\n")
        f.write(f"대화 길이: {result.get('conversation_length', 0)}자\n")
        f.write("\n" + "=" * 60 + "\n\n")

        # 요약 결과
        if selections['summary'] and 'summary' in result.get('results', {}):
            summary = result['results']['summary']
            f.write(f"== 요약 ({len(summary)}자) ==\n")
            f.write("-" * 60 + "\n")
            f.write(f"{summary}\n\n")

        # 키워드 결과
        if selections['keywords'] and 'keywords' in result.get('results', {}):
            keywords = result['results']['keywords']
            f.write(f"== 키워드 추천 ({len(keywords)}개) ==\n")
            f.write("-" * 60 + "\n")
            for i, keyword in enumerate(keywords, 1):
                confidence = keyword.get('confidence', 0)
                name = keyword.get('name', 'N/A')
                f.write(f"{i}. {name} (신뢰도: {confidence:.3f})\n")
            f.write("\n")

        # 제목 결과
        if selections['titles'] and 'titles' in result.get('results', {}):
            titles = result['results']['titles']
            f.write(f"== 제목 생성 ({len(titles)}개) ==\n")
            f.write("-" * 60 + "\n")
            for i, title in enumerate(titles, 1):
                title_text = title.get('title', 'N/A')
                title_type_raw = title.get('type', '')
                title_type = str(title_type_raw or '').lower()
                confidence = title.get('confidence', 0)
                label = ''
                if title_type == 'keyword':
                    label = '[키워드형] '
                elif title_type and title_type != 'descriptive':
                    label = f'[{title_type_raw}] '
                f.write(f"{i}. {label}{title_text} (신뢰도: {confidence:.3f})\n")
            f.write("\n")

        # 품질 평가
        if 'quality_score' in result:
            f.write(f"? 품질 평가:\n")
            f.write("-" * 60 + "\n")
            f.write(f"품질점수: {result['quality_score']:.3f}/1.000\n")
            if 'warnings' in result and result['warnings']:
                f.write(f"경고사항:\n")
                for warning in result['warnings']:
                    f.write(f"  - {warning}\n")
            f.write("\n")

        f.write("=" * 60 + "\n")

def save_individual_result(result: Dict, selections: Dict[str, bool], output_dir: Path, idx: int, total: int, model_tier: str = 'llm'):
    """개별 통화 결과를 즉시 저장"""
    if not result['success']:
        return

    # 선택된 기능 표시
    features = []
    if selections['summary']:
        features.append('S')
    if selections['keywords']:
        features.append('K')
    if selections['titles']:
        features.append('T')
    feature_str = ''.join(features) if features else 'NONE'

    # 모델 티어 표시
    tier_str = model_tier.upper()  # SLM 또는 LLM

    # 파일명 생성
    base_name = result['file'].replace('.json', '')
    file_timestamp = datetime.now().strftime('%H%M%S')
    individual_file = output_dir / f"{base_name}_{tier_str}_{feature_str}_{file_timestamp}.txt"
    
    # 파일 저장
    with open(individual_file, 'w', encoding='utf-8') as f:
        f.write(f"AI 상담 분석 결과 - {result['file']}\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"처리 순서: [{idx}/{total}]\n")
        f.write(f"처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"테스트 기능: {', '.join([k for k, v in selections.items() if v])}\n")
        f.write(f"전체 처리시간: {result['processing_time']:.2f}초\n")
        
        # AI 요약 시간 정보 (요약 기능이 선택된 경우)
        if 'ai_summary_time' in result and result['ai_summary_time'] > 0:
            f.write(f"AI 요약시간: {result['ai_summary_time']:.2f}초\n")
        
        # 대화 길이 정보
        if 'conversation_length' in result:
            f.write(f"대화 길이: {result['conversation_length']}자\n")
        
        f.write(f"\n" + "=" * 60 + "\n")
        
        # 요약 내용
        if 'summary' in result.get('results', {}):
            f.write(f"\n== 요약 ({len(result['results']['summary'])}자) ==\n")
            f.write("-" * 60 + "\n")
            f.write(f"{result['results']['summary']}\n")
        
        # 키워드
        if 'keywords' in result.get('results', {}):
            f.write(f"\n??? 키워드 추천 ({len(result['results']['keywords'])}개):\n")
            f.write("-" * 60 + "\n")
            for i, kw in enumerate(result['results']['keywords'], 1):
                f.write(f"{i}순위: {kw.get('name', 'N/A')}")
                if 'reason' in kw:
                    f.write(f"\n   이유: {kw['reason']}")
                f.write("\n")
        
        # 제목
        if 'titles' in result.get('results', {}):
            f.write(f"\n?? 제목 생성 ({len(result['results']['titles'])}개):\n")
            f.write("-" * 60 + "\n")
            for title in result['results']['titles']:
                if isinstance(title, dict):
                    title_text = title.get('title', 'N/A')
                    title_type_raw = title.get('type', '')
                    title_type = str(title_type_raw or '').lower()
                    if title_type == 'keyword':
                        f.write(f"  ? [키워드형] {title_text}\n")
                    elif title_type and title_type != 'descriptive':
                        f.write(f"  ? [{title_type_raw}] {title_text}\n")
                    else:
                        f.write(f"  ? {title_text}\n")
                else:
                    f.write(f"  ? {title}\n")
        
        # 품질 점수
        if 'quality_score' in result:
            f.write(f"\n? 품질 평가:\n")
            f.write("-" * 60 + "\n")
            f.write(f"품질점수: {result['quality_score']:.3f}/1.000\n")
            if 'warnings' in result and result['warnings']:
                f.write(f"\n?? 경고사항:\n")
                for warning in result['warnings']:
                    f.write(f"  - {warning}\n")
        
        f.write(f"\n" + "=" * 60 + "\n")
        f.write(f"저장 완료: {datetime.now().strftime('%H:%M:%S')}\n")

def save_results(results: List[Dict], selections: Dict[str, bool], output_dir: Path):
    """결과를 파일로 저장 (통합 파일 + 개별 파일)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 선택된 기능 표시
    features = []
    if selections['summary']:
        features.append('S')
    if selections['keywords']:
        features.append('K')
    if selections['titles']:
        features.append('T')
    feature_str = ''.join(features) if features else 'NONE'
    
    # JSON 결과 저장
    json_file = output_dir / f"selective_test_{feature_str}_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_info': {
                'timestamp': timestamp,
                'features_tested': selections,
                'total_files': len(results),
                'successful_files': sum(1 for r in results if r['success'])
            },
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    # 요약 텍스트 저장
    txt_file = output_dir / f"selective_test_{feature_str}_{timestamp}.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"선택적 AI 분석 테스트 결과\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"테스트 시간: {timestamp}\n")
        f.write(f"테스트 기능: {', '.join([k for k, v in selections.items() if v])}\n")
        f.write(f"전체 파일: {len(results)}개\n")
        f.write(f"성공: {sum(1 for r in results if r['success'])}개\n")
        f.write(f"실패: {sum(1 for r in results if not r['success'])}개\n")
        
        # 평균 처리시간 계산
        processing_times = [r['processing_time'] for r in results if 'processing_time' in r]
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        f.write(f"평균 처리시간: {avg_time:.2f}초/파일\n")
        
        # 품질 점수 통계
        quality_scores = [r.get('quality_score', 0) for r in results if 'quality_score' in r]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            f.write(f"평균 품질점수: {avg_quality:.3f}/1.000\n")
        
        f.write(f"\n" + "=" * 60 + "\n")
        f.write(f"개별 통화 분석 결과\n")
        f.write("=" * 60 + "\n")
        
        for idx, r in enumerate(results, 1):
            f.write(f"\n{'='*60}\n")
            f.write(f"[{idx}/{len(results)}] {r['file']}\n")
            f.write(f"{'='*60}\n")
            f.write(f"처리시간: {r.get('processing_time', 0):.2f}초\n")
            
            if r['success']:
                # 요약 내용 출력
                if 'summary' in r.get('results', {}):
                    f.write(f"\n== 요약 ({len(r['results']['summary'])}자) ==\n")
                    f.write(f"{r['results']['summary']}\n")
                
                # 키워드 출력
                if 'keywords' in r.get('results', {}):
                    f.write(f"\n??? 키워드 ({len(r['results']['keywords'])}개):\n")
                    for i, kw in enumerate(r['results']['keywords'], 1):
                        f.write(f"  {i}순위: {kw.get('name', 'N/A')}")
                        if 'reason' in kw:
                            f.write(f" - {kw['reason']}")
                        f.write("\n")
                
                # 제목 출력
                if 'titles' in r.get('results', {}):
                    f.write(f"\n?? 제목 ({len(r['results']['titles'])}개):\n")
                    for title in r['results']['titles']:
                        if isinstance(title, dict):
                            title_text = title.get('title', 'N/A')
                            title_type_raw = title.get('type', '')
                            title_type = str(title_type_raw or '').lower()
                            if title_type == 'keyword':
                                f.write(f"  ? [키워드형] {title_text}\n")
                            elif title_type and title_type != 'descriptive':
                                f.write(f"  ? [{title_type_raw}] {title_text}\n")
                            else:
                                f.write(f"  ? {title_text}\n")
                        else:
                            f.write(f"  ? {title}\n")
                
                # 품질 점수
                if 'quality_score' in r:
                    f.write(f"\n? 품질점수: {r['quality_score']:.2f}/1.00\n")
                    if 'warnings' in r and r['warnings']:
                        f.write(f"?? 경고:\n")
                        for warning in r['warnings']:
                            f.write(f"  - {warning}\n")
            else:
                f.write(f"? 오류: {r.get('error', 'Unknown')}\n")
    
    print(f"\n결과 저장됨:")
    print(f"  - 통합 JSON: {json_file}")
    print(f"  - 통합 TXT: {txt_file}")
    print(f"  - 개별 파일: {output_dir}/ ({sum(1 for r in results if r['success'])}개 파일)")

def main():
    parser = argparse.ArgumentParser(
        description='AI 상담 분석 선택적 테스트 - 웹 대시보드처럼 기능을 선택해서 테스트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 인터랙티브 모드 (선택 UI 표시)
  python local_test_selective_ai.py -i
  
  # 모든 기능 테스트
  python local_test_selective_ai.py -f call_data/call_00001.json
  
  # 요약만 테스트
  python local_test_selective_ai.py -f call_data/call_00001.json --only-summary
  
  # 키워드와 제목만 테스트
  python local_test_selective_ai.py -f call_data/call_00001.json --features keywords,titles
  
  # 요약 제외하고 테스트
  python local_test_selective_ai.py -f call_data/call_00001.json --no-summary
  
  # 여러 파일 테스트 (100개)
  python local_test_selective_ai.py -c 100 --features summary,keywords
  
  # 체크포인트 기능 사용 (중단된 테스트 재개)
  python local_test_selective_ai.py -c 1000 --use-checkpoint
  
  # 체크포인트 저장 간격 조정 (5개 파일마다 저장)
  python local_test_selective_ai.py -c 100 --use-checkpoint --checkpoint-interval 5
  
  # 날짜별 연속 처리 (자동으로 모든 날짜 순서대로)
  python local_test_selective_ai.py --auto-continue -c 1000
  
  # 특정 날짜부터 시작하여 자동 연속 처리
  python local_test_selective_ai.py --auto-continue --start-date 2025-08-01
  
  # 사용자 지정 날짜 순서로 처리
  python local_test_selective_ai.py --date-sequence 2025-07-15,2025-08-11,2025-08-25
  
  # 자동 날짜 순서로 처리 (auto 키워드 사용)
  python local_test_selective_ai.py --date-sequence auto --start-date 2025-08-01
  
  # 출력 폴더명에 접미사 추가 (2025-09-10_2)
  python local_test_selective_ai.py --date-sequence 2025-08-20,2025-08-25 --features summary,titles --output-suffix _2
  
  # 출력 폴더명 직접 지정
  python local_test_selective_ai.py --date-sequence 2025-08-20,2025-08-25 --features summary,titles --output-folder 2025-09-10_test
        """
    )
    
    # 파일/디렉토리 선택
    parser.add_argument('--file', '-f', help='처리할 JSON 파일 경로')
    parser.add_argument('--dir', '-d', default=str(project_root / 'call_data'), 
                       help='처리할 디렉토리 (기본: call_data)')
    parser.add_argument('--count', '-c', type=int, 
                       help='처리할 통화 개수')
    
    # 기능 선택 옵션
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='인터랙티브 모드로 기능 선택')
    parser.add_argument('--features', type=str,
                       help='테스트할 기능 (쉼표로 구분: summary,keywords,titles 또는 s,k,t)')
    
    # 체크포인트 옵션
    parser.add_argument('--use-checkpoint', action='store_true',
                       help='중단된 테스트를 체크포인트에서 이어서 진행')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='체크포인트 저장 간격 (파일 개수, 기본: 10)')
    
    # 날짜별 연속 처리 옵션
    parser.add_argument('--date-sequence', type=str,
                       help='처리할 날짜 순서 (예: 2025-07-15,2025-08-01 또는 auto)')
    parser.add_argument('--auto-continue', action='store_true',
                       help='한 날짜 폴더 완료 후 자동으로 다음 날짜로 진행')
    parser.add_argument('--start-date', type=str,
                       help='시작 날짜 (YYYY-MM-DD 형식)')
    parser.add_argument('--output-suffix', type=str,
                       help='출력 폴더명에 추가할 접미사 (예: _2, _test)')
    parser.add_argument('--output-folder', type=str,
                       help='출력 폴더명 직접 지정 (예: 2025-09-10_2)')
    
    # 개별 기능 플래그
    parser.add_argument('--no-summary', action='store_true',
                       help='요약 생성 제외')
    parser.add_argument('--no-keywords', action='store_true',
                       help='키워드 추천 제외')
    parser.add_argument('--no-titles', action='store_true',
                       help='제목 생성 제외')
    
    # 단일 기능만 선택
    parser.add_argument('--only-summary', action='store_true',
                       help='요약만 테스트')
    parser.add_argument('--only-keywords', action='store_true',
                       help='키워드만 테스트')
    parser.add_argument('--only-titles', action='store_true',
                       help='제목만 테스트')
    
    # Long-Call mode tuning options
    parser.add_argument('--long-call-threshold-chars', type=int, default=8000,
                       help='Long-Call Mode trigger threshold by characters (default: 8000)')
    parser.add_argument('--long-soft-timeout-sec', type=float, default=800.0,
                       help='Long-Call soft timeout seconds (default: 800 - 13.3 minutes)')
    parser.add_argument('--long-hard-timeout-sec', type=float, default=900.0,
                       help='Long-Call hard timeout seconds (default: 900 - 15 minutes)')
    parser.add_argument('--long-chunk-chars', type=int, default=1200,
                       help='Long-Call chunk max characters (default: 1200)')
    parser.add_argument('--long-overlap-chars', type=int, default=120,
                       help='Long-Call chunk overlap characters (default: 120)')

    # Model tier selection
    parser.add_argument('--model-tier', choices=['slm', 'llm', 'all-llm'], default='llm',
                       help='모델 티어 선택: slm (Small Language Model, 실시간용), llm (Large Language Model, 배치용, 기본값), all-llm (모든 LLM 모델 비교 테스트)')

    # 간단한 날짜 폴더 선택 옵션
    parser.add_argument('--test-date', type=str,
                       help='테스트할 날짜 폴더 (YYYY-MM-DD 형식). 해당 날짜 폴더의 모든 파일을 테스트합니다.')

    # SLM 모델 선택 옵션
    parser.add_argument('--slm-model', choices=['qwen3', 'midm', 'both'], default='qwen3',
                       help='SLM 모델 선택: qwen3 (Qwen3-1.7B), midm (Midm-2.0-Mini), both (두 모델 모두 테스트)')

    args = parser.parse_args()

    print_banner(args.model_tier if args.model_tier != 'all-llm' else 'all_llm')

    # Apply Long-Call Mode environment settings so summarizer can pick them up
    try:
        os.environ['LONG_TEXT_THRESHOLD_CHARS'] = str(args.long_call_threshold_chars)
        os.environ['SOFT_TIMEOUT_SEC'] = str(args.long_soft_timeout_sec)
        os.environ['HARD_TIMEOUT_SEC'] = str(args.long_hard_timeout_sec)
        os.environ['CHUNK_MAX_CHARS'] = str(args.long_chunk_chars)
        os.environ['CHUNK_OVERLAP_CHARS'] = str(args.long_overlap_chars)
    except Exception:
        pass
    
    # 기능 선택
    if args.interactive:
        selections = get_user_selection()
    else:
        selections = parse_feature_args(args)
        print("\n선택된 기능:")
        if selections['summary']:
            print("  + 요약 생성")
        if selections['keywords']:
            print("  + 키워드 추천")
        if selections['titles']:
            print("  + 제목 생성")
    
    if not any(selections.values()):
        print("\n??  테스트할 기능이 선택되지 않았습니다.")
        print("최소 하나 이상의 기능을 선택해주세요.")
        return
    
    # 모델 로드
    print("\n" + "=" * 60)
    print("모델 로드 중...")
    print("-" * 60)
    
    try:
        model_path = str(project_root / "models" / "Qwen3-4B")
        
        # 필요한 모델만 로드
        summarizer = None
        classifier = None
        generator = None
        
        # 요약이나 다른 기능이 필요한 경우 summarizer 로드
        if any(selections.values()):
            if args.model_tier == 'slm':
                print("  ?? SLM 모델 로드 중 (실시간 처리용)...")

                # SLM 모델 선택에 따라 로드
                if args.slm_model == 'qwen3':
                    slm_model_path = str(project_root / "models" / "Qwen3-1.7B")
                    summarizer = Qwen3Summarizer(slm_model_path)
                    model_name = "Qwen3-1.7B"
                elif args.slm_model == 'midm':
                    slm_model_path = str(project_root / "models" / "Midm-2.0-Mini")
                    summarizer = MidmSummarizer(slm_model_path)
                    model_name = "Midm-2.0-Mini"
                else:  # both - 첫 번째 테스트는 Qwen3로 시작
                    slm_model_path = str(project_root / "models" / "Qwen3-1.7B")
                    summarizer = Qwen3Summarizer(slm_model_path)
                    model_name = "Qwen3-1.7B (첫 번째)"

                if not summarizer.load_model():
                    print("  ? SLM 모델 로드 실패")
                    return
                print(f"  + SLM 모델 로드 완료 ({model_name})")

                # SLM은 요약만 지원
                if selections['keywords'] or selections['titles']:
                    print("  ??  SLM은 요약만 지원합니다. 키워드/제목은 건너뜁니다.")
                    selections['keywords'] = False
                    selections['titles'] = False

            elif args.model_tier == 'all-llm':
                print("  ?? ALL-LLM 모드: 모든 LLM 모델 순차 테스트...")
                print(f"  대상 모델: {len(LLM_MODELS)}개")
                for model_id, model_config in LLM_MODELS.items():
                    print(f"    - {model_config['name']}")
                print("  + ALL-LLM 모드 준비 완료")

                # ALL-LLM 모드에서는 개별 모델 로드하지 않음
                summarizer = None
                classifier = None
                generator = None

            else:  # LLM
                print("  ?? LLM 모델 로드 중 (배치 처리용)...")
                summarizer = Qwen2507Summarizer(model_path)
                if not summarizer.load_model():
                    print("  ? LLM 모델 로드 실패")
                    return
                print("  + LLM 모델 로드 완료 (Qwen3-4B)")

        # ALL-LLM 모드가 아닌 경우에만 개별 컴포넌트 초기화
        if args.model_tier != 'all-llm':
            # 키워드 추천기
            if selections['keywords']:
                print("  ???  키워드 추천기 초기화...")
                classifier = CategoryClassifier(shared_summarizer=summarizer)
                print("  + 키워드 추천기 준비 완료")

            # 제목 생성기
            if selections['titles']:
                print("  ?? 제목 생성기 초기화...")
                generator = TitleGenerator(shared_summarizer=summarizer)
                print("  + 제목 생성기 준비 완료")
        
    except Exception as e:
        print(f"\n? 모델 로드 오류: {e}")
        return
    
    # 처리할 파일 목록 생성
    files_to_process = []
    
    if args.file:
        if os.path.exists(args.file):
            files_to_process = [args.file]
        else:
            print(f"\n? 파일을 찾을 수 없습니다: {args.file}")
            return
    else:
        data_dir = Path(args.dir)
        if not data_dir.exists():
            print(f"\n? 디렉토리를 찾을 수 없습니다: {args.dir}")
            return
        
        # 특정 날짜 폴더 테스트 모드
        if args.test_date:
            test_date_dir = data_dir / args.test_date
            if not test_date_dir.exists():
                print(f"\n? 날짜 폴더를 찾을 수 없습니다: {args.test_date}")
                print(f"?? 사용 가능한 날짜 폴더들:")
                available_dates = get_date_folders(data_dir)
                for date in available_dates:
                    file_count = len(list((data_dir / date).glob("*.json")))
                    print(f"   - {date}: {file_count}개 파일")
                return

            print(f"\n[모드] 특정 날짜 폴더 테스트 모드")
            print(f"   테스트 날짜: {args.test_date}")

            # 해당 날짜 폴더의 모든 JSON 파일 수집
            date_files = []
            for json_file in sorted(test_date_dir.glob('*.json')):
                if 'rename_map' not in json_file.name:
                    date_files.append(str(json_file))

            print(f"   발견된 파일: {len(date_files)}개")

            # count 제한 적용 (선택사항)
            if args.count:
                date_files = date_files[:args.count]
                print(f"   처리할 파일: {len(date_files)}개 (제한 적용)")

            files_to_process = date_files

        # 날짜별 연속 처리 모드
        elif args.date_sequence or args.auto_continue:
            if args.date_sequence:
                date_sequence = parse_date_sequence(args.date_sequence, data_dir, args.start_date)
            else:  # args.auto_continue
                date_sequence = get_date_folders(data_dir)
                if args.start_date:
                    try:
                        start_idx = date_sequence.index(args.start_date)
                        date_sequence = date_sequence[start_idx:]
                    except ValueError:
                        print(f"??  시작 날짜를 찾을 수 없습니다: {args.start_date}")

            print(f"\n[모드] 날짜별 연속 처리 모드")
            print(f"   처리 순서: {' → '.join(date_sequence)}")

            files_to_process = get_files_by_date_sequence(data_dir, date_sequence, args.count)
        else:
            # 기존 방식: 모든 파일을 한 번에 수집
            for json_file in data_dir.rglob('*.json'):
                if 'rename_map' not in json_file.name:
                    files_to_process.append(str(json_file))
                    if args.count and len(files_to_process) >= args.count:
                        break
    
    if not files_to_process:
        print("\n??  처리할 파일이 없습니다.")
        return
    
    # 출력 디렉토리 구조: SLM_test/LLM_test > 오늘날짜 폴더
    model_folder = "SLM_test" if args.model_tier == 'slm' else "LLM_test"

    # 출력 폴더명은 항상 테스트 실행 날짜(오늘)로 설정
    if args.output_folder:
        # 사용자 지정 폴더명
        output_date = args.output_folder
    else:
        # 기본: 오늘 날짜
        output_date = datetime.now().strftime("%Y-%m-%d")
        if args.output_suffix:
            output_date = f"{output_date}{args.output_suffix}"

    # 출력 디렉토리: scripts/outputs/SLM_test|LLM_test/오늘날짜/
    output_dir = project_root / "scripts" / "outputs" / model_folder / output_date
    output_dir.mkdir(parents=True, exist_ok=True)

    # 테스트 대상 데이터 정보 표시
    if args.test_date:
        print(f"\n[테스트] 데이터 날짜: {args.test_date}")
    print(f"[폴더] 결과 저장: {model_folder}/{output_date}")
    print(f"       전체 경로: {output_dir}")
    
    # 체크포인트 파일 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    features = []
    if selections['summary']: features.append('S')
    if selections['keywords']: features.append('K')  
    if selections['titles']: features.append('T')
    feature_str = ''.join(features) if features else 'NONE'
    
    checkpoint_file = output_dir / f"checkpoint_{feature_str}_{timestamp}.json"
    processed_files = []
    results = []
    
    # 체크포인트 로드 (요청된 경우)
    if args.use_checkpoint:
        # 기존 체크포인트 파일 찾기 (가장 최근)
        checkpoint_pattern = f"checkpoint_{feature_str}_*.json"
        existing_checkpoints = list(output_dir.glob(checkpoint_pattern))
        if existing_checkpoints:
            # 가장 최근 체크포인트 사용
            latest_checkpoint = max(existing_checkpoints, key=lambda p: p.stat().st_mtime)
            processed_files, results = load_checkpoint(latest_checkpoint)
            checkpoint_file = latest_checkpoint  # 기존 체크포인트 파일 재사용
    
    # 이미 처리된 파일 제외
    files_to_process = [f for f in files_to_process if os.path.basename(f) not in processed_files]
    
    # 파일 처리
    print("\n" + "=" * 60)
    print(f"파일 처리 시작")
    if args.use_checkpoint and processed_files:
        print(f"  체크포인트에서 재개: {len(processed_files)}개 이미 완료")
        print(f"  남은 파일: {len(files_to_process)}개")
    else:
        print(f"  전체 파일: {len(files_to_process)}개")
    print("=" * 60)
    
    total_start = time.time()
    start_idx = len(results) + 1  # 체크포인트에서 재개할 때 인덱스 조정
    
    # 날짜별 진행 상황 추적을 위한 변수
    current_date = None
    date_file_count = 0
    
    for idx, file_path in enumerate(files_to_process, start_idx):
        # 날짜 폴더 변경 감지 및 표시
        if args.date_sequence or args.auto_continue:
            file_date = Path(file_path).parent.name
            if current_date != file_date:
                if current_date is not None:
                    print(f"\n[완료] {current_date} 폴더 완료! ({date_file_count}개 파일 처리됨)")
                current_date = file_date
                date_file_count = 0
                print(f"\n[시작] === {current_date} 폴더 시작 ===")
            date_file_count += 1
        
        print(f"\n[{idx}/{len(files_to_process) + len(processed_files)}] 처리 중: {os.path.basename(file_path)}")

        # ALL-LLM 모드 처리
        if args.model_tier == 'all-llm':
            # 모든 LLM 모델로 처리하고 하나의 파일에 결과 저장
            all_llm_results = process_file_with_all_llm_models(file_path, selections, output_dir, idx, len(files_to_process) + len(processed_files))

            # 통계를 위해 첫 번째 성공한 결과를 메인 결과로 사용
            successful_results = [r for r in all_llm_results if r['success']]
            if successful_results:
                result = successful_results[0]
                results.extend(all_llm_results)  # 모든 결과를 결과 리스트에 추가
                print(f"  >> ALL-LLM 모델 결과 저장 완료 ({len(successful_results)}/{len(LLM_MODELS)}개 성공)")
            else:
                result = {'success': False, 'error': '모든 LLM 모델 실패'}
                results.append(result)
                print(f"  ? 모든 LLM 모델 처리 실패")

            processed_files.append(os.path.basename(file_path))

        # SLM 'both' 모드 처리
        elif args.model_tier == 'slm' and args.slm_model == 'both':
            # 양쪽 모델로 처리하고 각각 결과 저장
            both_results = process_file_with_both_slm_models(file_path, selections, output_dir, idx, len(files_to_process) + len(processed_files))

            # 통계를 위해 첫 번째 결과를 메인 결과로 사용
            if both_results:
                result = both_results[0]
                results.extend(both_results)  # 모든 결과를 결과 리스트에 추가
            else:
                result = {'success': False, 'error': '양쪽 모델 모두 실패'}
                results.append(result)

            processed_files.append(os.path.basename(file_path))

            if result['success']:
                print(f"  >> 듀얼 모델 결과 저장 완료 (1개 통합 파일)")
            else:
                print(f"  ERROR: 처리 실패: {result.get('error', 'Unknown error')}")

        else:
            # 기존 단일 모델 처리
            result = process_file(file_path, summarizer, classifier, generator, selections)
            results.append(result)
            processed_files.append(os.path.basename(file_path))

            if result['success']:
                total_time = result['processing_time']
                ai_time = result.get('ai_summary_time', 0)
                print(f"  ??  전체: {total_time:.2f}초 | AI요약: {ai_time:.2f}초")
                # 개별 파일 즉시 저장
                model_name = ""
                if args.model_tier == 'slm':
                    model_name = args.slm_model
                save_individual_result_with_model(result, selections, output_dir, idx, len(files_to_process) + len(processed_files), args.model_tier, model_name)
                print(f"  >> 결과 저장 완료")
            else:
                print(f"  ? 처리 실패: {result.get('error', 'Unknown error')}")
        
        # 주기적 체크포인트 저장
        if idx % args.checkpoint_interval == 0:
            save_checkpoint(processed_files, results, checkpoint_file)
            print(f"  [저장] 체크포인트 저장됨 ({len(results)}개 파일)")
    
    # 마지막 날짜 폴더 완료 메시지
    if current_date and (args.date_sequence or args.auto_continue):
        print(f"\n? {current_date} 폴더 완료! ({date_file_count}개 파일 처리됨)")
    
    # 최종 체크포인트 저장
    if results:
        save_checkpoint(processed_files, results, checkpoint_file)
        print(f"\n[저장] 최종 체크포인트 저장됨")
    
    total_time = time.time() - total_start
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
    print(f"전체 파일: {len(results)}개")
    print(f"성공: {sum(1 for r in results if r['success'])}개")
    print(f"실패: {sum(1 for r in results if not r['success'])}개")
    print(f"총 처리시간: {total_time:.2f}초")
    print(f"평균 처리시간: {total_time/len(results):.2f}초/파일")
    
    # 통합 결과 저장 (개별 파일은 이미 저장됨)
    save_results(results, selections, output_dir)
    
    # 메모리 정리
    if summarizer:
        summarizer.cleanup()
    
    print("\n>> 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    main()



