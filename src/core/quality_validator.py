"""
Batch 요약 품질 검증 유틸리티
요약 포맷, 길이, 의미 밀도, 허위 정보 등을 검사해 점수화합니다.
"""

import re
import logging
from typing import Dict, List, Any, Optional
from functools import lru_cache
from collections import Counter

logger = logging.getLogger(__name__)

SEMANTIC_TOKEN_PATTERN = re.compile(r'[가-힣A-Za-z0-9]+')
RAW_SPEAKER_PATTERN = re.compile(r'(^|\n)\s*(고객|상담사)\s*:', re.MULTILINE)
RAW_SPEAKER_INLINE_PATTERN = re.compile(r'(?<!\*)\b(고객|상담사)\s*:', re.MULTILINE)
DEFAULT_STOPWORDS = {'고객', '상담사', '상담결과', '상담', '결과'}

PHONE_PATTERN = re.compile(r'\d{2,3}-\d{3,4}-\d{4}')
NUMBER_PATTERN = re.compile(r'\d{4,}')
YEAR_PATTERN = re.compile(r'20\d{2}')

class QualityValidator:
    """요약 결과에 대한 정량·정성 검증을 수행합니다."""

    def __init__(self) -> None:
        # Batch 기본 기준 (엄격)
        self.min_length = 30
        self.max_length = 1000
        self.accept_threshold = 0.70
        self.semantic_min_tokens = 25
        self.semantic_unique_ratio = 0.40
        self.semantic_repeat_threshold = 0.50

        # 실시간 Batch 완화 기준 (경량 모델용)
        self.realtime_accept_threshold = 0.40
        self.realtime_semantic_min_tokens = 10
        self.realtime_semantic_unique_ratio = 0.25
        self.realtime_semantic_repeat_threshold = 0.70

        self.stopwords = set(DEFAULT_STOPWORDS)

    @lru_cache(maxsize=100)
    def check_format(self, summary: str) -> Dict[str, bool]:
        if not summary:
            return {
                'has_format': False,
                'has_customer': False,
                'has_agent': False,
                'has_result': False
            }

        lines = [line.strip() for line in summary.splitlines() if line.strip()]
        has_customer = any(line.startswith('**고객**') or '고객' in line for line in lines[:3])
        has_agent = any(line.startswith('**상담사**') or '상담사' in line for line in lines[:3])
        has_result = len(lines) >= 3 or any('결과' in line for line in lines[:4])

        return {
            'has_format': has_customer and has_agent and has_result,
            'has_customer': has_customer,
            'has_agent': has_agent,
            'has_result': has_result
        }

    def check_length(self, summary: str) -> Dict[str, Any]:
        if not summary:
            return {'is_valid': False, 'length': 0, 'warning': '요약 없음'}

        length = len(summary)
        if length < self.min_length:
            return {'is_valid': False, 'length': length, 'warning': '길이가 너무 짧습니다'}
        if length > self.max_length:
            return {'is_valid': False, 'length': length, 'warning': '길이가 너무 깁니다'}
        return {'is_valid': True, 'length': length, 'warning': None}

    def _check_semantic_density_internal(
        self,
        summary: str,
        min_tokens: int,
        unique_ratio_threshold: float,
        repeat_threshold: float
    ) -> Dict[str, Any]:
        """
        의미 밀도 검사 공통 로직

        Args:
            summary: 검사할 요약 텍스트
            min_tokens: 최소 토큰 수 기준
            unique_ratio_threshold: 최소 고유 비율 기준
            repeat_threshold: 최대 반복 비율 기준

        Returns:
            검사 결과 딕셔너리
        """
        result: Dict[str, Any] = {
            'is_valid': False,
            'token_count': 0,
            'unique_ratio': 0.0,
            'repetition_ratio': 0.0,
            'raw_speaker': False,
            'warnings': []
        }

        if not summary:
            result['warnings'].append('요약 본문이 비어 있습니다')
            return result

        normalized = re.sub(r'\*\*(고객|상담사|상담결과)\*\*\s*:', ' ', summary)
        tokens = SEMANTIC_TOKEN_PATTERN.findall(normalized)
        tokens = [tok for tok in tokens if tok not in self.stopwords]

        token_count = len(tokens)
        result['token_count'] = token_count

        if token_count:
            counter = Counter(tokens)
            most_common = counter.most_common(1)[0][1]
            unique_ratio = len(counter) / float(token_count)
            repetition_ratio = most_common / float(token_count)
        else:
            unique_ratio = 0.0
            repetition_ratio = 0.0

        result['unique_ratio'] = unique_ratio
        result['repetition_ratio'] = repetition_ratio

        raw_speaker = bool(
            RAW_SPEAKER_PATTERN.search(summary) or
            RAW_SPEAKER_INLINE_PATTERN.search(summary)
        )
        result['raw_speaker'] = raw_speaker

        if token_count < min_tokens:
            result['warnings'].append('요약 본문 분량이 부족합니다')
        if unique_ratio < unique_ratio_threshold:
            result['warnings'].append('어휘 다양성이 낮습니다')
        if repetition_ratio > repeat_threshold:
            result['warnings'].append('특정 단어가 과도하게 반복됩니다')
        if raw_speaker:
            result['warnings'].append('요약에 원문 대화 표현(고객:/상담사:)이 포함되어 있습니다')

        result['is_valid'] = (
            token_count >= min_tokens and
            unique_ratio >= unique_ratio_threshold and
            repetition_ratio <= repeat_threshold and
            not raw_speaker
        )
        return result

    def check_semantic_density(self, summary: str) -> Dict[str, Any]:
        """배치 모드용 의미 밀도 검사 (엄격한 기준)"""
        return self._check_semantic_density_internal(
            summary,
            self.semantic_min_tokens,
            self.semantic_unique_ratio,
            self.semantic_repeat_threshold
        )

    @lru_cache(maxsize=50)
    def check_warnings(self, original_text: Optional[str], summary: str) -> List[str]:
        warnings: List[str] = []
        if not original_text or not summary:
            return warnings

        original_numbers = set(NUMBER_PATTERN.findall(original_text))
        summary_numbers = set(NUMBER_PATTERN.findall(summary))
        new_numbers = summary_numbers - original_numbers
        if new_numbers:
            warnings.append(f'원문에 없는 숫자 등장: {", ".join(sorted(new_numbers))}')

        original_years = set(YEAR_PATTERN.findall(original_text))
        summary_years = set(YEAR_PATTERN.findall(summary))
        new_years = summary_years - original_years
        if new_years:
            warnings.append(f'원문에 없는 연도 등장: {", ".join(sorted(new_years))}')

        original_phones = set(PHONE_PATTERN.findall(original_text))
        summary_phones = set(PHONE_PATTERN.findall(summary))
        new_phones = summary_phones - original_phones
        if new_phones:
            warnings.append(f'원문에 없는 전화번호 등장: {", ".join(sorted(new_phones))}')

        return warnings

    def validate_analysis_quality(self, ai_results: Dict[str, Any], conversation_text: str) -> float:
        """전체 AI 분석 품질을 평가합니다."""
        try:
            summary = ai_results.get('summary', '')
            if not summary:
                return 0.0

            # 기존 validate_summary 메서드 사용
            validation_result = self.validate_summary(summary, conversation_text)
            return validation_result.get('quality_score', 0.0)

        except Exception as e:
            logger.error(f"[QualityValidator] Analysis quality validation error: {e}")
            return 0.0

    def _validate_summary_internal(
        self,
        summary: str,
        original_text: Optional[str],
        semantic_checker: callable,
        accept_threshold: float,
        semantic_penalty: float
    ) -> Dict[str, Any]:
        """
        요약 검증 공통 로직

        Args:
            summary: 검증할 요약 텍스트
            original_text: 원본 텍스트 (환각 검사용)
            semantic_checker: 의미 밀도 검사 메서드
            accept_threshold: 품질 통과 기준 점수
            semantic_penalty: 의미 밀도 실패 시 페널티 점수

        Returns:
            검증 결과 딕셔너리
        """
        format_check = self.check_format(summary)
        length_check = self.check_length(summary)

        warnings = list(self.check_warnings(original_text, summary) if original_text else [])
        semantic_check = semantic_checker(summary)
        for warning in semantic_check['warnings']:
            if warning and warning not in warnings:
                warnings.append(warning)

        quality_score = 0.0
        if format_check['has_format']:
            quality_score += 0.30
        if length_check['is_valid']:
            quality_score += 0.20
        if format_check['has_result']:
            quality_score += 0.15
        if semantic_check['is_valid']:
            quality_score += 0.25
        else:
            quality_score -= semantic_penalty
        if not warnings:
            quality_score += 0.10
        if semantic_check['raw_speaker']:
            quality_score -= 0.05

        quality_score = max(0.0, min(1.0, quality_score))
        warnings = list(dict.fromkeys(warnings))
        is_acceptable = quality_score >= accept_threshold

        return {
            'format_check': format_check,
            'length_check': length_check,
            'semantic_check': semantic_check,
            'warnings': warnings,
            'quality_score': quality_score,
            'is_acceptable': is_acceptable
        }

    def validate_summary_realtime(
        self,
        summary: str,
        original_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """실시간 모드 전용 요약 검증 (완화된 기준 적용)"""
        return self._validate_summary_internal(
            summary,
            original_text,
            self.check_semantic_density_realtime,
            self.realtime_accept_threshold,
            semantic_penalty=0.10  # 실시간은 페널티 완화
        )

    def check_semantic_density_realtime(self, summary: str) -> Dict[str, Any]:
        """실시간 모드용 의미 밀도 검사 (완화된 기준)"""
        return self._check_semantic_density_internal(
            summary,
            self.realtime_semantic_min_tokens,
            self.realtime_semantic_unique_ratio,
            self.realtime_semantic_repeat_threshold
        )

    def validate_summary(
        self,
        summary: str,
        original_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """배치 모드용 요약 검증 (엄격한 기준 적용)"""
        return self._validate_summary_internal(
            summary,
            original_text,
            self.check_semantic_density,
            self.accept_threshold,
            semantic_penalty=0.20  # 배치는 엄격한 페널티
        )

    def validate_batch_results(self,
                             results: List[Dict[str, Any]],
                             original_texts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        total_count = len(results)
        if total_count == 0:
            return {'error': '검증할 결과가 없습니다.'}

        validation_results: List[Dict[str, Any]] = []
        stats = {
            'total': total_count,
            'format_correct': 0,
            'length_valid': 0,
            'acceptable': 0,
            'warning_count': 0,
            'quality_scores': []
        }

        for i, result in enumerate(results):
            call_id = result.get('call_id', f'unknown_{i}')
            summary = result.get('summary_text', '') or result.get('summary', '')
            original = original_texts.get(call_id) if original_texts else None

            validation = self.validate_summary(summary, original)
            validation['call_id'] = call_id
            validation_results.append(validation)

            if validation['format_check']['has_format']:
                stats['format_correct'] += 1
            if validation['length_check']['is_valid']:
                stats['length_valid'] += 1
            if validation['is_acceptable']:
                stats['acceptable'] += 1
            stats['warning_count'] += len(validation['warnings'])
            stats['quality_scores'].append(validation['quality_score'])

        if stats['quality_scores']:
            stats['avg_quality_score'] = sum(stats['quality_scores']) / len(stats['quality_scores'])
        else:
            stats['avg_quality_score'] = 0.0

        return {
            'stats': stats,
            'validation_results': validation_results
        }


def generate_validation_report(self,
                             validation_data: Dict[str, Any],
                             model_name: str = 'Unknown') -> str:
    if 'error' in validation_data:
        return f"검증 불가: {validation_data['error']}"

    stats = validation_data['stats']
    results = validation_data['validation_results']

    lines = [
        '# Batch 요약 품질 검증 리포트',
        '',
        '## 기본 정보',
        f"- **모델명**: {model_name}",
        f"- **검증 샘플 수**: {stats['total']}건",
        '',
        '## 핵심 지표',
        f"- **포맷 적합**: {stats['format_correct']}/{stats['total']} ({stats['format_correct']/stats['total']*100:.1f}%)",
        f"- **길이 적정**: {stats['length_valid']}/{stats['total']} ({stats['length_valid']/stats['total']*100:.1f}%)",
        f"- **품질 기준 통과**: {stats['acceptable']}/{stats['total']} ({stats['acceptable']/stats['total']*100:.1f}%)",
        f"- **평균 품질 점수**: {stats['avg_quality_score']:.2f}/1.00",
        f"- **경고 누적 건수**: {stats['warning_count']}건",
        ''
    ]

    if results:
        lines.append('## 세부 결과')
        for validation in results:
            call_id = validation.get('call_id', 'unknown')
            lines.append(f"### 콜 ID: {call_id}")
            lines.append(f"- 품질 점수: {validation['quality_score']:.2f}")
            lines.append(f"- 포맷 적합: {validation['format_check']['has_format']}")
            lines.append(f"- 길이 적정: {validation['length_check']['is_valid']}")
            lines.append(f"- 의미 밀도 통과: {validation['semantic_check']['is_valid']}")
            if validation['warnings']:
                lines.append(f"- 경고: {'; '.join(validation['warnings'])}")
            else:
                lines.append('- 경고: 없음')
            lines.append('')

    return '\n'.join(lines)



quality_validator = QualityValidator()
