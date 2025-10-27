"""
Midm-2.0-Base-Instruct 전용 제목 생성기
- 한국어 중심 AI 11.5B 모델 (KT 개발)
- 한국 사회문화 컨텍스트 이해 특화
- 확정적 생성: do_sample=False (샘플링 없음)
- 간결하고 명확한 제목 생성
"""

import logging
import re
from typing import List, Dict, Any
import torch
from ..summary_utils import build_keyword_title_from_text, extract_descriptive_title_candidate

logger = logging.getLogger(__name__)

class TitleGenerator:
    """Midm-2.0-Base 제목 생성기 (한국어 최적화)"""

    def __init__(self, shared_summarizer=None):
        """
        Args:
            shared_summarizer: 공유 Midm-Base summarizer 인스턴스 (메모리 최적화)
        """
        self.summarizer = shared_summarizer
        self.model = None
        self.tokenizer = None

        # Midm-Base 11.5B 공식 파라미터 (확정적 생성)
        self.generation_params = {
            'max_new_tokens': 80,        # 제목 생성용 권장 토큰
            'do_sample': False,          # 샘플링 비활성
            'temperature': 0.25,         # 개발사 권장
            'top_p': 0.9,               # 누적 확률 컷
            'repetition_penalty': 1.05, # 반복 방지
            'early_stopping': True,
            'pad_token_id': None,
            'eos_token_id': None
        }

    def generate(self, conversation_text: str) -> List[Dict[str, Any]]:
        """제목 생성 (한국어 최적화)"""
        try:
            if not self.summarizer:
                logger.error("[제목생성] Midm-Base summarizer가 초기화되지 않았습니다")
                return []

            # 공유 모델 사용
            if not self.summarizer.model:
                logger.warning("[제목생성] Midm-Base 모델이 로드되지 않았습니다")
                return []

            self.model = self.summarizer.model
            self.tokenizer = self.summarizer.tokenizer

            # 한국어 최적화 프롬프트 (간결)
            system_prompt = """당신은 상담 대화에서 핵심 제목을 작성하는 전문가입니다. 아래 상담 내용을 모두 읽고, 대화에 근거한 제목 1개를 작성하세요.

작성 규칙:
10~20자 이내의 간결한 문장
- 실제 대화에 없는 정보나 추측을 추가하지 말 것
- 생각 과정이나 내부 태그는 출력하지 말 것
"""

            # 대화 내용 요약 (경량 모델 - 더 짧게)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"상담 내용:\n{conversation_text[:500]}\n\n제목:"}
            ]

            # 프롬프트 생성
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize prompt for title generation
            encoding = self.tokenizer(
                [text],
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False
            ).to(self.model.device)
            encoding.pop('token_type_ids', None)

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.generation_params['pad_token_id'] = self.tokenizer.pad_token_id
            self.generation_params['eos_token_id'] = self.tokenizer.eos_token_id

            input_ids = encoding['input_ids']
            attention_mask = encoding.get('attention_mask')
            gen_kwargs = dict(self.generation_params)
            if attention_mask is not None:
                gen_kwargs['attention_mask'] = attention_mask

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids,
                    **gen_kwargs
                )

            generated_ids = generated_ids[0][len(input_ids[0]):]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # 제목 추출
            titles = self._extract_titles(response)
            formatted_titles = self._format_titles(titles)

            fallback_titles = self._fallback_titles(conversation_text)
            if not formatted_titles:
                formatted_titles = fallback_titles
            elif fallback_titles:
                existing_types = {item['type'] for item in formatted_titles}
                for candidate in fallback_titles:
                    if candidate['type'] not in existing_types and len(formatted_titles) < 2:
                        formatted_titles.append(candidate)
                        existing_types.add(candidate['type'])
                    if len(formatted_titles) >= 2:
                        break

            logger.info(f"[제목생성] Midm-Base 생성 제목: {[item['title'] for item in formatted_titles]}")
            return self._select_primary_title(formatted_titles)

        except Exception as e:
            logger.error(f"[제목생성] Midm-Base 제목 생성 실패: {e}")
            return []

    def _format_titles(self, titles: List[str]) -> List[Dict[str, Any]]:
        """제목 문자열을 구조화된 dict 리스트로 변환"""
        formatted = []
        for idx, title in enumerate(titles, start=1):
            if not title:
                continue
            title_type = 'keyword' if '_' in title and title.replace('_', '') else 'descriptive'
            confidence = max(0.5, 0.85 - (idx - 1) * 0.1)
            formatted.append({
                "title": title.strip(),
                "type": title_type,
                "confidence": confidence,
                "source": "llm"
            })
        return formatted[:2]

    def _extract_titles(self, response: str) -> List[str]:
        """응답에서 제목 추출 (한국어 최적화)"""
        titles = []

        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()

            # 번호나 라벨 제거
            if ':' in line:
                label, content = line.split(':', 1)
                content = content.strip()

                if any(keyword in label.lower() for keyword in ['키워드', 'keyword', '1']):
                    # 키워드형 제목
                    title = self._clean_keyword_title(content)
                    if title:
                        titles.append(title)
                elif any(keyword in label.lower() for keyword in ['설명', 'description', '2']):
                    # 설명형 제목
                    title = self._clean_description_title(content)
                    if title:
                        titles.append(title)
            elif '_' in line:
                # 언더바가 있으면 키워드형
                title = self._clean_keyword_title(line)
                if title:
                    titles.append(title)
            elif 5 <= len(line) <= 30:
                # 적절한 길이면 설명형
                title = self._clean_description_title(line)
                if title:
                    titles.append(title)

        # 중복 제거 및 최대 2개 반환
        seen = set()
        unique_titles = []
        for title in titles:
            if title not in seen:
                seen.add(title)
                unique_titles.append(title)
                if len(unique_titles) >= 2:
                    break

        return unique_titles

    def _select_primary_title(self, titles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pick a single best title entry for downstream consumers"""
        if not titles:
            return []
        best = None
        best_score = float('-inf')
        for candidate in titles:
            if not isinstance(candidate, dict):
                continue
            score = candidate.get('confidence', 0.0)
            if best is None or score > best_score:
                best = candidate
                best_score = score
        if best is None:
            best = next((t for t in titles if isinstance(t, dict)), None)
        return [best] if best else []

    def _fallback_titles(self, conversation_text: str) -> List[Dict[str, Any]]:
        """Disable rule-based fallback to avoid synthetic titles."""
        return []


    def _clean_keyword_title(self, text: str) -> str:
        """키워드형 제목 정리"""
        # 특수문자 제거 (언더바 제외)
        text = re.sub(r'[^\w가-힣\s_]', '', text)

        # 공백을 언더바로
        text = text.replace(' ', '_')

        # 중복 언더바 제거
        text = re.sub(r'_{2,}', '_', text)

        # 앞뒤 언더바 제거
        text = text.strip('_')

        # 너무 짧거나 긴 경우 제외
        if 3 <= len(text) <= 30:
            return text
        return ""

    def _clean_description_title(self, text: str) -> str:
        """설명형 제목 정리"""
        # 특수문자 제거
        text = re.sub(r'["\'\(\)\[\]{}]', '', text)

        # 앞뒤 공백 제거
        text = text.strip()

        # 마침표 제거
        text = text.rstrip('.')

        # 길이 체크
        if 5 <= len(text) <= 30:
            return text
        elif len(text) > 30:
            # 너무 길면 자르기
            return text[:27] + "..."
        return ""
