"""
A.X-4.0-Light 전용 제목 생성기
- SKT 개발 7.26B 모델 (Llama 아키텍처 기반)
- 확정적 생성: do_sample=False (샘플링 없음)
- 키워드형/설명형 제목 생성
"""

import logging
import re
from typing import List, Dict, Any
import torch
from ..summary_utils import build_keyword_title_from_text, extract_descriptive_title_candidate

logger = logging.getLogger(__name__)

class TitleGenerator:
    """A.X-4.0-Light 제목 생성기 (키워드형/설명형)"""

    def __init__(self, shared_summarizer=None):
        """
        Args:
            shared_summarizer: 공유 A.X-Light summarizer 인스턴스 (메모리 최적화)
        """
        self.summarizer = shared_summarizer
        self.model = None
        self.tokenizer = None

        # A.X-4.0-Light 최적화 파라미터 (확정적 생성)
        self.generation_params = {
            'max_new_tokens': 120,
            'do_sample': False,          # 결정적 생성 (SKT 권장)
            'temperature': 0.2,          # 개발사 권장
            'top_p': 0.9,               # 누적 확률 컷
            'repetition_penalty': 1.05, # 반복 방지
            'early_stopping': True,
            'pad_token_id': None,
            'eos_token_id': None
        }

    def generate(self, conversation_text: str) -> List[Dict[str, Any]]:
        """제목 생성 (확정적 모드)"""
        try:
            if not self.summarizer:
                logger.error("[제목생성] A.X-Light summarizer가 초기화되지 않았습니다")
                return []

            # 공유 모델 사용
            if not self.summarizer.model:
                logger.warning("[제목생성] A.X-Light 모델이 로드되지 않았습니다")
                return []

            self.model = self.summarizer.model
            self.tokenizer = self.summarizer.tokenizer

            # A.X 시리즈 최적화 프롬프트 (간결하고 명확)
            system_prompt = """당신은 상담 대화로부터 제목을 작성하는 전문가입니다. 아래 상담 내용을 모두 읽고, 대화에 근거한 제목 1개를 작성하세요.

작성 지침:
- 10~20자 이내의 간결한 문장
- 상담에 없는 정보나 추측을 추가하지 말 것
- 생각 과정이나 <think> 등의 내부 태그는 출력하지 말 것
"""

            # 대화 내용 요약

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음 상담 대화의 제목을 생성하세요:\n\n{conversation_text}"}
            ]

            # 프롬프트 생성 (Llama 스타일)
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

            logger.info(f"[제목생성] A.X-Light 생성 제목: {[item['title'] for item in formatted_titles]}")
            return self._select_primary_title(formatted_titles)

        except Exception as e:
            logger.error(f"[제목생성] A.X-Light 제목 생성 실패: {e}")
            return []

    def _format_titles(self, titles: List[str]) -> List[Dict[str, Any]]:
        """LLM 응답 문자열을 표준 제목 구조로 변환"""
        formatted = []
        for idx, title in enumerate(titles, start=1):
            if not title:
                continue
            title_type = 'keyword' if '_' in title and title.replace('_', '') else 'descriptive'
            confidence = max(0.5, 0.8 - (idx - 1) * 0.1)
            formatted.append({
                "title": title.strip(),
                "type": title_type,
                "confidence": confidence,
                "source": "llm"
            })
        return formatted[:2]

    def _extract_titles(self, response: str) -> List[str]:
        """응답에서 제목 추출"""
        titles = []

        # 키워드형과 설명형 찾기
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if '키워드형:' in line or '키워드형 :' in line:
                title = line.split(':', 1)[-1].strip()
                title = self._clean_keyword_title(title)
                if title:
                    titles.append(title)
            elif '설명형:' in line or '설명형 :' in line:
                title = line.split(':', 1)[-1].strip()
                title = self._clean_description_title(title)
                if title:
                    titles.append(title)
            elif '_' in line and len(line.split('_')) >= 2:
                # 언더바가 있으면 키워드형으로 간주
                title = self._clean_keyword_title(line)
                if title:
                    titles.append(title)
            elif 6 <= len(line) <= 25 and not line.startswith('#'):
                # 적절한 길이의 문장은 설명형으로 간주
                title = self._clean_description_title(line)
                if title:
                    titles.append(title)

        # 중복 제거
        seen = set()
        unique_titles = []
        for title in titles:
            if title not in seen:
                seen.add(title)
                unique_titles.append(title)

        return unique_titles[:2]  # 최대 2개만 반환

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
        """키워드형 제목의 언더바 정리"""
        # 1. 앞뒤 언더바 제거
        cleaned = text.strip('_')

        # 2. 중복된 언더바를 하나로 통합
        cleaned = re.sub(r'_{2,}', '_', cleaned)

        # 3. 특수문자 제거 (언더바 제외)
        cleaned = re.sub(r'[^\w가-힣\s_]', '', cleaned)

        # 4. 공백을 언더바로 변경
        cleaned = cleaned.replace(' ', '_')

        # 5. 다시 중복 언더바 제거
        cleaned = re.sub(r'_{2,}', '_', cleaned)

        # 6. 길이 체크
        cleaned = cleaned.strip('_')
        if 3 <= len(cleaned) <= 25:
            return cleaned
        return ""

    def _clean_description_title(self, text: str) -> str:
        """설명형 제목 정리"""
        # 불필요한 특수문자 제거
        text = re.sub(r'["\'()\[\]{}]', '', text)

        # 앞뒤 공백 제거
        text = text.strip()

        # 마침표 제거
        text = text.rstrip('.')

        # 길이 체크
        if 6 <= len(text) <= 25:
            return text
        elif len(text) > 25:
            # 너무 길면 자르기
            return text[:22] + "..."
        return ""
