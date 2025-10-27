"""
Midm-2.0-Base 기반 카테고리 분류기

상담 내용을 분석하여 적절한 카테고리를 1-3순위로 추천합니다.
가짜정보 생성 완전 금지 - 실제 대화에서 언급된 내용만 추출
"""
import logging
import torch
from typing import List, Dict, Any
from .summarizer import MidmBaseSummarizer
from ..summary_utils import extract_consultation_keywords

logger = logging.getLogger(__name__)

class CategoryClassifier:
    """Midm-2.0-Base를 사용한 카테고리 분류기"""

    def __init__(self, model_path: str = None, shared_summarizer: 'MidmBaseSummarizer' = None):
        """
        Args:
            model_path: Midm 모델 경로 (shared_summarizer 사용 시 무시됨)
            shared_summarizer: 공유할 MidmBaseSummarizer 인스턴스 (성능 최적화용)
        """
        if shared_summarizer is not None:
            # 공유 summarizer 사용 (성능 최적화)
            self.summarizer = shared_summarizer
            self.model_loaded = True  # 이미 로드된 모델 사용
            self.use_shared_model = True  # 공유 모델 플래그
        else:
            # 독립 summarizer 생성 (기존 방식)
            self.summarizer = MidmBaseSummarizer(model_path)
            self.model_loaded = False
            self.use_shared_model = False

    def load_model(self) -> bool:
        """모델 로드"""
        if not self.model_loaded:
            success = self.summarizer.load_model()
            self.model_loaded = success
            return success
        return True

    def classify(self, content: str) -> List[Dict[str, Any]]:
        """상담분류 1~3순위 추천 - 가짜정보 생성 금지"""
        try:
            # 공유 모델 사용 시 로드 체크 완전 생략 (성능 최적화)
            if self.use_shared_model:
                pass
            elif not self.model_loaded:
                if not self.load_model():
                    return []

            # 명확한 프롬프트로 키워드 추출
            system_prompt = """상담 대화에서 실제로 언급된 구체적 키워드 3개를 추출하세요.

규칙:
1. 대화에서 직접 언급된 단어만 사용
2. 시스템명, 업무명, 절차명 우선
3. 금지: 문의, 안내, 여기서, 그다음, 상담

형식:
1순위: [키워드]
2순위: [키워드]
3순위: [키워드]"""

            user_prompt = f"""다음 상담에서 핵심 키워드 3개를 추출:

{content[:2000]}

키워드:"""

            # Midm 모델로 키워드 추출
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # 토큰화 및 생성
            chat_prompt = self.summarizer.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.summarizer.tokenizer(
                chat_prompt,
                return_tensors="pt",
                return_attention_mask=True
            )

            if inputs['input_ids'].shape[-1] > 2048:
                inputs['input_ids'] = inputs['input_ids'][:, -2048:]
                if 'attention_mask' in inputs:
                    inputs['attention_mask'] = inputs['attention_mask'][:, -2048:]

            inputs = {k: v.to(self.summarizer.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.summarizer.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=80,  # 키워드만 추출
                    do_sample=False,    # 개발사 권장 설정
                    repetition_penalty=1.05,
                    pad_token_id=self.summarizer.tokenizer.pad_token_id,
                    eos_token_id=self.summarizer.tokenizer.eos_token_id
                )

            response = self.summarizer.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:],
                skip_special_tokens=True
            )

            # 응답 파싱
            categories = self._parse_keywords_response(response)

            # 파싱 실패 시 기본 키워드 추출 사용
            if not categories:
                keywords = extract_consultation_keywords(content, max_keywords=3)
                for idx, keyword in enumerate(keywords[:3], 1):
                    categories.append({
                        'rank': idx,
                        'name': keyword,
                        'confidence': max(0.45, 0.8 - (idx - 1) * 0.1),
                        'reason': f"상담 본문에서 핵심 키워드로 추출"
                    })

            return categories

        except Exception as e:
            logger.error(f"분류 중 오류: {e}")
            return []  # 실패 시 빈 리스트 반환 (가짜정보 생성 금지)

    def _parse_keywords_response(self, response: str) -> List[Dict[str, Any]]:
        """키워드 응답 파싱 - 무의미한 키워드 필터링"""
        categories = []

        # 무의미한 키워드 필터 (최소한만)
        meaningless_words = {'여기서', '그다음', '그리고', '하지만', '그런데', '그래서'}

        lines = response.strip().split('\n')
        for line in lines:
            if '순위:' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    keyword = parts[1].strip().strip('[]')
                    # 무의미한 단어 제거
                    if keyword and keyword not in meaningless_words and len(keyword) > 1:
                        rank = len(categories) + 1
                        if rank <= 3:
                            categories.append({
                                'rank': rank,
                                'name': keyword,
                                'confidence': max(0.45, 0.8 - (rank - 1) * 0.1),
                                'reason': f"상담 본문에서 핵심 키워드로 추출"
                            })

        return categories

    def cleanup(self):
        """리소스 정리"""
        if not self.use_shared_model and hasattr(self, 'summarizer'):
            self.summarizer.cleanup()

# Midm_Base_Classifier 별칭 추가 (호환성)
Midm_Base_Classifier = CategoryClassifier
