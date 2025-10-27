"""
Qwen3-4B 기반 카테고리 분류기

상담 내용을 분석하여 적절한 카테고리를 1-3순위로 추천합니다.
"""
import logging
import torch
from typing import List, Dict, Any
from .summarizer import Qwen2507Summarizer
# generation_utils 제거 - 직접 파라미터 사용으로 변경

logger = logging.getLogger(__name__)

class CategoryClassifier:
    """Qwen3-4B를 사용한 카테고리 분류기"""
    
    def __init__(self, model_path: str = None, shared_summarizer: 'Qwen2507Summarizer' = None):
        """
        Args:
            model_path: Qwen 모델 경로 (shared_summarizer 사용 시 무시됨)
            shared_summarizer: 공유할 Qwen2507Summarizer 인스턴스 (성능 최적화용)
        """
        if shared_summarizer is not None:
            # 공유 summarizer 사용 (성능 최적화)
            self.summarizer = shared_summarizer
            self.model_loaded = True  # 이미 로드된 모델 사용
            self.use_shared_model = True  # 공유 모델 플래그
            # logger.debug("[CategoryClassifier] 공유 summarizer 사용")  # 프로덕션 최적화
        else:
            # 독립 summarizer 생성 (기존 방식)
            self.summarizer = Qwen2507Summarizer(model_path)
            self.model_loaded = False
            self.use_shared_model = False
        
        # generation_params 제거 - summarizer의 파라미터 사용
    
    def load_model(self) -> bool:
        """모델 로드"""
        if not self.model_loaded:
            success = self.summarizer.load_model()
            self.model_loaded = success
            return success
        return True
    
    def classify(self, content: str) -> List[Dict[str, Any]]:
        """상담분류 1~3순위 추천"""
        try:
            # 공유 모델 사용 시 로드 체크 완전 생략 (성능 최적화)
            if self.use_shared_model:
                # 공유 모델은 이미 로드된 것으로 간주
                pass
            elif not self.model_loaded:
                if not self.load_model():
                    raise RuntimeError("모델 로드 실패")
            
            # 범용적 카테고리 분류 프롬프트
            category_prompt = self._build_classification_prompt(content)

            # Qwen 모델을 사용하여 추론
            result = self._generate_response(category_prompt)
            
            # 결과 파싱
            categories = self._parse_category_recommendations(result)
            
            return categories
            
        except Exception as e:
            logger.error(f"[카테고리추천] 실패: {e}")
            # 가짜정보 생성 완전 금지 - 실패시 빈 리스트 반환
            return []
    
    def _generate_response(self, prompt: str) -> str:
        """모델 응답 생성 - WhisperX_web 최적화된 방식"""
        try:
            # 요약기의 최적화된 생성 방식 직접 사용
            model_inputs = self.summarizer.tokenizer([prompt], return_tensors="pt").to(self.summarizer.device)
            
            # WhisperX_web과 동일한 생성 방식 사용
            with torch.no_grad():
                generated_ids = self.summarizer.model.generate(
                    model_inputs['input_ids'],
                    attention_mask=model_inputs['attention_mask'],
                    max_new_tokens=160,  # 키워드 안정성 강화 (140→160)
                    do_sample=True,
                    temperature=0.7,
                    top_k=20,
                    top_p=0.8,
                    min_p=0.0,
                    repetition_penalty=1.05,
                    early_stopping=True,
                    pad_token_id=self.summarizer.tokenizer.pad_token_id,
                    eos_token_id=self.summarizer.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # 디코딩 (요약기와 동일한 방식)
            response_ids = generated_ids[0][len(model_inputs['input_ids'][0]):]
            response = self.summarizer.tokenizer.decode(response_ids, skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            logger.error(f"[응답생성] 실패: {e}")
            raise
    
    def _build_classification_prompt(self, content: str) -> str:
        """단순화된 키워드 추출 프롬프트"""
        system_prompt = (
            "상담 대화에서 핵심 키워드 3개를 찾아주세요.\n\n"
            "규칙:\n"
            "- 실제 언급된 구체적인 단어만 사용\n"
            "- 시스템명, 업무명, 절차명 우선\n"
            "- '문의', '안내', '상담' 같은 일반적인 단어는 제외\n\n"
            "형식:\n"
            "1순위: [키워드] \n"
            "2순위: [키워드] \n"
            "3순위: [키워드] "
        )
        
        return f"{system_prompt}\n\n상담내용:\n{content}"
    
    def _parse_category_recommendations(self, raw_result: str) -> List[Dict[str, Any]]:
        """범용적 카테고리 추천 결과 파싱"""
        try:
            # Human 텍스트 누출 감지 및 차단
            if "Human:" in raw_result or "당신은" in raw_result:
                logger.warning("[카테고리분류] 프롬프트 누출 감지, 빈 결과 반환")
                return []
            
            categories = []
            lines = raw_result.strip().split('\n')
            
            for line in lines:
                # "1순위:", "2순위:", "3순위:" 패턴 찾기
                for rank in range(1, 4):
                    if f"{rank}순위:" in line:
                        try:
                            # "1순위: 분류명 - 이유" 형태 파싱
                            parts = line.split(":", 1)
                            if len(parts) >= 2:
                                content_part = parts[1].strip()
                                
                                # 분류명과 이유 분리
                                if " - " in content_part:
                                    name, reason = content_part.split(" - ", 1)
                                else:
                                    name = content_part
                                    reason = f"{rank}순위 추천 분류"
                                
                                name = name.strip()
                                reason = reason.strip()
                                
                                if name:  # 빈 문자열이 아닌 경우만
                                    confidence = 1.0 - (rank - 1) * 0.2  # 1순위: 1.0, 2순위: 0.8, 3순위: 0.6
                                    categories.append({
                                        "rank": rank,
                                        "name": name,
                                        "confidence": confidence,
                                        "reason": reason
                                    })
                        except Exception as e:
                            # logger.debug(f"[파싱] {rank}순위 라인 파싱 실패: {e}")  # 프로덕션 최적화
                            continue
                        break
            
            # 중복 제거 - 동일한 순위의 중복 항목 제거
            seen_ranks = set()
            unique_categories = []
            for cat in categories:
                rank = cat['rank']
                if rank not in seen_ranks:
                    seen_ranks.add(rank)
                    unique_categories.append(cat)
            
            # 최대 3개까지만 반환
            return unique_categories[:3]
            
        except Exception as e:
            logger.error(f"[카테고리파싱] 실패: {e}")
            # 가짜정보 생성 완전 금지 - 실패시 빈 리스트 반환
            return []
    
    def cleanup(self):
        """리소스 정리 (공유 모델은 정리하지 않음)"""
        # 공유 summarizer를 사용하는 경우 cleanup하지 않음
        if hasattr(self, 'summarizer') and self.summarizer and not hasattr(self.summarizer, '_is_shared'):
            self.summarizer.cleanup()
        self.model_loaded = False