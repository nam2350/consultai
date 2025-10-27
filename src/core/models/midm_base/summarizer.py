"""
Midm-2.0-Base-Instruct MLM summarizer for high-quality Korean consultation analysis.
- 11.5B parameters optimized for Korean/English
- BF16 precision for optimal performance
- KT's Mi:dm specialized instruction following
"""

import os
import time
import logging
import re
from typing import Dict, Any, Tuple, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from ..summary_utils import normalize_summary

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*pad_token_id.*eos_token_id.*")

# RTX 5080 optimization
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


class MidmBaseSummarizer:
    """High-quality Korean consultation summarizer using Midm-2.0-Base (LLM tier)."""

    def __init__(self, model_path: str = r"models\Midm-2.0-Base"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        # 안전한 디바이스 감지
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = "cuda"
            else:
                self.device = "cpu"
        except Exception:
            self.device = "cpu"

        # Midm-2.0-Base parameters - 요약 완성도를 위해 토큰 증가
        self.min_generation_tokens = 80  # 65 -> 80 증가
        self.max_generation_tokens = 250  # 120 -> 250 증가 (완전한 요약 보장)
        self.max_conversation_chars = 3400
        self.head_chars = 2300
        self.tail_chars = 1100
        self.generation_params = {
            'max_new_tokens': self.max_generation_tokens,
            'min_new_tokens': self.min_generation_tokens,
            'do_sample': False,
            'repetition_penalty': 1.05,
            'pad_token_id': None,
            'eos_token_id': None,
        }

        self.min_summary_chars = 160

    def load_model(self) -> bool:
        """Load Midm-2.0-Base model with KT optimizations."""
        try:
            start_time = time.time()
            logger.info(f"Loading Midm-2.0-Base from {self.model_path}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )

            # Load generation config
            try:
                self.generation_config = GenerationConfig.from_pretrained(
                    self.model_path,
                    local_files_only=True
                )
            except Exception as e:
                logger.warning(f"Could not load generation config: {e}")
                self.generation_config = None

            # Pad token setup
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = "<|endoftext|>"

            # Model loading with safer device handling
            try:
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    model_dtype = torch.bfloat16  # KT specification
                    device_map = "auto"
                    max_memory = {"0": "14GB"}    # RTX 5080 optimization
                else:
                    model_dtype = torch.float32
                    device_map = None
                    max_memory = None
            except Exception:
                # GPU 설정 실패시 CPU로 폴백
                model_dtype = torch.float32
                device_map = None
                max_memory = None
                self.device = "cpu"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                local_files_only=True,
                device_map="auto" if self.device == "cuda" else None
            )

            self.model.eval()

            # Ensure generation config aligns with deterministic settings
            if self.generation_config:
                self.generation_config.do_sample = False
                self.generation_config.temperature = None
                self.generation_config.top_p = None
                self.generation_config.top_k = None
            # Set generation parameters
            self.generation_params['pad_token_id'] = self.tokenizer.pad_token_id
            self.generation_params['eos_token_id'] = self.tokenizer.eos_token_id

            load_time = time.time() - start_time
            logger.info(f"Midm-2.0-Base loaded successfully in {load_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to load Midm-2.0-Base: {str(e)}")
            return False

    def _determine_max_tokens(self, conversation_text: str) -> int:
        """Determine dynamic max_new_tokens for Midm base model."""
        length = len(conversation_text or "")
        # 더 많은 토큰 할당으로 완성도 향상
        bonus = min(100, max(0, length // 100))  # 보너스를 더 관대하게
        return min(self.min_generation_tokens + bonus, self.max_generation_tokens)

    def _truncate_conversation(self, conversation_text: str) -> str:
        if not conversation_text:
            return ""
        if len(conversation_text) <= self.max_conversation_chars:
            return conversation_text
        head = conversation_text[: self.head_chars].rstrip()
        tail = conversation_text[-self.tail_chars :].lstrip()
        return head + "\n...\n" + tail

    def build_chat_messages(self, conversation_text: str) -> list:
        """Build KT Mi:dm optimized chat messages."""
        system_message = (
            "당신은 콜센터 상담 대화를 요약하는 전문 AI입니다.\n\n"
            "입력 대화 분석:\n"
            "- 상담사와 고객의 화자 분리가 적용된 실제 상담 대화입니다\n"
            "- 화자 표시 오류가 있을 수 있으므로 전체 맥락으로 판단하세요\n"
            "- 대화 흐름을 파악하여 올바른 화자를 추론하고 요약하세요\n\n"
            "요약 핵심 원칙:\n"
            "1. 원본 대화에서 실제 언급된 내용만 요약\n"
            "2. 핵심 용어, 시스템명, 절차명, 기관명은 정확히 기재\n"
            "3. 고객의 질문과 상담사의 답변을 명확히 구분\n"
            "4. 추측, 가정, 상상은 절대 포함 금지\n\n"
            "출력 형식 (반드시 3줄 구조로 작성):\n"
            "**고객**: 고객의 핵심 질문이나 요청사항\n"
            "**상담사**: 상담사의 핵심 안내나 해결방안\n"
            "**상담결과**: 상담의 최종 결과나 처리상태\n\n"
            "각 줄은 완전한 문장으로 작성하며, 위 형식을 정확히 준수하세요."
        )

        # user_message는 대화 텍스트만 전달 (지시사항 중복 제거)
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": conversation_text}
        ]

    def generate_summary_fast(self, conversation_text: str) -> Tuple[str, float]:
        """Generate high-quality summary using Midm-2.0-Base."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prevent fake info generation for empty conversations
        if not conversation_text or len(conversation_text.strip()) < 10:
            raise RuntimeError("대화 내용이 없어서 요약할 수 없습니다")

        start_time = time.time()

        try:
            # Build chat messages
            truncated_conversation = self._truncate_conversation(conversation_text)
            messages = self.build_chat_messages(truncated_conversation)

            # Build chat prompt and tokenize with attention mask
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer(
                [chat_prompt],
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False
            ).to(self.device)
            model_inputs.pop('token_type_ids', None)
            attention_mask = model_inputs.get('attention_mask')
            input_ids = model_inputs['input_ids']

            generation_kwargs = dict(self.generation_params)
            dynamic_max = self._determine_max_tokens(truncated_conversation)
            generation_kwargs['max_new_tokens'] = dynamic_max
            # min_new_tokens를 더 높게 설정하여 완성도 보장
            generation_kwargs['min_new_tokens'] = max(100, min(dynamic_max - 20, 150))  # 최소 100토큰 보장
            if attention_mask is not None:
                generation_kwargs['attention_mask'] = attention_mask

            output = self.model.generate(
                input_ids,
                **generation_kwargs
            )

            # Decode response
            generated_tokens = output[0][len(input_ids[0]):]
            response = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Post-process for quality
            summary = self.post_process_summary(response, truncated_conversation)

            processing_time = time.time() - start_time
            return summary, processing_time

        except torch.cuda.OutOfMemoryError as e:
            processing_time = time.time() - start_time
            logger.error(f"GPU memory exhausted: {e}")
            raise RuntimeError(f"GPU memory insufficient: {str(e)}")
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Generation error: {e}")
            raise RuntimeError(f"Summary generation failed: {str(e)}")

    def post_process_summary(self, raw_summary: str, source_text: str = "") -> str:
        """Post-process Midm-2.0-Base output for Korean consultation quality."""
        if not raw_summary or len(raw_summary.strip()) < 10:
            raise RuntimeError("Generated summary is empty or too short")

        summary = normalize_summary(
            raw_summary,
            cleanup_patterns=(
                "Mi:dm",
                "??? ?? ??",
                "?? ??",
                "?? ??",
                "?? ??:",
            ),
            min_length=70,
            logger=logger,
            fallback_text=source_text or raw_summary,
        )

        # 미완성 요약 감지 및 완성
        summary = self._complete_unfinished_summary(summary)

        summary = self._extend_summary_if_needed(summary, source_text)

        if self._contains_fake_information(summary, source_text):
            raise RuntimeError("Fake information detected, summary rejected")

        return summary

    def _complete_unfinished_summary(self, summary: str) -> str:
        """미완성된 요약을 감지하고 완성시킴"""
        lines = summary.split('\n')

        # 3줄 구조 확인
        customer_line = None
        agent_line = None
        result_line = None

        for i, line in enumerate(lines):
            if '**고객**' in line:
                customer_line = i
            elif '**상담사**' in line:
                agent_line = i
            elif '**상담결과**' in line:
                result_line = i

        # 상담결과가 없거나 미완성인 경우
        if result_line is not None:
            result_text = lines[result_line].replace('**상담결과**:', '').strip()

            # 미완성 감지 (문장이 끝나지 않음)
            if result_text and not result_text.endswith(('.', '다', '요', '음', '습니다')):
                # 간단한 완성 처리
                if '상황에서' in result_text or '과정에서' in result_text:
                    lines[result_line] = f"**상담결과**: {result_text} 상담을 진행하였습니다."
                elif '확인' in result_text or '처리' in result_text:
                    lines[result_line] = f"**상담결과**: {result_text} 완료되었습니다."
                else:
                    lines[result_line] = f"**상담결과**: {result_text}."

        # 상담결과가 아예 없는 경우
        elif customer_line is not None and agent_line is not None:
            lines.append("**상담결과**: 고객의 문의사항에 대해 상담사가 안내를 완료하였습니다.")

        return '\n'.join(lines)
    def _contains_fake_information(self, summary: str, source_text: str = "") -> bool:
        """Enhanced fake information detection for Midm-2.0-Base."""
        fake_patterns = [
            # Specific dates not in conversation
            r'\d{4}년\d{1,2}월\d{1,2}일',
            # Specific locations
            '서울', '부산', '대전', '인천', '광주', '울산',
            # Weather/disaster events
            '태풍', '지진', '폭우', '침수', '한파', '폭염',
            # Generic institutional references
            '행정안전부', '국무조정실', '교육부', '국방부',
            # Fabricated KT services (unless actually mentioned)
            'KT 고객센터', 'KT 멤버십', 'olleh',
        ]


        reference = source_text or ""

        for pattern in fake_patterns:
            if re.search(pattern, summary):
                if reference and re.search(pattern, reference):
                    continue
                logger.warning(f"[MIDM_FAKE_INFO] Detected pattern: {pattern}")
                return True
        return False
    def _extend_summary_if_needed(self, summary: str, source_text: str) -> str:
        if not source_text or len(summary) >= self.min_summary_chars:
            return summary

        sections = summary.split('\n')
        if len(sections) < 3:
            return summary

        result_idx = next((idx for idx, line in enumerate(sections) if line.startswith('**\uc0c1\ub2f4\uacb0\uacfc**')), None)
        if result_idx is None:
            return summary

        candidates = self._extract_context_sentences(source_text)
        additions: List[str] = []
        for sentence in candidates:
            if sentence in summary:
                continue
            additions.append(sentence)
            if len(' '.join(additions)) >= 80:
                break

        if additions:
            result_line = sections[result_idx].rstrip()
            if result_line and not result_line.endswith(('\ub2e4', '.', '!', '?', '\uc694')):
                result_line = result_line.rstrip() + '.'
            result_line = (result_line + ' ' + ' '.join(additions)).strip()
            sections[result_idx] = result_line
            updated = '\n'.join(sections)
            if len(updated) > len(summary):
                summary = updated
        return summary

    def _extract_context_sentences(self, source_text: str) -> List[str]:
        cleaned = re.sub(r'\.\.\.', ' ', source_text)
        cleaned = re.sub(r'\[[^\]]+\]', ' ', cleaned)
        sentences: List[str] = []
        for chunk in re.split(r'[\r\n]+', cleaned):
            chunk = chunk.strip()
            if not chunk:
                continue
            chunk = re.sub(r'^(\uace0\uac1d|\uc0c1\ub2f4\uc0ac|Agent|\uc0c1\ub2f4\uc6d0)[:\s]+', '', chunk)
            for sentence in re.split(r'(?<=[.!?])\s+', chunk):
                stripped = sentence.strip()
                if len(stripped) < 12:
                    continue
                if stripped in sentences:
                    continue
                sentences.append(stripped)
                if len(sentences) >= 6:
                    return sentences
        return sentences


    def summarize_consultation(self, conversation_text: str, max_length: int = 300) -> Dict[str, Any]:
        """
        High-quality Korean consultation summarization using Midm-2.0-Base.

        Args:
            conversation_text: Consultation conversation text
            max_length: Maximum summary length (compatibility parameter)

        Returns:
            Dict with success, summary, processing_time, model_used, error
        """
        start_time = time.time()

        try:
            # Ensure model is loaded
            if not self.model or not self.tokenizer:
                if not self.load_model():
                    raise RuntimeError("Failed to load Midm-2.0-Base model")

            # Prevent fake info generation for empty conversations
            if not conversation_text or len(conversation_text.strip()) < 10:
                return {
                    'success': False,
                    'summary': '',
                    'processing_time': time.time() - start_time,
                    'model_used': 'Midm-2.0-Base',
                    'error': '대화 내용이 없어서 요약할 수 없습니다'
                }

            # Generate summary
            summary, proc_time = self.generate_summary_fast(conversation_text)

            total_time = time.time() - start_time

            return {
                'success': True,
                'summary': summary,
                'processing_time': total_time,
                'model_used': 'Midm-2.0-Base',
                'error': ''
            }

        except Exception as e:
            total_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Midm-2.0-Base summarization failed: {error_msg}")

            return {
                'success': False,
                'summary': '',
                'processing_time': total_time,
                'model_used': 'Midm-2.0-Base',
                'error': error_msg
            }

    def cleanup(self):
        """Optimized memory cleanup for Midm-2.0-Base."""
        try:
            if self.model:
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                del self.model
                self.model = None

            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None

            if self.generation_config:
                del self.generation_config
                self.generation_config = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            import gc
            gc.collect()

        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


# Standalone function for compatibility
def summarize_with_midm_base(conversation_text: str, model_path: str = r"models\Midm-2.0-Base") -> Dict[str, Any]:
    """
    High-quality Korean summarization with Midm-2.0-Base (LLM tier).

    Args:
        conversation_text: Conversation to summarize
        model_path: Path to Midm-2.0-Base model

    Returns:
        Dict with success, summary, error, processing_time, model_name
    """
    start_time = time.time()
    summarizer = None

    try:
        # Initialize and load model
        summarizer = MidmBaseSummarizer(model_path)

        if not summarizer.load_model():
            raise RuntimeError("Failed to load Midm-2.0-Base model")

        # Generate summary
        summary, proc_time = summarizer.generate_summary_fast(conversation_text)

        return {
            'success': True,
            'summary': summary,
            'processing_time': proc_time,
            'model_name': 'Midm-2.0-Base',
            'error': ''
        }

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        logger.error(f"Midm-2.0-Base standalone summarization failed: {error_msg}")

        return {
            'success': False,
            'summary': '',
            'error': error_msg,
            'processing_time': processing_time,
            'model_name': 'Midm-2.0-Base'
        }

    finally:
        if summarizer:
            summarizer.cleanup()


if __name__ == "__main__":
    print("[ERROR] Midm-2.0-Base 분석기는 직접 실행하지 마세요")
    print("[INFO] MLM 배치 처리 시스템에서 호출됩니다")