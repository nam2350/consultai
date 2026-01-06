"""Qwen3-1.7B realtime summarizer."""

import time
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MIN_NEW_TOKENS = 16  # Ensures a short but non-empty completion.


class Qwen3Summarizer:
    """Generate lightweight summaries with Qwen3-1.7B (realtime tier)."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()

    def load_model(self) -> bool:
        """Compatibility shim for callers expecting explicit load."""
        return self.model is not None and self.tokenizer is not None

    def _load_model(self) -> None:
        """Load tokenizer and model from local storage only."""
        try:
            print(f"[LOADING] Qwen3-1.7B 로딩... ({self.device})")
            start_time = time.time()

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )

            self.model.eval()

            load_time = time.time() - start_time
            print(f"[SUCCESS] Qwen3-1.7B 로드 완료 ({load_time:.2f}s)")

        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[ERROR] Qwen3-1.7B 로드 실패: {exc}")
            raise

    def summarize_consultation(
        self,
        conversation_text: str,
        max_input_length: int = 12_000,
    ) -> Dict[str, Any]:
        """Return a short consultation summary for real-time use."""

        start_time = time.time()

        try:
            # ⚠️ 프롬프트 노출 완전 방지: 빈 대화는 처리 거부
            if not conversation_text or len(conversation_text.strip()) < 10:
                return {
                    "success": False,
                    "summary": "",
                    "processing_time": time.time() - start_time,
                    "model_name": "Qwen3-1.7B",
                    "input_length": len(conversation_text),
                    "error": "대화 내용이 없어서 프롬프트 노출을 방지하기 위해 요약하지 않습니다"
                }

            truncated_text = conversation_text[:max_input_length]

            # 프롬프트 노출 방지 강화 + 품질 개선
            messages = [
                {
                    "role": "user",
                    "content": (
                        "다음 고객상담 내용을 한 문장으로 요약하되, 고객의 구체적 문제와 상담사의 처리결과를 포함해주세요.\n"
                        "형식: 고객이 [구체적 문제]로 문의하여, 상담사가 [처리방법]을 안내하고 [결과] 처리됨\n\n"
                        f"상담 내용:\n{truncated_text}\n\n"
                        "요약:"
                    ),
                }
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            model_inputs = self.tokenizer([prompt], return_tensors="pt")

            # 안정성을 위해 모든 텐서를 명시적으로 올바른 디바이스로 이동
            model_inputs = {key: tensor.to(self.device) for key, tensor in model_inputs.items()}

            generation_params = {
                "max_new_tokens": 128,
                "min_new_tokens": MIN_NEW_TOKENS,
                "temperature": 0.7,  # Hugging Face recommended non-thinking defaults
                "top_p": 0.8,
                "top_k": 20,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    **generation_params,
                )

            generated_tokens = generated_ids[0][len(model_inputs["input_ids"][0]):]
            summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # 프롬프트 노출 완전 차단 + 품질 개선
            summary = self._clean_prompt_echo(summary)
            summary = self._improve_summary_quality(summary)

            if len(summary) > 200:
                summary = summary[:197] + "..."

            processing_time = time.time() - start_time

            # 프롬프트 노출 검사 추가
            success = (bool(summary.strip()) and
                      not self._contains_prompt_echo(summary))

            return {
                "success": success,
                "summary": summary if success else "",
                "processing_time": processing_time,
                "model_name": "Qwen3-1.7B",
                "input_length": len(conversation_text),
                "truncated_length": len(truncated_text),
            }

        except Exception as exc:  # pragma: no cover - defensive logging
            processing_time = time.time() - start_time
            print(f"[ERROR] Qwen3 요약 생성 실패: {exc}")
            return {
                "success": False,
                "summary": "",
                "processing_time": processing_time,
                "model_name": "Qwen3-1.7B",
                "error": str(exc),
            }

    def _clean_prompt_echo(self, summary: str) -> str:
        """프롬프트 에코 제거"""
        import re

        # 프롬프트 패턴 제거
        prompt_patterns = [
            "콜센터 상담 내용을",
            "간단하게 요약해",
            "요약해 주세요",
            "요약하세요",
            "아래",
            "실제 대화 내용만을",
            "프롬프트나 지시문을",
            "위 내용을",
            "한 문장으로"
        ]

        for pattern in prompt_patterns:
            summary = re.sub(rf".*{re.escape(pattern)}[^.]*[.:]\s*", "", summary, flags=re.IGNORECASE)

        return summary.strip()

    def _improve_summary_quality(self, summary: str) -> str:
        """요약 품질 개선 후처리"""
        if not summary:
            return summary

        # 1. 불완전한 문장 보완
        if not summary.endswith(('.', '됨', '함', '음', '다')):
            if '처리' in summary or '해결' in summary or '안내' in summary:
                summary += "됨"
            else:
                summary += "."

        # 2. 더 구체적인 표현으로 개선
        improvements = {
            "문의했": "문의하여",
            "요청했": "요청하여",
            "처리했": "처리하고",
            "해결했": "해결하고",
            "안내했": "안내하여",
            "설명했": "설명하고"
        }

        for old, new in improvements.items():
            summary = summary.replace(old, new)

        # 3. 길이 최적화 (30-120자 범위)
        if len(summary) < 30:
            # 너무 짧은 경우는 그대로 유지 (속도 우선)
            pass
        elif len(summary) > 150:
            # 문장 경계에서 자르기
            sentences = summary.split(', ')
            summary = sentences[0]
            if len(summary) < 50 and len(sentences) > 1:
                summary += ", " + sentences[1]

        return summary.strip()

    def _contains_prompt_echo(self, summary: str) -> bool:
        """프롬프트 노출 여부 검사"""
        echo_patterns = [
            "콜센터 상담 내용을",
            "간단하게 요약해",
            "요약해 주세요",
            "요약하세요",
            "요약:",
            "내용을 요약하면",
            "대화 내용:",
            "위 내용을",
            "고객상담 내용을",
            "형식:"
        ]

        for pattern in echo_patterns:
            if pattern in summary:
                print(f"[PROMPT_ECHO_DETECTED] 패턴 감지: {pattern}")
                return True
        return False

    def cleanup(self) -> None:
        """Release model resources."""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[CLEANUP] Qwen3-1.7B 메모리 정리 완료")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[WARNING] Qwen3 정리 중 경고: {exc}")


if __name__ == "__main__":
    print("[ERROR] 가벼운 요약기는 스크립트로 직접 실행하지 마세요")
    print("[INFO] scripts/local_test_selective_ai.py 를 통해 호출해 주세요")
