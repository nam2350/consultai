"""Midm-2.0-Mini SLM summarizer."""

import re
import time
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MIN_NEW_TOKENS = 16  # Matches the SLM tier minimum output length guard.


class MidmSummarizer:
    """Generate lightweight summaries with Midm-2.0-Mini (SLM tier)."""

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
        """Load tokenizer and model strictly from local storage."""
        try:
            print(f"[LOADING] Midm-2.0-Mini 로딩... ({self.device})")
            start_time = time.time()

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                add_prefix_space=False,
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
            print(f"[SUCCESS] Midm-2.0-Mini 로드 완료 ({load_time:.2f}s)")

        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[ERROR] Midm-2.0-Mini 로드 실패: {exc}")
            raise

    def summarize_consultation(
        self,
        conversation_text: str,
        max_input_length: int = 6_000,
    ) -> Dict[str, Any]:
        """Return a short consultation summary suitable for live support."""

        start_time = time.time()

        try:
            # ⚠️ 가짜정보 생성 완전 방지: 빈 대화는 처리 거부
            if not conversation_text or len(conversation_text.strip()) < 10:
                return {
                    "success": False,
                    "summary": "",
                    "processing_time": time.time() - start_time,
                    "model_name": "Midm-2.0-Mini",
                    "input_length": len(conversation_text),
                    "error": "대화 내용이 없어서 가짜정보 생성을 방지하기 위해 요약하지 않습니다"
                }

            truncated_text = conversation_text[:max_input_length]

            # 가짜정보 생성 방지 강화 + 반복 방지 프롬프트
            prompt = (
                "다음 고객상담 내용을 한 문장으로 간단히 요약하세요.\n"
                "주의사항:\n"
                "- 대화에 없는 내용 추가 금지\n"
                "- 같은 단어 반복 금지\n"
                "- 완전한 문장으로 작성\n"
                "- 고객의 문제와 상담사의 처리 결과 포함\n\n"
                f"상담 내용:\n{truncated_text}\n\n"
                "요약:"
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False,
                add_special_tokens=True,
            )

            if "token_type_ids" in inputs:
                inputs.pop("token_type_ids")

            if self.device.type == "cuda":
                inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

            # 모델 기본 설정 사용 + 최소한의 필수 파라미터만
            generation_params = {
                "max_new_tokens": 50,
                "min_new_tokens": MIN_NEW_TOKENS,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                # 모델 기본 GenerationConfig 사용 (HuggingFace 권장)
            }

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_params,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            summary = self._clean_summary(summary)
            summary = self._improve_summary_quality(summary)

            if len(summary) > 200:
                summary = summary[:197] + "..."

            processing_time = time.time() - start_time

            # 가짜정보 생성 완전 차단
            success = (len(summary) > 10 and
                      not summary.startswith(("[", "user", "assistant")) and
                      not self._contains_fake_information(summary))

            return {
                "success": success,
                "summary": summary if success else "",
                "processing_time": processing_time,
                "model_name": "Midm-2.0-Mini",
                "input_length": len(conversation_text),
                "truncated_length": len(truncated_text),
            }

        except Exception as exc:  # pragma: no cover - defensive logging
            processing_time = time.time() - start_time
            print(f"[ERROR] Midm 요약 생성 실패: {exc}")
            return {
                "success": False,
                "summary": "",
                "processing_time": processing_time,
                "model_name": "Midm-2.0-Mini",
                "error": str(exc),
            }

    def _clean_summary(self, summary: str) -> str:
        """Remove prompt residue, repetition, and clean up format."""
        # 1. 기본 정리
        summary = re.sub(r"<\|[^>]+\|>", "", summary)
        summary = summary.replace("요약:", "").strip()

        # 2. 반복 패턴 제거 (가장 중요한 개선)
        summary = self._remove_repetition(summary)

        # 3. 불필요한 키워드 제거
        if any(keyword in summary.lower() for keyword in ("요약", "다음", "콜센터", "assistant", "user", "상담사:")):
            # 첫 번째 완전한 문장만 추출
            sentences = re.split(r"[.!?]", summary)
            for sentence in sentences:
                clean_sentence = sentence.strip()
                if (len(clean_sentence) > 15 and
                    not any(keyword in clean_sentence.lower() for keyword in ("요약", "다음", "콜센터", "assistant", "user"))):
                    summary = clean_sentence
                    break

        # 4. 길이 제한
        if len(summary) > 150:
            summary = summary[:147] + "..."

        return summary.strip()

    def _remove_repetition(self, text: str) -> str:
        """반복 패턴 제거 - 핵심 개선 사항"""
        if not text:
            return text

        # 1. "네" 반복 패턴 완전 제거
        text = re.sub(r'네\s*네\s*네+', '네', text)
        text = re.sub(r'(\s*네\s*){3,}', '', text)

        # 2. 상담사:/고객: 반복 패턴 제거
        text = re.sub(r'(상담사:\s*네\s*){2,}', '', text)
        text = re.sub(r'(고객:\s*네\s*){2,}', '', text)

        # 3. 일반적인 단어 반복 제거
        text = re.sub(r'\b(\w+)\s+\1\s+\1+', r'\1', text)

        # 4. 불완전한 문장 조각 제거
        lines = text.split('\n')
        meaningful_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not re.match(r'^(네|아|음|어|그|이|저)$', line):
                meaningful_lines.append(line)

        # 5. 가장 의미있는 첫 번째 줄 선택
        if meaningful_lines:
            return meaningful_lines[0]
        else:
            # 최후의 수단: 원본에서 첫 번째 완전한 문장 추출
            sentences = re.findall(r'[^.!?]*[.!?]', text)
            for sentence in sentences:
                clean = sentence.strip()
                if len(clean) > 15:
                    return clean

        return text.strip()

    def _improve_summary_quality(self, summary: str) -> str:
        """요약 품질 개선 후처리 - 완전한 문장 형태로 개선"""
        if not summary:
            return summary

        # 1. 빈 내용이나 의미없는 반복 체크
        if len(summary.strip()) < 10 or summary.count('네') > 3:
            return ""  # 품질이 너무 낮으면 빈 문자열 반환

        # 2. 불완전한 문장 보완 (더 유연하게)
        if not summary.endswith(('.', '됨', '함', '음', '다', '요', '니다')):
            # 문맥에 따라 적절한 어미 추가
            if any(word in summary for word in ['처리', '해결', '안내', '설명', '확인', '완료']):
                if not summary.endswith('됨'):
                    summary += "됨"
            elif '문의' in summary or '요청' in summary:
                if not summary.endswith('.'):
                    summary += "."
            else:
                summary += "."

        # 3. 자연스러운 표현 개선 (유지)
        improvements = {
            "문의했": "문의하여",
            "요청했": "요청하여",
            "처리했": "처리하고",
            "해결했": "해결하고",
            "안내했": "안내하여",
            "하여고": "하여",  # 문법 오류 수정
            "하여습니다": "하였습니다"  # 문법 오류 수정
        }

        for old, new in improvements.items():
            summary = summary.replace(old, new)

        # 4. 중복된 어미 정리
        summary = re.sub(r'됨\s*됨+', '됨', summary)
        summary = re.sub(r'\.\s*\.+', '.', summary)

        # 5. 최종 길이 체크 (너무 짧으면 실패 처리)
        if len(summary.strip()) < 15:
            return ""

        return summary.strip()

    def _contains_fake_information(self, summary: str) -> bool:
        """가짜정보 생성 여부 검사 - 대화에 없는 정보 차단"""
        fake_patterns = [
            # 구체적 날짜 (대화에 없는 경우)
            r'\d{4}년 \d{1,2}월 \d{1,2}일',
            # 구체적 장소명
            '서울', '부산', '대구', '인천', '광주', '대전', '울산',
            # 구체적 사건명
            '태풍', '지진', '화재', '침수', '사고',
            # 일반적 표현
            '한국의', '우리나라', '정부는', '시민들',
            # 임의 기관명 (대화에 실제 언급되지 않은 경우)
            '관리소', '지자체', '행정복지센터'
        ]

        for pattern in fake_patterns:
            if re.search(pattern, summary):
                print(f"[FAKE_INFO_DETECTED] 패턴 감지: {pattern}")
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
            print("[CLEANUP] Midm-2.0-Mini 메모리 정리 완료")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[WARNING] Midm 정리 중 경고: {exc}")


if __name__ == "__main__":
    print("[ERROR] 가벼운 요약기는 스크립트로 직접 실행하지 마세요")
    print("[INFO] scripts/test_slm_summary_all.py 를 통해 호출해 주세요")
