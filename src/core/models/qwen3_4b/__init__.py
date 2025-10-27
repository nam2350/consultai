"""
Qwen3-4B-Instruct-2507 모델 패키지

이 패키지는 Qwen3-4B 모델의 모든 기능을 포함합니다:
- summarizer: 상담 대화 요약
- classifier: 카테고리 분류 
- title_generator: 제목 생성
"""

from .summarizer import Qwen2507Summarizer

__all__ = ['Qwen2507Summarizer']