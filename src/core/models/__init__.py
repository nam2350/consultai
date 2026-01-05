"""AI model package exports."""

# SLM models (real-time)
from .qwen3_1_7b import Qwen3Summarizer

# LLM models (batch/high-quality)
from .qwen3_4b import Qwen2507Summarizer

__all__ = [
    # SLM
    'Qwen3Summarizer',

    # LLM
    'Qwen2507Summarizer',
]
