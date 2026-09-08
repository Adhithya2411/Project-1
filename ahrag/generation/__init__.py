"""Answer generation: deterministic extractive default, optional LLM adapter."""

from .base import GeneratedAnswer, Generator, build_generator
from .extractive import ExtractiveGenerator
from .prompts import EVIDENCE_ONLY_SYSTEM_PROMPT, build_user_prompt

__all__ = [
    "EVIDENCE_ONLY_SYSTEM_PROMPT",
    "ExtractiveGenerator",
    "GeneratedAnswer",
    "Generator",
    "build_generator",
    "build_user_prompt",
]
