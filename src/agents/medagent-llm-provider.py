"""
Shared LLM provider for all MedAgent agents.

Supports Anthropic (Claude), OpenAI, and Google Gemini backends.
Selection is based on which API key is available, with priority:
1. ANTHROPIC_API_KEY → Claude
2. OPENAI_API_KEY → GPT-4o-mini
3. GOOGLE_API_KEY → Gemini
"""

from __future__ import annotations

import logging
import os

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger("medagent.agents.llm")


def get_llm(temperature: float = 0.1) -> BaseChatModel:
    """Return the best available LLM based on environment variables."""

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if anthropic_key:
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("MEDAGENT_MODEL", "claude-sonnet-4-20250514")
        logger.info("Using Anthropic model: %s", model)
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=anthropic_key,
            max_tokens=4096,
        )

    if openai_key:
        from langchain_openai import ChatOpenAI

        model = os.getenv("MEDAGENT_MODEL", "gpt-4o-mini")
        logger.info("Using OpenAI model: %s", model)
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=openai_key,
        )

    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = os.getenv("MEDAGENT_MODEL", "gemini-2.5-flash")
            logger.info("Using Google model: %s", model)
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=google_key,
            )
        except ImportError:
            logger.warning("langchain-google-genai not installed, skipping Google")

    raise RuntimeError(
        "No LLM API key found. Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY"
    )
