"""LangChain Groq chat client. The model name comes from settings."""

from __future__ import annotations

from typing import Protocol

from langchain_core.messages import BaseMessage

from app.generation.errors import GenerationConfigError
from app.generation.settings import GenerationSettings


class GenerationClient(Protocol):
    """A chat client that accepts prompt messages and returns a model result."""

    def invoke(self, messages: list[BaseMessage]) -> object:
        """Return the model result for these messages."""


class GroqGenerationClient:
    """Call the configured Groq chat model through LangChain."""

    def __init__(self, settings: GenerationSettings) -> None:
        self.settings = settings
        self._chat: object | None = None

    def invoke(self, messages: list[BaseMessage]) -> object:
        """Send the prompt to the configured model."""
        if self._chat is None:
            self._chat = build_chat_model(self.settings)
        invoke = getattr(self._chat, "invoke")
        return invoke(messages)


def build_chat_model(settings: GenerationSettings) -> object:
    """Construct ChatGroq. An empty key or model name is a configuration error."""
    key = settings.configured_key()
    model = settings.configured_model()
    if not key or _is_placeholder(key):
        raise GenerationConfigError("GROQ_API_KEY is not configured.")
    if not model or _is_placeholder(model):
        raise GenerationConfigError("GROQ_MODEL is not configured.")
    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=key,
        model=model,
        temperature=settings.groq_temperature,
        timeout=120,
        # Keep chain-of-thought out of message.content so JSON parsing sees the answer.
        reasoning_format="parsed",
    )


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.startswith("your_") or "placeholder" in lowered
