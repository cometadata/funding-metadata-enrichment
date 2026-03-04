import logging
from dataclasses import dataclass

import openai

from funding_extraction.config import VLLMConfig

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    content: str
    reasoning: str | None = None


class VLLMClient:
    """Thin wrapper around OpenAI client for talking to a vLLM server."""

    def __init__(self, config: VLLMConfig) -> None:
        self._client = openai.OpenAI(
            base_url=config.server.url,
            api_key=config.server.api_key or "dummy-key",
        )
        self._model = config.lora.name or config.model
        self._config = config

    def chat(
        self,
        messages: list[dict[str, str]],
        guided_json: dict | None = None,
    ) -> ChatResponse:
        """Send a chat completion request to the vLLM server.

        Args:
            messages: Chat messages (system, user, assistant).
            guided_json: Optional JSON schema for guided decoding.

        Returns:
            ChatResponse with content and optional reasoning trace.
        """
        extra_body: dict = {}
        if self._config.sampling.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
        if guided_json:
            extra_body["guided_json"] = guided_json

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._config.sampling.temperature,
            top_p=self._config.sampling.top_p,
            max_tokens=self._config.sampling.max_tokens,
            extra_body=extra_body or None,
        )

        message = response.choices[0].message
        reasoning = getattr(message, "reasoning_content", None)

        return ChatResponse(
            content=message.content,
            reasoning=reasoning,
        )
