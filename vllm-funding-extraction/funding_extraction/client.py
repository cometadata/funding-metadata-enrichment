import logging
from dataclasses import dataclass

import httpx
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
        timeout = httpx.Timeout(
            timeout=config.server.timeout or 300,
            connect=10.0,
        )
        self._client = openai.OpenAI(
            base_url=config.server.url,
            api_key=config.server.api_key or "dummy-key",
            timeout=timeout,
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
            if self._config.sampling.thinking_budget is not None:
                extra_body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self._config.sampling.thinking_budget,
                }
        if guided_json:
            extra_body["guided_json"] = guided_json

        sampling = self._config.sampling
        if sampling.top_k > 0:
            extra_body["top_k"] = sampling.top_k
        if sampling.min_p > 0:
            extra_body["min_p"] = sampling.min_p
        if sampling.repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = sampling.repetition_penalty

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            presence_penalty=sampling.presence_penalty,
            extra_body=extra_body or None,
        )

        message = response.choices[0].message
        reasoning = getattr(message, "reasoning_content", None)

        # The standard OpenAI client doesn't deserialize vLLM's
        # reasoning_content field. Fall back to the raw model_extra dicts.
        if reasoning is None:
            # Try on the message object first, then on the choice object
            msg_extra = getattr(message, "model_extra", None) or {}
            reasoning = msg_extra.get("reasoning_content")
        if reasoning is None:
            choice_extra = getattr(response.choices[0], "model_extra", None) or {}
            raw_msg = choice_extra.get("message", {})
            if isinstance(raw_msg, dict):
                reasoning = raw_msg.get("reasoning_content")

        return ChatResponse(
            content=message.content,
            reasoning=reasoning,
        )
