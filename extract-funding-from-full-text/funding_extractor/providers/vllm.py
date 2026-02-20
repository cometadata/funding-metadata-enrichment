import logging
import re
import threading
from collections.abc import Iterator, Sequence
from functools import wraps
from typing import Any, Dict, List, Optional

from langextract.core.base_model import BaseLanguageModel
from langextract.core.types import ScoredOutput

from funding_extractor.exceptions import ProviderConfigurationError
from funding_extractor.providers.base import BaseProvider, ModelProvider
from funding_extractor.providers.vllm_config import VLLMConfig, load_vllm_config

logger = logging.getLogger(__name__)

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_PATTERN = re.compile(r"<think>.*", re.DOTALL)


class VLLMLanguageModel(BaseLanguageModel):
    _engine = None
    _engine_lock = threading.Lock()

    def __init__(self, config: VLLMConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._temperature = config.sampling.temperature
        self._top_p = config.sampling.top_p
        self._top_k = config.sampling.top_k
        self._max_tokens = config.sampling.max_tokens
        self._enable_thinking = config.sampling.enable_thinking
        self._lora_request = self._build_lora_request(config)
        self._reasoning_traces: list[str] = []
        self._reasoning_lock = threading.Lock()

    @staticmethod
    def _build_lora_request(config: VLLMConfig) -> Optional[Any]:
        if not config.lora.path:
            return None
        from vllm.lora.request import LoRARequest

        return LoRARequest(
            lora_name=config.lora.name or "default",
            lora_int_id=1,
            lora_path=config.lora.path,
        )

    @classmethod
    def _get_or_create_engine(cls, config: VLLMConfig) -> Any:
        with cls._engine_lock:
            if cls._engine is None:
                from vllm import LLM

                engine_kwargs: dict[str, Any] = {
                    "model": config.model,
                    "tensor_parallel_size": config.engine.tensor_parallel_size,
                    "max_model_len": config.engine.max_model_len,
                    "gpu_memory_utilization": config.engine.gpu_memory_utilization,
                    "dtype": config.engine.dtype,
                    "enable_prefix_caching": config.engine.enable_prefix_caching,
                }
                if config.engine.quantization:
                    engine_kwargs["quantization"] = config.engine.quantization
                if config.lora.path:
                    engine_kwargs["enable_lora"] = True
                    engine_kwargs["max_lora_rank"] = config.lora.max_rank
                    engine_kwargs["max_loras"] = config.lora.max_loras

                logger.info("Loading vLLM engine: %s", config.model)
                cls._engine = LLM(**engine_kwargs)
            return cls._engine

    def _strip_thinking(self, text: str) -> str:
        """Strip <think>...</think> tags from text, storing reasoning traces.

        Handles both closed tags and unclosed tags (output truncated mid-think).
        """
        traces: list[str] = []

        def _capture(match: re.Match) -> str:
            content = match.group(0)
            # Strip the tags to get just the reasoning content
            inner = content.removeprefix("<think>").removesuffix("</think>").strip()
            if inner:
                traces.append(inner)
            return ""

        cleaned = _THINK_PATTERN.sub(_capture, text)
        # Handle unclosed <think> tag (truncated output)
        if "<think>" in cleaned:
            unclosed = _UNCLOSED_THINK_PATTERN.search(cleaned)
            if unclosed:
                inner = unclosed.group(0).removeprefix("<think>").strip()
                if inner:
                    traces.append(inner)
                cleaned = cleaned[:unclosed.start()]

        if traces:
            with self._reasoning_lock:
                self._reasoning_traces.extend(traces)

        return cleaned.strip()

    def infer(
        self, batch_prompts: Sequence[str], **kwargs: Any
    ) -> Iterator[Sequence[ScoredOutput]]:
        from vllm import SamplingParams

        merged = self.merge_kwargs(kwargs)
        sampling_params = SamplingParams(
            temperature=merged.get("temperature", self._temperature),
            top_p=merged.get("top_p", self._top_p),
            top_k=merged.get("top_k", self._top_k),
            max_tokens=merged.get("max_output_tokens", self._max_tokens),
        )

        engine = self._get_or_create_engine(self._config)
        outputs = engine.generate(
            list(batch_prompts),
            sampling_params,
            lora_request=self._lora_request,
        )

        for output in outputs:
            raw_text = output.outputs[0].text
            if self._enable_thinking:
                raw_text = self._strip_thinking(raw_text)
            yield [ScoredOutput(score=1.0, output=raw_text)]


def _patch_client_for_thinking(language_model: Any, enable_thinking: bool) -> None:
    """Wrap the OpenAI client's create method to inject chat_template_kwargs
    for Qwen3 thinking control and capture reasoning content from responses."""
    original_create = language_model._client.chat.completions.create

    @wraps(original_create)
    def patched_create(**kwargs):
        extra_body = kwargs.pop("extra_body", {}) or {}
        extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        kwargs["extra_body"] = extra_body
        response = original_create(**kwargs)
        if enable_thinking and response.choices:
            reasoning = getattr(response.choices[0].message, "reasoning", None)
            if reasoning is None:
                reasoning = getattr(
                    response.choices[0].message, "reasoning_content", None
                )
            if reasoning:
                with language_model._reasoning_lock:
                    language_model._reasoning_traces.append(reasoning)
        return response

    language_model._client.chat.completions.create = patched_create


class VLLMProvider(BaseProvider):
    def __init__(
        self,
        model_id: Optional[str] = None,
        model_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
        debug: bool = False,
        reasoning_effort: Optional[str] = None,
        vllm_config_path: Optional[str] = None,
        lora_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            model_url=model_url,
            api_key=api_key,
            timeout=timeout,
            debug=debug,
            reasoning_effort=reasoning_effort,
        )
        if not vllm_config_path:
            raise ProviderConfigurationError(
                "--vllm-config is required when using --provider vllm."
            )
        mode_override = "online" if model_url else None
        server_url_override = model_url if model_url else None
        self._vllm_config = load_vllm_config(
            vllm_config_path,
            model_override=model_id,
            lora_path_override=lora_path,
            mode_override=mode_override,
            server_url_override=server_url_override,
        )
        if self._vllm_config.mode == "online":
            self._language_model = self._build_online_model()
        else:
            self._language_model = VLLMLanguageModel(self._vllm_config)

    def _build_online_model(self):
        from langextract.providers.openai import OpenAILanguageModel

        model_id = self._vllm_config.lora.name or self._vllm_config.model
        api_key = self._vllm_config.server.api_key or self.api_key or "dummy-key"

        logger.info("vLLM online mode: server=%s model=%s", self._vllm_config.server.url, model_id)

        lm = OpenAILanguageModel(
            model_id=model_id,
            api_key=api_key,
            base_url=self._vllm_config.server.url,
            temperature=self._vllm_config.sampling.temperature,
            top_p=self._vllm_config.sampling.top_p,
            top_k=self._vllm_config.sampling.top_k,
            max_output_tokens=self._vllm_config.sampling.max_tokens,
            max_workers=1,
        )

        # Always patch to explicitly control Qwen3's thinking mode
        # (Qwen3 defaults to thinking-on, so we must set it even when disabling)
        lm._reasoning_traces = []
        lm._reasoning_lock = threading.Lock()
        _patch_client_for_thinking(lm, self._vllm_config.sampling.enable_thinking)

        return lm

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.VLLM

    def build_extract_params(self, statement: str, prompt: str, examples: List[Any]) -> Dict[str, Any]:
        is_online = self._vllm_config.mode == "online"
        extraction_passes = 3
        return {
            "text_or_documents": statement,
            "prompt_description": prompt,
            "examples": examples,
            "temperature": self._vllm_config.sampling.temperature,
            "extraction_passes": extraction_passes,
            "max_workers": extraction_passes if is_online else 1,
            "debug": self.debug,
            "fence_output": True,
            "use_schema_constraints": False,
            "model": self._language_model,
            "resolver_params": {"suppress_parse_errors": True},
        }

    def drain_reasoning(self) -> list[str]:
        """Return and clear accumulated reasoning traces from the language model."""
        lm = self._language_model
        with lm._reasoning_lock:
            traces = list(lm._reasoning_traces)
            lm._reasoning_traces.clear()
        return traces
