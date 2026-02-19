"""vLLM direct inference provider."""

import logging
import threading
from collections.abc import Iterator, Sequence
from typing import Any, Optional

from langextract.core.base_model import BaseLanguageModel
from langextract.core.types import ScoredOutput

from funding_extractor.providers.vllm_config import VLLMConfig

logger = logging.getLogger(__name__)


class VLLMLanguageModel(BaseLanguageModel):
    """Langextract-compatible wrapper around a vLLM engine."""

    _engine = None
    _engine_lock = threading.Lock()

    def __init__(self, config: VLLMConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._temperature = config.sampling.temperature
        self._max_tokens = config.sampling.max_tokens
        self._lora_request = self._build_lora_request(config)

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

    def infer(
        self, batch_prompts: Sequence[str], **kwargs: Any
    ) -> Iterator[Sequence[ScoredOutput]]:
        from vllm import SamplingParams

        merged = self.merge_kwargs(kwargs)
        sampling_params = SamplingParams(
            temperature=merged.get("temperature", self._temperature),
            max_tokens=merged.get("max_output_tokens", self._max_tokens),
        )

        engine = self._get_or_create_engine(self._config)
        outputs = engine.generate(
            list(batch_prompts),
            sampling_params,
            lora_request=self._lora_request,
        )

        for output in outputs:
            yield [ScoredOutput(score=1.0, output=output.outputs[0].text)]
