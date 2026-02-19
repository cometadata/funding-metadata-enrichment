"""vLLM engine configuration loading."""

from dataclasses import dataclass, field, fields
from typing import Optional

import yaml


@dataclass
class VLLMLoRAConfig:
    path: Optional[str] = None
    name: Optional[str] = None
    max_rank: int = 64
    max_loras: int = 1


@dataclass
class VLLMEngineConfig:
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    quantization: Optional[str] = None
    enable_prefix_caching: bool = True


@dataclass
class VLLMSamplingConfig:
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class VLLMConfig:
    model: str
    lora: VLLMLoRAConfig = field(default_factory=VLLMLoRAConfig)
    engine: VLLMEngineConfig = field(default_factory=VLLMEngineConfig)
    sampling: VLLMSamplingConfig = field(default_factory=VLLMSamplingConfig)


def _filter_known_fields(cls, data: dict) -> dict:
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


def load_vllm_config(
    config_path: str,
    model_override: Optional[str] = None,
    lora_path_override: Optional[str] = None,
) -> VLLMConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    model = data.get("model")
    if not model and not model_override:
        raise ValueError("vLLM config must specify 'model'.")

    lora_data = _filter_known_fields(VLLMLoRAConfig, data.get("lora", {}))
    engine_data = _filter_known_fields(VLLMEngineConfig, data.get("engine", {}))
    sampling_data = _filter_known_fields(VLLMSamplingConfig, data.get("sampling", {}))

    config = VLLMConfig(
        model=model or "",
        lora=VLLMLoRAConfig(**lora_data),
        engine=VLLMEngineConfig(**engine_data),
        sampling=VLLMSamplingConfig(**sampling_data),
    )

    if model_override:
        config.model = model_override
    if lora_path_override:
        config.lora.path = lora_path_override

    return config
