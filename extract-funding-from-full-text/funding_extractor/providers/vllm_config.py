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
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    quantization: Optional[str] = None
    enable_prefix_caching: bool = True


@dataclass
class VLLMSamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 4096
    enable_thinking: bool = False
    thinking_budget: Optional[int] = None
    presence_penalty: float = 0.0


@dataclass
class VLLMServerConfig:
    url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 120


@dataclass
class VLLMConfig:
    model: str
    mode: str = "offline"
    lora: VLLMLoRAConfig = field(default_factory=VLLMLoRAConfig)
    engine: VLLMEngineConfig = field(default_factory=VLLMEngineConfig)
    sampling: VLLMSamplingConfig = field(default_factory=VLLMSamplingConfig)
    server: VLLMServerConfig = field(default_factory=VLLMServerConfig)


def _filter_known_fields(cls, data: dict) -> dict:
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


def load_vllm_config(
    config_path: str,
    model_override: Optional[str] = None,
    lora_path_override: Optional[str] = None,
    mode_override: Optional[str] = None,
    server_url_override: Optional[str] = None,
) -> VLLMConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    model = data.get("model")
    if not model and not model_override:
        raise ValueError("vLLM config must specify 'model'.")

    mode = data.get("mode", "offline")
    lora_data = _filter_known_fields(VLLMLoRAConfig, data.get("lora", {}))
    engine_data = _filter_known_fields(VLLMEngineConfig, data.get("engine", {}))
    sampling_data = _filter_known_fields(VLLMSamplingConfig, data.get("sampling", {}))
    server_data = _filter_known_fields(VLLMServerConfig, data.get("server", {}))

    config = VLLMConfig(
        model=model or "",
        mode=mode,
        lora=VLLMLoRAConfig(**lora_data),
        engine=VLLMEngineConfig(**engine_data),
        sampling=VLLMSamplingConfig(**sampling_data),
        server=VLLMServerConfig(**server_data),
    )

    if model_override:
        config.model = model_override
    if lora_path_override:
        config.lora.path = lora_path_override
    if mode_override:
        config.mode = mode_override
    if server_url_override:
        config.server.url = server_url_override

    if config.mode not in ("offline", "online"):
        raise ValueError(
            f"Invalid mode '{config.mode}'. Must be 'offline' or 'online'."
        )
    if config.mode == "online" and not config.server.url:
        raise ValueError("server.url is required when mode is 'online'.")

    return config
