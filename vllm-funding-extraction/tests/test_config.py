import tempfile
from pathlib import Path

import yaml

from funding_extraction.config import (
    VLLMBenchmarkConfig,
    VLLMConfig,
    VLLMEngineConfig,
    VLLMLoRAConfig,
    VLLMSamplingConfig,
    VLLMServerConfig,
    load_vllm_config,
)


def _write_yaml(data: dict) -> str:
    """Write a dict to a temp YAML file and return the path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, tmp, default_flow_style=False)
    tmp.flush()
    tmp.close()
    return tmp.name


def test_defaults():
    config = VLLMConfig(model="test-model")
    assert config.model == "test-model"
    assert config.mode == "offline"
    assert config.sampling.temperature == 0.7
    assert config.sampling.guided_decoding is False
    assert config.sampling.extraction_passes == 3
    assert config.server.url is None
    assert config.benchmark.workers == 64


def test_load_minimal_config():
    path = _write_yaml({"model": "Qwen/Qwen3.5-9B"})
    config = load_vllm_config(path)
    assert config.model == "Qwen/Qwen3.5-9B"
    assert config.mode == "offline"


def test_load_full_config():
    data = {
        "model": "Qwen/Qwen3.5-9B",
        "mode": "online",
        "engine": {"max_model_len": 81920, "gpu_memory_utilization": 0.95},
        "sampling": {
            "temperature": 0.6,
            "max_tokens": 65536,
            "enable_thinking": True,
            "extraction_passes": 1,
            "guided_decoding": True,
        },
        "server": {"url": "http://localhost:8000/v1", "timeout": 600, "reasoning_parser": "qwen3"},
        "benchmark": {"config_name": "qwen3.5-9b-entities-thinking", "workers": 64},
    }
    path = _write_yaml(data)
    config = load_vllm_config(path)
    assert config.model == "Qwen/Qwen3.5-9B"
    assert config.mode == "online"
    assert config.engine.max_model_len == 81920
    assert config.sampling.temperature == 0.6
    assert config.sampling.enable_thinking is True
    assert config.sampling.guided_decoding is True
    assert config.sampling.extraction_passes == 1
    assert config.server.url == "http://localhost:8000/v1"
    assert config.server.reasoning_parser == "qwen3"
    assert config.benchmark.config_name == "qwen3.5-9b-entities-thinking"


def test_model_override():
    path = _write_yaml({"model": "original-model"})
    config = load_vllm_config(path, model_override="override-model")
    assert config.model == "override-model"


def test_lora_path_override():
    path = _write_yaml({"model": "test-model"})
    config = load_vllm_config(path, lora_path_override="/path/to/lora")
    assert config.lora.path == "/path/to/lora"


def test_mode_override():
    path = _write_yaml({"model": "test-model", "server": {"url": "http://localhost:8000/v1"}})
    config = load_vllm_config(path, mode_override="online")
    assert config.mode == "online"


def test_server_url_override():
    path = _write_yaml({"model": "test-model", "mode": "online", "server": {"url": "http://localhost:8000/v1"}})
    config = load_vllm_config(path, server_url_override="http://other:9000/v1")
    assert config.server.url == "http://other:9000/v1"


def test_missing_model_raises():
    path = _write_yaml({})
    try:
        load_vllm_config(path)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "model" in str(e).lower()


def test_invalid_mode_raises():
    path = _write_yaml({"model": "test-model", "mode": "invalid"})
    try:
        load_vllm_config(path)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid" in str(e).lower()


def test_online_mode_requires_server_url():
    path = _write_yaml({"model": "test-model", "mode": "online"})
    try:
        load_vllm_config(path)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "server.url" in str(e)


def test_unknown_fields_ignored():
    """Extra fields in YAML should not cause errors."""
    data = {
        "model": "test-model",
        "sampling": {"temperature": 0.5, "unknown_field": 999},
    }
    path = _write_yaml(data)
    config = load_vllm_config(path)
    assert config.sampling.temperature == 0.5
