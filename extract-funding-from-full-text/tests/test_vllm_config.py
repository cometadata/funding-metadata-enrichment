import pytest
import yaml

from funding_extractor.providers.vllm_config import (
    VLLMConfig,
    VLLMEngineConfig,
    VLLMLoRAConfig,
    VLLMSamplingConfig,
    load_vllm_config,
)


def _write_config(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return str(path)


def test_load_full_config(tmp_path):
    data = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "lora": {"path": "/tmp/lora", "name": "my-lora", "max_rank": 32, "max_loras": 2},
        "engine": {
            "tensor_parallel_size": 2,
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.85,
            "dtype": "float16",
            "quantization": "awq",
            "enable_prefix_caching": False,
        },
        "sampling": {"temperature": 0.2, "max_tokens": 4096},
    }
    config = load_vllm_config(_write_config(tmp_path, data))
    assert config.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert config.lora.path == "/tmp/lora"
    assert config.lora.name == "my-lora"
    assert config.lora.max_rank == 32
    assert config.lora.max_loras == 2
    assert config.engine.tensor_parallel_size == 2
    assert config.engine.max_model_len == 8192
    assert config.engine.gpu_memory_utilization == 0.85
    assert config.engine.dtype == "float16"
    assert config.engine.quantization == "awq"
    assert config.engine.enable_prefix_caching is False
    assert config.sampling.temperature == 0.2
    assert config.sampling.max_tokens == 4096


def test_load_minimal_config(tmp_path):
    data = {"model": "some-model"}
    config = load_vllm_config(_write_config(tmp_path, data))
    assert config.model == "some-model"
    assert config.lora.path is None
    assert config.engine.tensor_parallel_size == 1
    assert config.engine.max_model_len == 4096
    assert config.engine.gpu_memory_utilization == 0.9
    assert config.engine.dtype == "auto"
    assert config.engine.quantization is None
    assert config.engine.enable_prefix_caching is True
    assert config.sampling.temperature == 0.1
    assert config.sampling.max_tokens == 2048


def test_load_config_model_override(tmp_path):
    data = {"model": "original-model"}
    config = load_vllm_config(
        _write_config(tmp_path, data), model_override="override-model"
    )
    assert config.model == "override-model"


def test_load_config_lora_path_override(tmp_path):
    data = {"model": "some-model", "lora": {"path": "/original"}}
    config = load_vllm_config(
        _write_config(tmp_path, data), lora_path_override="/override"
    )
    assert config.lora.path == "/override"


def test_load_config_missing_model(tmp_path):
    data = {"engine": {"tensor_parallel_size": 1}}
    with pytest.raises(ValueError, match="model"):
        load_vllm_config(_write_config(tmp_path, data))


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_vllm_config("/nonexistent/config.yaml")
