"""Compose the `vllm serve` argv with the funding-extraction LoRA preloaded.

`run_serve(cfg)` execvp's vllm so signals propagate cleanly and the server PID
becomes the utility PID — no subprocess management.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ServeConfig:
    base_model: str
    lora_repo: str
    served_name: str
    host: str
    port: int
    dtype: str
    gpu_memory_utilization: float
    max_model_len: int
    tensor_parallel_size: int
    max_lora_rank: int
    max_loras: int
    passthrough: list[str] = field(default_factory=list)


def build_serve_argv(cfg: ServeConfig) -> list[str]:
    """Return the argv list to exec for `vllm serve`."""
    argv: list[str] = [
        "vllm",
        "serve",
        cfg.base_model,
        "--host", cfg.host,
        "--port", str(cfg.port),
        "--dtype", cfg.dtype,
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        "--max-model-len", str(cfg.max_model_len),
        "--tensor-parallel-size", str(cfg.tensor_parallel_size),
        "--enable-lora",
        "--max-lora-rank", str(cfg.max_lora_rank),
        "--max-loras", str(cfg.max_loras),
        "--lora-modules", f"{cfg.served_name}={cfg.lora_repo}",
        "--enable-prefix-caching",
    ]
    argv.extend(cfg.passthrough)
    return argv


def run_serve(cfg: ServeConfig) -> int:
    """Exec `vllm serve` with the composed argv. Does not return on success."""
    argv = build_serve_argv(cfg)
    logger.info("execvp: %s", " ".join(argv))
    os.execvp("vllm", argv)
    # execvp does not return on success; if we get here, exec failed
    return 1
