from funding_entity_extractor.serve import build_serve_argv, ServeConfig


def test_build_serve_argv_minimum_defaults():
    cfg = ServeConfig(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        lora_repo="cometadata/funding-extraction-llama-3.1-8b-instruct-artifact-data-mix-grpo-mixed-reward",
        served_name="funding-extraction",
        host="0.0.0.0",
        port=8000,
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        max_model_len=4096,
        tensor_parallel_size=1,
        max_lora_rank=64,
        max_loras=1,
    )
    argv = build_serve_argv(cfg)

    assert argv[0] == "vllm"
    assert argv[1] == "serve"
    assert argv[2] == "meta-llama/Llama-3.1-8B-Instruct"
    # The order beyond positional must be deterministic for log readability
    assert "--enable-lora" in argv
    assert "--enable-prefix-caching" in argv
    assert "--lora-modules" in argv
    lora_module_idx = argv.index("--lora-modules")
    assert argv[lora_module_idx + 1] == (
        "funding-extraction=cometadata/funding-extraction-llama-3.1-8b-instruct-artifact-data-mix-grpo-mixed-reward"
    )
    # numerics surface as their string representation
    assert "--port" in argv and argv[argv.index("--port") + 1] == "8000"
    assert "--gpu-memory-utilization" in argv
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.92"
    assert "--max-lora-rank" in argv
    assert argv[argv.index("--max-lora-rank") + 1] == "64"


def test_build_serve_argv_appends_passthrough_args():
    cfg = ServeConfig(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        lora_repo="cometadata/funding-extraction-llama-3.1-8b-instruct-artifact-data-mix-grpo-mixed-reward",
        served_name="funding-extraction",
        host="0.0.0.0",
        port=8000,
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        max_model_len=4096,
        tensor_parallel_size=1,
        max_lora_rank=64,
        max_loras=1,
        passthrough=["--quantization", "fp8", "--enforce-eager"],
    )
    argv = build_serve_argv(cfg)
    assert argv[-3:] == ["--quantization", "fp8", "--enforce-eager"]
