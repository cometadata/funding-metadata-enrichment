# Qwen3.5-9B Benchmark Report

## Summary

Qwen3.5-9B entity extraction produced **~49% null predictions** across both
train and test splits, severely degrading recall. The root cause is a
combination of vLLM/Qwen3.5 interoperability issues and suboptimal sampling
parameters for structured output.

## Benchmark Results (test split)

| Level          | Gold | Pred |     P |     R |    F1 |
|----------------|-----:|-----:|------:|------:|------:|
| Funder         |  810 |  584 | 0.844 | 0.609 | 0.707 |
| Award ID       |  809 |  524 | 0.760 | 0.492 | 0.597 |
| Funding Scheme |  358 |   95 | 0.642 | 0.170 | 0.269 |
| Award Title    |   86 |   18 | 0.611 | 0.128 | 0.212 |

Precision is reasonable (the model extracts correctly when it produces output),
but recall is roughly halved because ~49% of documents returned empty funders.

## Comparison to Other Models (test split, Funder F1)

| Model                        | Funder F1 | Empty % |
|------------------------------|----------:|--------:|
| Qwen3-8B non-thinking       |     0.847 |    0.9% |
| LLaMA-3.1-8B                |     0.851 |    0.3% |
| LLaMA-3.1-8B LoRA           |     0.848 |    1.6% |
| Qwen3-8B LoRA thinking      |     0.818 |   15.2% |
| Qwen3-8B thinking           |     0.782 |   19.3% |
| **Qwen3.5-9B non-thinking** | **0.707** | **49.1%** |

## Root Causes

### 1. langextract resolver rejections (586 of 1638 statements)

The dominant error in the job logs:

```
ERROR Extraction text must be a string, integer, or float. Found: <class 'NoneType'>
ERROR Extraction text must be a string, integer, or float. Found: <class 'list'>
```

Breakdown: 497 NoneType errors, 138 list errors. The model generated extraction
output where text fields were null or arrays instead of strings. When the
langextract resolver rejects the output, the handler returns an empty result.

### 2. Request timeouts (197 of 1638 statements)

197 requests exceeded the 120-second timeout. Many clustered at the start of
processing (thundering herd: 64 workers hitting the server simultaneously at
startup). Later timeouts were caused by long-running extractions on complex
statements.

### 3. Interacting vLLM/Qwen3.5 issues

**Missing `--reasoning-parser qwen3`**: The vLLM server was launched without a
reasoning parser because `enable_thinking` was false. However, Qwen3.5-9B has
thinking enabled by default in its chat template. When `enable_thinking: false`
is sent per-request via `chat_template_kwargs`, the template emits an empty
`<think>\n\n</think>\n\n` block. Without the reasoning parser to strip these
tokens, they leak into the response and corrupt JSON parsing.

**Open vLLM bug ([#35574](https://github.com/vllm-project/vllm/issues/35574),
filed 2026-02-28)**: Per-request `chat_template_kwargs: {"enable_thinking":
false}` does not reliably disable thinking for Qwen3.5 on current vLLM nightly
builds. This means thinking tokens may still be generated even when explicitly
disabled.

**`presence_penalty: 1.5` harmful for structured output**: While this is the
official Qwen recommendation for general non-thinking tasks, the model card
warns it causes "language mixing and slight performance decrease." For structured
JSON extraction that requires repeated structural tokens (`"`, `{`, `}`), this
penalty actively degrades output quality. The "precise coding" recommendation
uses `presence_penalty: 0.0`.

## Fixes Applied

1. Added `reasoning_parser: "qwen3"` to `qwen3.5-9b.yaml` server config
2. Changed `presence_penalty` from 1.5 to 0.0 in both `qwen3.5-9b.yaml` and
   `qwen3.5-9b-thinking.yaml`
3. Updated `run_benchmark_entities.py` to always pass `--reasoning-parser` when
   configured in YAML (not just when thinking is enabled)
4. Added `--default-chat-template-kwargs '{"enable_thinking": false}'` at the
   server level when thinking is disabled, as a workaround for the open vLLM bug

## Outstanding Risks

- vLLM [#35574](https://github.com/vllm-project/vllm/issues/35574) remains
  open; `enable_thinking: false` may still not fully suppress thinking tokens
  for Qwen3.5 until a fix lands in a stable vLLM release (>= 0.17.0)
- Qwen3.5 support requires vLLM nightly; behavior may vary between builds
- The 120-second timeout may still be too short for complex statements under
  load with 64 concurrent workers

## Job Reference

- HuggingFace Job: `adambuttrick/69a5c2fcdfb316ac3f7c1c49`
- Dataset: `cometadata/funding-extraction-harness-benchmark` config `qwen3.5-9b-entities`
- Commit with fixes: `164ab22`
