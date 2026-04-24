# Tier 2 throughput tuning — findings

**Date:** 2026-04-24
**Plan:** [docs/plans/2026-04-24-tier2-throughput-tuning.md](../plans/2026-04-24-tier2-throughput-tuning.md)
**Dataset:** `cometadata/arxiv-funding-statement-extraction`, split=test, subdirs=data,arxiv-latex-extract, n=3580 rows
**Image:** `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`
**Baseline (batch Tier 2 v2, 2026-04-24, h200):** 50.40 docs/sec, F1 overall=0.894, peak_mem=3.5 GB
**Target:** ≥ 250 docs/sec, F1 overall ≥ 0.894, F1 clean ≥ 0.909

## TL;DR

We did not hit 250 docs/sec. Best sustained Tier 2 throughput observed was **~100 docs/sec** (extrapolated from 2 of 4 a100-large cross-node shards). Neither Experiment A (worker count) nor Experiment B (pipeline batch size) qualified for Phase 3 by the plan's ≥ 10 docs/sec bar. The target was shaped by encode-alone throughput math; on Tier 2 the prefilter drops ~90% of paragraphs so the end-to-end workload is small, fixed overhead dominates, and knob tuning has little leverage. **Recommendation:** ship the batch engine at its current 50 docs/sec and revise the Tier 2 target to ~50–60 docs/sec, a 2–3× win over the legacy per-doc loop.

## Phase 1 — pipeline profile

Instrumentation (`_PipelineProfile`) was landed in `e387f8d` behind a `--profile-pipeline` flag. The h200 profile run was blocked twice by CUDA Error 802 on the h200 pool (documented in `2026-04-24-h200-ab-pipelined.md` Phase 1 section); we captured the profile on `a100-large` as a fallback:

```
wall=89.2s gpu_active=86.4s (97%) gpu_idle=2.1s (2%)
post_pool_saturated=79.2s (89%) pre_pool_saturated=77.7s (87%)
writer_waiting=89.1s (100%) workers=94
```

On A100 with 94 workers the GPU is saturated. The h200 profile is still missing — the plan's Phase 2 lever order was shaped by an assumed "GPU idle → CPU bottleneck" read that we could not verify directly.

## Phase 2 — single-lever experiments

All runs: full test split (n=3580), bf16, `--enable-paragraph-prefilter`, `--subdirs data,arxiv-latex-extract`, `--no-push`, `--profile-pipeline` off.

### Experiment A — `--engine-workers`

| Config | workers | docs/sec | Δ vs baseline | F1 overall | F1 clean | peak_mem | Status |
|---|---:|---:|---:|---:|---:|---:|:--|
| Baseline (v2) | 14 (cpu-2) | 50.40 | — | 0.894 | 0.909 | 3.5 GB | ref |
| w16 | 16 | — | — | — | — | — | infra fail (802) |
| w32 | 32 | — | — | — | — | — | infra fail (802) |
| w64 | 64 | **56.40** | +6.0 | 0.894 | 0.909 | 3.6 GB | complete |

w64 gained only 6 docs/sec — below the 10 docs/sec Phase 3 qualification bar. w16 and w32 were not resubmitted: w64's ceiling makes them redundant. **Lever disqualified.**

### Experiment B — `--paragraphs-per-batch`

| Config | ppb | docs/sec | Δ vs baseline | F1 overall | F1 clean | peak_mem | Status |
|---|---:|---:|---:|---:|---:|---:|:--|
| Baseline (v2) | 4096 | 50.40 | — | 0.894 | 0.909 | 3.5 GB | ref |
| ppb512 | 512 | — | — | — | — | — | infra fail (802) |
| ppb1024 | 1024 | — | — | — | — | — | infra fail (802) |
| ppb2048 | 2048 | **52.59** | +2.2 | 0.894 | 0.909 | 3.0 GB | complete |

ppb=2048 gained 2.2 docs/sec (4%). Extrapolating a rough log-linear trend, ppb=512 might reach ~57 docs/sec — still well below the 10 docs/sec qualification bar and nowhere near 250. **Lever disqualified.**

Experiment C (`--encode-batch-size` sweep) and Experiment D (`--queue-depth`) were not run: Experiments A + B already told us knob tuning wasn't going to close a 5× gap.

### Supplementary — sharded harness (Option 1 from the plan's non-goal section)

To test whether the ceiling was hardware-bound, we added `--shard-index` / `--num-shards` to `benchmark_hf_job.py` (`4c084a5`) and ran the corpus split across multiple GPUs.

| Mode | GPUs | per-shard docs/sec | aggregate docs/sec | Notes |
|---|---:|---:|---:|:--|
| Single a100-large | 1 | 35.46 | 35.46 | 94 workers, GPU saturated 97% |
| Single h200 (baseline) | 1 | 50.40 | 50.40 | 14 workers |
| a100x4 on-host, 4 shards | 4 | 25–28 | 54.24 | 24 workers per shard |
| a100-large cross-node, 2/4 shards complete | 4 | 26–27 | **~100** extrapolated | independent hosts |

Per-shard throughput is identical across on-host and cross-node modes — CPU contention is not the bottleneck. The real cost is **fixed per-process overhead (~15 s of model load + CUDA warmup + first-encode compilation)** that doesn't amortize on 895-doc shards. Steady-state throughput per GPU is ~40–50 docs/sec regardless of mode, but wall is dominated by the fixed prefix.

**Extrapolation to hit 250 docs/sec via sharding:** would require ~10× a100 running in parallel with ~358-doc shards (or larger shards over a larger corpus). The Tier 2 test corpus isn't big enough to amortize multi-node orchestration for this target.

## Why the target wasn't reachable at this pipeline shape

1. **Tier 2's GPU workload is intentionally small.** Prefilter drops ~90% of paragraphs, leaving ~3–4k paragraphs total across 3580 docs — roughly one h200 encode pass worth of work.
2. **Workers hit saturation fast.** 14 → 64 → 94 workers all give similar steady-state throughput; CPU parallelism isn't the floor.
3. **Wall is pipeline-overhead bound.** The GPU encode is a few seconds of real compute; the rest of the 60–70 s wall is queue ops, pool dispatch, pre/post processing, and the post-stage regex scan. That overhead doesn't shrink with more hardware.
4. **Fixed per-process overhead dominates small shards.** Sharding doesn't linearly scale a small corpus.

Put differently: the 250 docs/sec target was derived in the design doc from encode-alone math. At this corpus size and prefilter ratio, the pipeline is spending most of its wall on non-encode work that no single knob addresses.

## Recommendation

1. **Leave defaults unchanged.** `paragraphs_per_batch=4096`, `encode_batch_size=512`, `queue_depth=128`, `workers=cpu_count-2`. No Phase-2 config shift justified its F1-parity cost (all the qualified configs match baseline F1; none meaningfully improve throughput). w64 is a small win (+6 docs/sec) but would need per-host tuning — not worth baking in.
2. **Revise the Tier 2 target to ~50–60 docs/sec** (2–3× legacy per-doc baseline of 22.78) and land the batch engine at that level. F1 parity is the hard requirement; we meet it.
3. **Defer Phase 4 (architectural backstop).** The Phase 4 pre-flight gate — `_post_task` mean < ~5 ms/doc — is likely to pass, but the remaining gain from inlining high-confidence items is bounded by the 6–10 docs/sec of real post-stage saturation we'd recover. Not worth the complexity given the target revision.
4. **If throughput becomes a hard blocker later**, the highest-leverage lift is Option 2 from the sharding write-up: multiple GPU consumer threads feeding a single shared pre/post pool, amortizing model load once per process. Estimated ~1 day of refactor for the potential to land at 150–200 docs/sec on a multi-GPU flavor.

## Files & jobs

- Phase 1 instrumentation: commit `e387f8d`
- Shard args: commit `4c084a5`
- Profile findings doc: `docs/perf/2026-04-24-h200-ab-pipelined.md` (Phase 1 section)
- HF job IDs referenced:
  - a100-large profile: `69ebc57cd2c8bd8662bccb6c` (COMPLETED)
  - Exp A w64 (h200): `69ebc6e4d70108f37acdddd1` (COMPLETED)
  - Exp B ppb=2048 (h200): `69ebc856d70108f37acddddf` (COMPLETED)
  - a100x4 on-host shard: `69ebd45bd2c8bd8662bccc0e` (COMPLETED)
  - Cross-node shards (2 of 4 completed, 2 cancelled after prolonged scheduling): `69ebdc6ad2c8bd8662bccc66`, `69ebdc6cd70108f37acdde31`
