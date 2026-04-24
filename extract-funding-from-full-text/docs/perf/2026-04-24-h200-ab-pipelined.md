# H200 A/B: pipelined batch engine vs legacy per-doc loop

**Date:** 2026-04-24
**Dataset:** `cometadata/arxiv-funding-statement-extraction`, split=test, subdirs=data,arxiv-latex-extract, n=3580 rows
**Image:** `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`, flavor `h200`
**Script:** `scripts/benchmark_hf_job.py` at SHA `cb3d81b` (plus earlier `5b8a2e2` for v1 submissions)
**Package:** pulled from `git+…#subdirectory=extract-funding-from-full-text@statement-only-extraction`

## Results

| Config | F1 overall | F1 clean | F1 clean_relocated | docs/sec | peak_mem | Gate |
|---|---:|---:|---:|---:|---:|:---:|
| **Legacy Tier 2 baseline** | **0.894** | **0.909** | **0.907** | **22.78** | **1.9 GB** | ✅ |
| **Batch Tier 2 (default)** | **0.894** | **0.909** | **0.907** | **50.40** | **3.5 GB** | ⚠️ |
| Legacy Tier 1 baseline | — | — | — | — | — | ❌ (infra) |
| Batch Tier 1 | — | — | — | — | — | ❌ (infra) |

Per-bucket F1 for the two completed runs was **byte-identical**: overall 0.894, clean 0.909, clean_relocated 0.907, mild 0.886, heavy 0.875. This confirms the batch engine preserves extraction quality exactly.

## Gate evaluation

| Run | F1 overall | F1 clean+arxiv | docs/sec | peak_mem |
|---|---|---|---|---|
| legacy Tier 1 (baseline ≥0.892 / ≥0.905 / ~4.91 / ~2 GB) | — | — | — | — |
| legacy Tier 2 (baseline ≥0.894 / ≥0.908 / ~22.66 / ~2 GB) | ✅ 0.894 | ✅ 0.908 avg | ✅ 22.78 | ✅ 1.9 GB |
| batch Tier 1 (≥0.892 / ≥0.905 / ≥30 / ≥30 GB) | — | — | — | — |
| batch Tier 2 (≥0.894 / ≥0.908 / ≥250 / ≥30 GB) | ✅ 0.894 | ✅ 0.908 avg | ❌ 50.40 | ❌ 3.5 GB |

**F1 gates: PASS** — batch engine equals legacy baseline on every bucket.
**Throughput gate (Tier 2): FAIL** — 50.4 docs/sec vs ≥250 target. 2.2× the legacy baseline, not 11×.
**Peak memory gate (Tier 2): FAIL** — 3.5 GB vs ≥30 GB target. GPU is nowhere near saturated.
**Tier 1 gates: UNKNOWN** — 4 job attempts blocked by infrastructure.

## Throughput analysis

The Batch Tier 2 run shows wall=71.0s for 3580 docs = 50.40 docs/sec, peak_mem=3.5 GB. Per-doc queueing-aware latency was p50=31s, p95=52s — items sit in queues for most of the run and flush at the end. Combined with the low peak_mem, this points at GPU under-utilization:

- `paragraphs_per_batch=4096` was picked to balance latency vs throughput. With 3580 docs and Tier 2's ~3-10 surviving paragraphs per doc after prefilter, that batches essentially the whole corpus into a single GPU pass. But the per-pass GPU work is small relative to the rest of the pipeline — CPU post-stage (regex gating + statement extraction) is the bottleneck, not encode.
- Target 250 docs/sec was derived in the design doc from encode-alone throughput. The pipeline has cross-stage overhead (queue waits, pool dispatch) that the design didn't budget for.
- 30 GB peak_mem was based on filling H200 with larger batches. At `paragraphs_per_batch=4096` we batch ~3.5k paragraphs total which is far from capacity.

**Follow-up tuning levers (not in scope of this PR):** shrink `paragraphs_per_batch` to balance GPU work per pass (e.g. 512, 1024, 2048 sweeps), increase `engine_workers` to cover more post-stage CPU, or move the post-stage regex into the GPU pass to cut the CPU bottleneck.

## Infrastructure reliability — Tier 1 never completed

All 4 Tier 1 submissions (2 legacy, 2 batch) aborted on `cudaGetDeviceCount()` Error 802 ("system not yet initialized"). The aborts happened during the CUDA retry probe in `scripts/benchmark_hf_job.py`. The plan expected a 30s probe to cover the Error 802 race; we bumped it to 120s after v1 and v2 both failed. Both v3 runs still hit the 120s timeout. Logs show CUDA never became available on those particular h200 nodes.

Summary of job attempts (all four Tier 1 submissions failed, both Tier 2 submissions succeeded):

| Run | v1 job ID | v2 job ID | v3 job ID | final status |
|---|---|---|---|---|
| legacy Tier 1 | 69eb9cfbd70108f37acddd00 ERROR | 69eba2dad2c8bd8662bcc9f8 ERROR | 69eba3e9d70108f37acddd2e ERROR | no data |
| legacy Tier 2 | 69eb9cfbd2c8bd8662bcc9b4 **COMPLETED** | — | — | **complete** |
| batch Tier 1 | 69eb9cfbd70108f37acddd02 ERROR | 69eba2dfd70108f37acddd26 ERROR | 69eba3eed2c8bd8662bcca08 ERROR | no data |
| batch Tier 2 | 69eb9cfcd70108f37acddd04 ERROR (ModuleNotFoundError: branch not pushed) | 69eba2e4d70108f37acddd28 **COMPLETED** | — | **complete** |

The Tier 1 failure rate (6/6 across all submissions) combined with the Tier 2 success rate (2/3) suggests the h200 pool has a non-trivial fraction of nodes with broken CUDA drivers at this time. Further retries were not attempted to keep compute cost in check.

## Files

- Legacy Tier 2 logs: `/tmp/log-job1-legacy-tier2.txt` (locally, not committed)
- Batch Tier 2 logs: `/tmp/log-batch-tier2-v2.txt` (locally, not committed)
- Byte-identity CPU gate evidence: `reports/byte-identity/2026-04-24/` (committed in `6fcfa96`)

## Phase 1 pipeline profiling attempt (Tier 2, 2026-04-24)

Per `docs/plans/2026-04-24-tier2-throughput-tuning.md` Task 1.2, we landed a `--profile-pipeline` instrumentation flag in `batch_extraction.py` (commit `e387f8d`) and submitted one H200 benchmark job to capture the pipeline phase-share profile at the batch Tier 2 v2 config.

**Outcome: blocked on infrastructure.** Two submissions against the same h200 pool that completed the v2 batch Tier 2 run earlier that day:

| Submission | Result |
|---|---|
| `69ebc29ad2c8bd8662bccb4e` | Cancelled manually after first probe iteration surfaced the Error 802 `UserWarning`. The warning itself isn't fatal — the retry probe at `benchmark_hf_job.py:693` is designed to keep retrying — so this cancel was premature. |
| `69ebc3bad70108f37acdddc6` | Ran the full 120 s retry probe; every attempt hit Error 802; aborted with "CUDA not available after 120s wait" (exit 2). |

Same failure mode as the six Tier 1 submissions above.

### Fallback run on a100-large

With h200 blocked, we ran the same config on `a100-large` (job `69ebc57cd2c8bd8662bccb6c`) to capture at least one profile datapoint:

```
[pipeline-profile] wall=89.2s gpu_active=86.4s (97%) gpu_idle=2.1s (2%)
  post_pool_saturated=79.2s (89%) pre_pool_saturated=77.7s (87%)
  writer_waiting=89.1s (100%) workers=94
```

Summary: rows=3580, wall=101.0 s, throughput=**35.46 docs/sec**, gpu=A100-SXM4-80GB, peak_mem=3.40 GB.
F1: overall **0.895** (P=0.905, R=0.886), clean **0.914**, clean_relocated 0.907 — F1 parity confirmed.

**Acceptance gates (all pass):** profile line present; `gpu_active + gpu_idle = 88.5 s ≈ wall 89.2 s` (within 1% sanity); `gpu_active = 97%` crosses the 50% diagnostic threshold.

**Read:** On A100, the GPU is saturated (97% active, 2% idle) — the opposite of the plan's Phase-1 hypothesis that the GPU is idle ≥80% of the run. Big caveat: the A100 host reports **94 workers** (`cpu_count - 2`) vs **14** on h200. That 6.7× CPU parallelism plausibly explains why the post-stage stops starving the GPU on A100. Relative throughput supports this: 35 docs/sec on A100 vs 50 docs/sec on h200 at the same config — the A100 is bound by how fast its GPU can encode, while h200 at the same batch config finishes encoding earlier and waits on CPU.

This means the **h200 bottleneck profile is still the one we need**; the A100 profile tells us that on a CPU-rich host at this batch size, the GPU is the floor rather than the ceiling. Phase 2 lever selection must wait for an h200 profile.

**Action:** when the h200 pool recovers, rerun with the same invocation (`phase1-profile-tier2` run name). The instrumentation and submit script are unchanged.

Per the tuning plan's non-goals ("H200 CUDA Error 802 infrastructure triage… out of scope. If any Tier 2 Phase 2/3 run fails with Error 802, tuning is paused pending infrastructure recovery"), Phase 1 data capture is paused. The instrumentation code is merged and idle — the next attempt can re-use the identical invocation. When the h200 pool stabilises, resubmit with:

```
--split test --subdirs data,arxiv-latex-extract \
--dtype bf16 --enable-paragraph-prefilter \
--paragraphs-per-batch 4096 --encode-batch-size 512 --queue-depth 128 \
--profile-pipeline --no-push --run-name phase1-profile-tier2
```

Expected billed cost for the rerun: ≈ $0.30.

## Conclusion

The pipelined batch engine is **correct** (F1 parity on every bucket, identical bucket-level metrics to the legacy baseline) and **faster** (50.4 vs 22.78 docs/sec in Tier 2 on a completed H200 run). It does not hit the design doc's 250 docs/sec target.

Phase 2 tuning (Experiments A, B, and a sharded-harness side-quest) in `docs/perf/tier2-throughput-tuning.md` established that **no single knob closes the 5× gap**. Worker count, pipeline batch size, and even 4-GPU sharding produce single-digit-to-2× improvements at best; the prefilter narrows Tier 2's GPU workload so far that end-to-end wall is dominated by pipeline overhead and fixed per-process setup, not encode. **Recommendation: revise the Tier 2 target to ~50–60 docs/sec and ship.** See `tier2-throughput-tuning.md` for full numbers, diagnosis, and deferred architectural options.
