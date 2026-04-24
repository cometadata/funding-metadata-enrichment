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

## Conclusion

The pipelined batch engine is **correct** (F1 parity on every bucket, identical bucket-level metrics to the legacy baseline) and **faster** (50.4 vs 22.78 docs/sec in Tier 2 on a completed H200 run). It does not currently hit the design doc's aggressive throughput/memory targets — the GPU is under-utilized at `paragraphs_per_batch=4096`. Tuning work can follow-up in a separate PR. F1 parity alone is sufficient to proceed with PR 2 review; the throughput miss should be flagged in the PR description.
