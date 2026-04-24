# ColBERT extractor throughput — per-phase attribution and committed fix

## Summary

The statement-only ColBERT extractor (`funding_statement_extractor.statements.extraction.SemanticExtractionService` using `lightonai/GTE-ModernColBERT-v1` via `pylate==1.4.0`) runs at ~898 ms / doc on H200 bf16 at `--batch-size 512`. The dominant cost is **not** the ColBERT encode as the original plan assumed — it is `pylate.rank.rerank` being called 32 times per document, each call redundantly padding and moving the same `documents_embeddings` tensor.

**Root cause:** the extractor calls `rank.rerank(..., documents_embeddings=[documents_embeddings])` 32 times per document (one per query, `extraction.py:270-271`), and each rerank invocation runs `torch.nn.utils.rnn.pad_sequence(...)` and `.to(device)` on the exact same paragraphs (pylate `rank/rank.py:128-138`). That pad+move averages **393 ms / doc (44%)** on H200.

**Committed recommendation — three layered fixes in priority order:**

1. **Hoist pad+move out of the rerank loop** (primary, this PR): pre-pad and pre-move `documents_embeddings` once per document, either by patching `pylate.rank.rerank` or by inlining a minimal rerank loop in `SemanticExtractionService.extract_funding_statements` that calls `colbert_scores` directly with shared document embeddings. Expected: ~380 ms/doc saved, taking per-doc time to ~520 ms (42% reduction).
2. **Batch queries + pre-convert + hoist `doc_ids`** (same PR as hoist, tiny additional diff): one `colbert_scores` einsum across all 32 queries instead of 32 sequential calls; store query embeddings already as tensors; compute `doc_ids` once per doc. Expected: another ~110 ms saved, ~410 ms/doc total.
3. **Regex pre-filter paragraphs before encoding** (separate PR, architectural): the average doc has ~300 paragraphs, of which typically 1–3 contain funding statements; all encode and rerank work scales with paragraph count. Filter paragraphs by keyword match (the same keyword set `_is_likely_funding_statement` already uses) before `model.encode`. Expected: ~100–150 ms/doc total, an 8–9× speedup over baseline. Requires measuring recall delta on the held-out split.

At the planned 14,261-doc train benchmark, ~$18 at $5/hr h200 drops to ~$10 after fix 1, ~$8 after fix 2, and ~$2–3 after fix 3. See the "Stacked additional wins" section for details.

Do **not** apply `torch.compile(mode='reduce-overhead')` — it is 8.7× slower for this workload (dynamic paragraph-count shapes force continuous recompilation). Default mode is worth a validation run only after fixes 1–3 are measured.

## Measurements

Environment: H200 single-GPU, bf16 weights, `--batch-size 512`, `--top-k 5`, `--threshold 10.0`, 50 docs from `cometadata/arxiv-funding-statement-extraction` `data/train.jsonl`, seed=42. Script: `scripts/profile_on_h200.py`. Commit: `statement-only-extraction @ dd1816e`.

### H200 per-doc breakdown (bf16, BS=512, n=49 excl. warmup)

| Phase | Mean ms | Median ms | p95 ms | % of wall-clock |
|---|---:|---:|---:|---:|
| rerank_total | 527 | 273 | 1354 | **69.6%** |
| model_encode_doc | 199 | 146 | 391 | 26.4% |
| filter_is_likely_funding_statement | 30 | 10 | 86 | 4.0% |
| paragraph_split + other filters | < 1 | < 1 | < 2 | < 0.1% |
| (unattributed) | 141 | — | — | 15.7% |
| **wall-clock per doc** | **898** | — | — | 100% |

Warmup doc 0 (excluded from means): 1290 ms.

Unattributed (~141 ms) is loop overhead inside `extract_funding_statements`: `doc_ids = list(range(len(paragraphs)))` allocated 32× per doc, Python iteration over 32 queries, `seen_statements` set lookups, top-k slicing, and CUDA sync slack between phases. Not a primary target.

### Rerank internals, summed across all 32 calls per doc (H200, n=49)

| Phase | Mean ms | % of rerank | Note |
|---|---:|---:|---|
| **rerank_pad_sequence** | **393** | **74.6%** | `torch.nn.utils.rnn.pad_sequence` on CPU. Documents are identical across all 32 calls — 31/32 of this is waste. |
| rerank_colbert_scores | 93 | 17.7% | `torch.einsum('ash,bth->abst', ...)` GPU work |
| rerank_convert_tensor | 22 | 4.2% | `func_convert_to_tensor` converts numpy arrays to torch tensors, 32× per doc |
| rerank_construct | 10 | 1.9% | RerankResult list comprehension |
| rerank_cpu_sync_tolist | 2 | 0.4% | `.cpu().tolist()` |
| rerank_device_move | 2 | 0.4% | `.to(cuda)` |
| rerank_sort | 1 | 0.3% | `torch.sort` |

rerank call count: 1600 = 32 queries × 50 docs ✓.

### Effect of `torch.compile(mode='reduce-overhead')` (H200, bf16, BS=512, n=49)

| Phase | Baseline | With compile | Delta |
|---|---:|---:|---:|
| model_encode_doc | 199 ms | **1741 ms** | **+773%** |
| rerank_total | 527 ms | 557 ms | +6% |
| filter_is_likely | 30 ms | 22 ms | −28% |
| wall-clock / doc | 898 ms | **2461 ms** | **+174%** |

Warmup with compile: 64 s (vs 1.3 s baseline). Each document has a different paragraph count, and `reduce-overhead` mode uses CUDA graphs keyed on shape — so virtually every document triggers a recompilation. Do not apply.

### Supporting: local MPS bf16 (n=49)

| Phase | Mean ms | % of wall-clock |
|---|---:|---:|
| model_encode_doc | 16,466 | 99.1% |
| rerank_total | 146 | 0.9% |
| filter_is_likely | 6 | 0.0% |
| wall-clock / doc | 16,632 | 100% |

Local MPS is 83× slower than H200 for encode but ~3.6× faster for rerank (MPS host CPU = Mac M-series; H200 host CPU is a shared cloud Xeon). **Mac-based rerank timing extrapolates poorly to H200.** See `feedback_mac_cpu_is_faster_than_hf_h200_host_cpu.md`.

## How the prediction went wrong

The investigation plan predicted H200 encode at 849–860 ms (90–92% of the 940 ms baseline) and rerank at ~86 ms. Actual H200 numbers are inverted: encode 199 ms (22%), rerank 527 ms (70%). Two reasons:

1. **Mac CPU timing does not transfer to H200 host CPU.** pad_sequence on Mac MPS averaged 64 ms/doc; on H200 it averaged 393 ms/doc. The h200 flavor provides 23 vCPUs of shared cloud Xeon, much slower per-thread than Apple M-series for Python/numpy/torch.nn CPU ops.
2. **H200 is very fast for short-sequence encode at BS=512.** ~300 paragraphs × ~40 tokens fits well within one forward pass; H200 wall-clock is ~200 ms not 850 ms. The "short sequences starve tensor cores" hypothesis was partly wrong — encode is CPU-launch-bound, but cross-doc batching buys far less than expected because encode is already only 22% of the budget.

## Committed fix recommendation

**Fix the 32× redundant pad+device_move in the rerank loop.** Two equivalent implementations — pick whichever is cleaner:

### Option A — patch pylate.rank.rerank to accept pre-shaped tensor (smaller diff)

Short monkey-patch installed at `SemanticExtractionService._get_model(...)` startup. In the rerank path: accept `documents_embeddings` already as a `(batch, max_tokens, emb_dim)` tensor on the target device and skip the pad+move. Extractor does the pad+move once per doc before the query loop.

### Option B — inline a minimal rerank loop in extraction.py (self-contained)

In `SemanticExtractionService.extract_funding_statements` between the current lines 251 and 259:

```python
import torch
from pylate.scores import colbert_scores

device = next(model.parameters()).device
doc_tensors = [torch.as_tensor(e) for e in documents_embeddings]
padded_docs = torch.nn.utils.rnn.pad_sequence(
    doc_tensors, batch_first=True, padding_value=0
).to(device)  # ONE pad+move, shape (n_paragraphs, max_toks, emb_dim)

doc_ids = list(range(len(paragraphs)))
for query_name, query_text in queries.items():
    q = torch.as_tensor(query_embeddings_map[query_name]).to(device)
    if q.ndim == 2:
        q = q.unsqueeze(0)
    scores = colbert_scores(queries_embeddings=q, documents_embeddings=padded_docs)[0]
    sorted_scores, sorted_idx = torch.sort(scores, descending=True)
    reranked = [
        {"id": doc_ids[i], "score": float(s)}
        for s, i in zip(sorted_scores[:top_k].cpu().tolist(), sorted_idx[:top_k].tolist())
    ]
    # ... existing filter/construct block unchanged, using reranked[:top_k]
```

Option B is preferred — no pylate monkey-patch, no behavior change to other callers, pad_sequence runs once per doc instead of 32×, and the `[:top_k]` + `.cpu().tolist()` is done on 5 elements instead of 300.

### Verification of the fix (when implemented)

- Re-run `scripts/profile_on_h200.py --num-docs 50 --batch-size 512 --dtype bf16` on h200.
- Expect `rerank_total` to drop from ~527 ms to **~150 ms** per doc (pad_sequence drops by ~380 ms, other rerank internals stay ~150 ms).
- Expect per-doc wall-clock to drop from ~898 ms to **~520 ms**.
- Confirm predicted_statements are byte-identical to the baseline 50-doc output JSON for the same seed=42 sample (no semantic regression).

## Stacked additional wins

520 ms/doc is still dominated by work that doesn't need to scale with paragraph count or query count. Three tiers of further speedups, ordered by diminishing return on engineering effort:

### Tier 1 — nearly-free follow-ons (same PR as the hoist)

No semantic change; byte-identical predictions. All measured against the post-hoist baseline of ~520 ms/doc.

1. **Batch all 32 queries into one `colbert_scores` einsum.** Currently the rerank loop runs 32 sequential calls of `colbert_scores(q_emb.unsqueeze(0), padded_docs)[0]`. Replace with one call: stack queries into a `(32, max_q_toks, emb_dim)` tensor (use `pad_sequence` on query embeddings once at service init, cached alongside `_query_embeddings_cache`), call `colbert_scores` once to get `(32, n_paragraphs)`, then loop over the result rows for filtering. Saves ~85 ms (93 ms → ~8 ms) by collapsing 32 small kernel launches into one larger launch that better utilizes H200.
2. **Pre-convert query embeddings to tensors once** outside the loop. `func_convert_to_tensor(q_emb)` runs on every iteration in the current code; at service init or during `_get_query_embeddings`, store them already as tensors. Saves ~22 ms (trivially).
3. **Hoist `doc_ids = list(range(len(paragraphs)))` out of the query loop** — currently allocated 32× per doc. ~2 ms, near-free.

Expected cumulative after Tier 1 (hoist + batched queries + pre-convert + doc_ids hoist): **~410 ms/doc, ~2.2× over the 898 ms baseline.**

### Tier 2 — regex pre-filter paragraphs before encoding (architectural, but high-leverage)

The average doc has ~300 paragraphs, of which typically 1–3 contain funding statements. Everything after paragraph splitting — encode (~199 ms) and every rerank component that scales with paragraph count (pad_sequence, colbert_scores einsum, convert_tensor) — is spent reranking paragraphs that have essentially zero chance of being funding.

Patch: between `paragraphs = _split_into_paragraphs(content)` and `model.encode(paragraphs, ...)` in `extraction.py:245-252`, run a cheap regex filter over paragraphs and keep only those containing any of the funding keyword set already used by `_is_likely_funding_statement` (`extraction.py:100-102`) — `fund|grant|support|acknowledg|award|sponsor|thank|scholarship|fellowship|financial`. Retain ±1 neighboring paragraph for context to cover the occasional "We acknowledge …" header split. Preserve original paragraph indexes so `paragraph_idx` in the output matches the pre-filter position.

Projected impact at 10–15 surviving paragraphs per doc:
- encode: 199 ms → ~10 ms (95% fewer paragraphs encoded)
- rerank pad_sequence (post-hoist): ~12 ms → ~0.5 ms
- rerank colbert_scores (post-batched): ~8 ms → ~0.5 ms
- rerank convert_tensor: ~22 ms → ~1 ms
- filter_is_likely: 30 ms → ~3 ms (runs on reranked top-k, which shrinks proportionally)

Expected after Tier 2 combined with Tier 1: **~100–150 ms/doc, 6–9× over the 898 ms baseline.**

**Recall risk:** any funding statement that uses no keyword in the filter list gets dropped. Your existing `_is_likely_funding_statement` already relies on the same keyword set for low-score paragraphs, so most statements that would survive that filter should also survive a pre-filter with the same regex. High-score paragraphs above `threshold=10.0` currently bypass the keyword check; pre-filter would catch those too. Mitigation: run the pre-filter + post-filter combined against the held-out test split and report recall delta vs the no-pre-filter baseline. Tune the regex list to expand until recall is within 1–2 pts of the baseline, or make pre-filter opt-in behind a flag.

### Tier 3 — hold until Tier 1 + Tier 2 are measured

- **`torch.compile(mode='default')` on encode**: the failed compile experiment used `reduce-overhead` (CUDA graphs keyed on shape). Default mode does not use CUDA graphs and may not recompile on every new shape. Worth a single validation run post-Tier-2, once encode is ≤10% of per-doc time; before that, the effort-to-return ratio is bad.
- **Cross-doc encode batching**: after Tier 2 encode is ~10 ms/doc, flattening paragraphs across N docs into one forward pass would save maybe 50% of that. Adds non-trivial complexity (doc boundary tracking, memory sizing). Skip unless encode becomes the new bottleneck.
- **Async/pipelined processing** (encode doc N+1 while rerank for doc N runs): real engineering investment. Not worth it until Tier 1 + Tier 2 are shipped and measured.

### Consolidated projection

| Configuration | Per-doc wall-clock | 14,261-doc train cost @ $5/hr h200 |
|---|---:|---:|
| Current baseline (H200 measured) | 898 ms | ~3.6 hr, ~$18 |
| Committed hoist (pad+move once per doc) | ~520 ms | ~2.1 hr, ~$10 |
| + Tier 1 (batched queries + pre-convert + hoist doc_ids) | ~410 ms | ~1.6 hr, ~$8 |
| + Tier 2 (regex pre-filter before encode) | ~100–150 ms | ~25–35 min, ~$2–3 |
| + Tier 3 (compile, cross-doc, async) | TBD | TBD |

## Reproducibility

Commands, in order, from this repository root:

```bash
# Local CPU fp32 sanity check (fast, ~3 min)
.venv/bin/python3 scripts/profile_extraction.py \
    --device cpu --dtype fp32 --num-docs 3 --batch-size 64 \
    --output reports/profile_cpu_fp32_3.json

# Local MPS bf16 full profile (~10 min)
.venv/bin/python3 scripts/profile_extraction.py \
    --device mps --dtype bf16 --num-docs 50 --batch-size 512 \
    --output reports/profile_mps_bf16_50.json

# H200 baseline (bf16, BS=512, ~5 min, ~$0.42)
python -c "from huggingface_hub import HfApi; import base64, os; \
    api = HfApi(); token = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip(); \
    b64 = base64.b64encode(open('scripts/profile_on_h200.py','rb').read()).decode(); \
    cmd = ['bash','-c', f'set -euxo pipefail && apt-get update -qq && apt-get install -y -qq git && pip install --quiet --root-user-action=ignore uv && echo {b64} | base64 -d > /tmp/p.py && rm -rf /root/.cache/uv/environments-v2 && uv run /tmp/p.py --num-docs 50 --batch-size 512 --dtype bf16']; \
    print(api.run_job(image='pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel', command=cmd, secrets={'HF_TOKEN': token}, flavor='h200', timeout='20m').id)"

# Retrieve logs, extract JSON:
# hf jobs logs <job_id> > /tmp/log.txt
# sed -n '/===PROFILE_JSON===/,/===END_PROFILE_JSON===/p' /tmp/log.txt | sed '1d;$d' > reports/profile_h200_bf16_50.json
```

Reports checked in alongside this doc:
- `reports/profile_mps_bf16_50.json` — 50 docs, Mac MPS bf16, BS=512
- `reports/profile_cpu_fp32_3.json` — 3 docs, Mac CPU fp32, BS=64
- `reports/profile_h200_bf16_50.json` — 50 docs, H200 bf16, BS=512 (the authoritative number)
- `reports/profile_h200_bf16_compile_50.json` — 50 docs, H200 bf16 BS=512, `--compile` (negative result)
- `reports/profile_h200_cpu_fallback.json` — 50 docs, H200 node host-CPU (CUDA init failed on first attempt; documents the Mac/H200-host CPU speed gap)
