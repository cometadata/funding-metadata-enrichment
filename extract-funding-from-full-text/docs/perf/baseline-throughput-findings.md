# ColBERT extractor throughput — per-phase attribution and committed fix

## Summary

The statement-only ColBERT extractor (`funding_statement_extractor.statements.extraction.SemanticExtractionService` using `lightonai/GTE-ModernColBERT-v1` via `pylate==1.4.0`) runs at ~898 ms / doc on H200 bf16 at `--batch-size 512`. The dominant cost is **not** the ColBERT encode as the original plan assumed — it is `pylate.rank.rerank` being called 32 times per document, each call redundantly padding and moving the same `documents_embeddings` tensor.

**Root cause:** the extractor calls `rank.rerank(..., documents_embeddings=[documents_embeddings])` 32 times per document (one per query, `extraction.py:270-271`), and each rerank invocation runs `torch.nn.utils.rnn.pad_sequence(...)` and `.to(device)` on the exact same paragraphs (pylate `rank/rank.py:128-138`). That pad+move averages **393 ms / doc (44%)** on H200.

**Committed recommendation:** hoist pad+move out of the rerank loop — pre-pad and pre-move `documents_embeddings` once per document, and either (a) patch `pylate.rank.rerank` to accept a pre-padded tensor, or (b) inline a minimal rerank loop in `SemanticExtractionService.extract_funding_statements` that calls `colbert_scores` directly with the shared document embeddings. Expected savings: ~31/32 × 393 ms ≈ **380 ms / doc, a 42% reduction**. At the planned 14,261-doc train benchmark this turns a ~3.7-hour run into ~2.1 hours (~$11 instead of ~$18 at $5/hr h200).

Do **not** apply `torch.compile(mode='reduce-overhead')` — it is 8.7× slower for this workload (dynamic paragraph-count shapes force continuous recompilation).

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

### What NOT to attempt as "while we're here" extensions

- **Cross-doc encode batching**: encode is only 22% of per-doc time and already well-saturated at BS=512 on ~300-paragraph docs. Cross-doc batching adds architectural complexity for a ≤20% local win; do this only if the pad_sequence fix still leaves you encode-bound.
- **Vectorizing queries across a single `colbert_scores` call** (`queries_embeddings` shape `(32, n_q_toks, emb_dim)`): the 32× sequential einsum is only ~93 ms on H200, under 11% of current per-doc time. Not a priority.
- **torch.compile in any mode**: confirmed slower on this workload; do not revisit unless the encode cost becomes dominant post-fix.

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
