# Funding-entity-extractor throughput — first real run + improvements catalog

## Summary

The first real benchmark of `funding-extract` on `cometadata/arxiv-funding-statement-extractions` (single A100-80GB, `predictions/arXiv_src_2009_041.parquet`, 111 rows / 121 statements) ran at **0.42 statements/sec**, **68.5 generated tokens/sec**, **p50 latency 1.9s/statement**. An A100-80GB on a Llama-3.1-8B with a small LoRA should sustain ~1500 generated tok/s under load, so the GPU is sitting idle ~95% of the time.

**Root cause:** the driver loop is row-serial. With `--concurrency=256` set on the CLI, the `asyncio.Semaphore` inside `extract_statements` *can* hold 256 in-flight requests, but the *driver* in `run_driver._run_async` and `benchmark_hf_job._run_benchmark_extraction` calls `extract_statements` once per row inside a `for row: await extract_statements(row.statements)` loop. Most rows have 1 statement; some have 2. So the actual peak in-flight is `concurrency * statements_per_row ≈ 1-2`, not 256. The `--concurrency` flag is effectively a no-op for this dataset shape.

**Reference design that doesn't have this problem:** `pdf-ocr/pdf_ocr/server.py:VLMClient.infer_batch` and `pdf-ocr/pdf_ocr/convert.py`. Their `infer_batch(images: Sequence) -> List[str]` is structurally identical to our `extract_statements`: takes a list, fans out via `asyncio.Semaphore(max_concurrency)` over `asyncio.gather(...)`, returns ordered results. They get away with calling it batch-by-batch because:
- VLM OCR is heavy: ~1-3s per page at the model already,
- their default `batch_size=4` is enough to saturate a GPU on big-context vision work,
- they're streaming through pages with controlled batching, not chasing peak throughput.

We can't get away with that. Our extraction is light (~50-100 output tokens per call), the GPU is fast relative to a single in-flight request, and we *want* to chase peak throughput. So our driver loop needs to flatten across the row boundary that pdf-ocr's batch boundary already provides for them.

---

## First-run measurements

Source: HF Job `69ed5ddad70108f37acdf30d` on `a100-large` flavor, `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`, `vllm` resolved by uv from PEP 723 deps. Branch: `funding-entity-extraction @ f2d8af5`. Run name: `first-real-run`.

| Metric | Value | Notes |
|---|---:|---|
| n_rows | 111 | After filtering empty `predicted_statements` |
| n_statements | 121 | ~1.09 statements/row average |
| wall_seconds | 290.6 | ~4.8 min |
| **rows / sec** | **0.38** | |
| **statements / sec** | **0.42** | |
| **gen tokens / sec** | **68.5** | A100-80GB on Llama-8B should do ~1500+ |
| mean_latency_ms | 2475 | Per-statement, includes server queue + generate |
| p50_latency_ms | 1928 | |
| p95_latency_ms | 7514 | Long tail — likely the `max_tokens=512` truncations (#2 below) |
| p99_latency_ms | 7682 | |
| parse_success_rate | 0.942 | 114 / 121 |
| nonempty_extraction_rate | 0.909 | 110 / 121 |
| mean_funders_per_statement | 2.11 | Suggests model finds multiple funders correctly when statements are rich |
| mean_awards_per_funder | 1.04 | |

**Spot-checked extractions are correct.** Examples:
- `arxiv:2009.07429` → `NSFC 91746301`
- `arxiv:2009.07290` → `FONDECYT 11190427/11171148`; `ERC 852386` ("HoloHair", Starting Grant)

**All 7 parse failures share a signature:** `ParseError: invalid JSON: Unterminated string at pos ~1880-1900`. The raw column shows the JSON starts well-formed, then truncates mid-string at exactly the point where `max_tokens=512` runs out. These statements have multiple funders with full schemes/titles, which need ~700-1000+ tokens to serialize; the default of 512 is too tight. See improvement #2.

**The `__run_config__` row in the metrics parquet is mostly null.** Only `bucket` and `n_rows` survived. `Dataset.from_list([metrics_row, run_config])` apparently inferred schema from the first dict and silently dropped the run-config-only fields (`run_name`, `gpu_name`, `vllm_version`, etc.). See improvement #6.

---

## Improvement opportunities, ranked by impact

### 1. **Flatten across rows in the driver loop** — primary fix, ~50× throughput on this dataset shape

**Where it lives now:**
- `src/funding_entity_extractor/run_driver.py:_run_async`, the `for row in iter_input_rows(...): results = await extract_statements(...)` loop
- `benchmark/benchmark_hf_job.py:_run_benchmark_extraction`, the analogous `for i, row in enumerate(rows): results = await extract_statements(...)` loop

**The pdf-ocr point of comparison:** `pdf-ocr/pdf_ocr/convert.py:254-261`:
```python
for batch in pipeline:                                          # batch_size=4
    markdowns = _infer_with_retry(client, batch, ...)           # 4 in flight
```
Their batch *is* the unit of parallelism. For their workload (heavy VLM, ~1-3s/page), 4-way parallelism is fine. For ours (light text gen, ~100ms minimum if the GPU were busy), it isn't.

**The fix:** before calling `extract_statements`, build a flat `[(row_idx, stmt_idx, statement_text)]` list across the *entire* input (or across a tunable mega-batch if memory is a concern), call `extract_statements` once with all the statements, then re-bucket the returned per-statement results by `row_idx`. Code sketch:

```python
async def _run_async(cfg: RunConfig) -> int:
    # ... read input, set up writer, skip resume keys (unchanged) ...

    # Phase 1: collect input
    pending: list[dict] = []                                # rows we still owe output for
    flat_statements: list[str] = []                         # 1-D list of all statements
    flat_back_index: list[tuple[int, int]] = []             # parallel: (pending_idx, stmt_idx)
    for row in iter_input_rows(input_path, text_field=cfg.text_field):
        if tuple(row[k] for k in cfg.row_id_fields) in skip_keys:
            continue
        ri = len(pending)
        pending.append(row)
        for si, stmt in enumerate(row[cfg.text_field]):
            flat_statements.append(stmt)
            flat_back_index.append((ri, si))

    # Phase 2: one fan-out for everything (semaphore caps in-flight at cfg.concurrency)
    flat_results = await extract_statements(
        flat_statements,
        vllm_url=cfg.vllm_url, served_name=cfg.served_name,
        concurrency=cfg.concurrency,
        max_retries=cfg.max_retries, request_timeout=cfg.request_timeout,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
    )

    # Phase 3: regroup back into rows and write
    per_row: list[list[StatementExtraction]] = [
        [None] * len(r[cfg.text_field]) for r in pending  # type: ignore[list-item]
    ]
    for r, (ri, si) in zip(flat_results, flat_back_index):
        per_row[ri][si] = r

    with ExtractionWriter(write_target, schema=out_schema, write_batch_size=cfg.write_batch_size) as writer:
        for row, results in zip(pending, per_row):
            writer.write(_build_output_row(row, results))
```

**Memory cost:** holds all input rows + all results in memory until the write phase. For 110 rows × ~5 KB each + 121 results × ~2 KB each ≈ <1 MB — fine. For a ~10k-row shard, similar order — still fine. For the full 9999-shard dataset, you'd want to chunk this (process N rows at a time) but still call `extract_statements` once per chunk, not once per row.

**Streaming variant** (if you want output to start landing while extraction is in-flight): use `asyncio.as_completed` over the per-statement futures inside `extract_statements`, route each completion to a `(row_idx, stmt_idx)` slot, and emit a row to the writer as soon as all its slots fill. This is mostly cosmetic for our scale; the all-at-once version is fine.

**Expected impact, this dataset:** 121 in-flight at once (capped at `concurrency=256`). At even just 200ms/statement on a saturated A100, that's ~600 ms total wall time for the model phase. Adding network + writer + warmup, expect end-to-end somewhere in the 5-15s range, vs. the current 290s — ~20-50× speedup. Tokens/sec should jump from 68 to several hundred or more.

### 2. **Bump `--max-tokens` default from 512 to 1024 (or higher)** — closes the 7 parse failures

**Symptom:** every parse failure on this run was `Unterminated string at pos ~1880-1900` — the JSON ran out of token budget mid-serialization, not because the model misbehaved.

**Where it lives:**
- `src/funding_entity_extractor/cli.py:r.add_argument("--max-tokens", type=int, default=512)`
- `benchmark/benchmark_hf_job.py:p.add_argument("--max-tokens", type=int, default=512)`

**Action:** bump both defaults to 1024. Per-statement gen tokens will go up modestly (the median statement only emits ~200-400 tokens of JSON, well under either ceiling), but the long-tail rich statements that need 700-1000+ tokens will stop truncating. Expected: parse_success_rate from 94.2% → ~99%+.

There is no perf cost to raising the cap when most outputs don't approach it; vLLM stops as soon as the model emits an EOS regardless of `max_tokens`.

### 3. **Hoist `httpx.AsyncClient` to driver scope** — minor win once #1 lands; significant if it doesn't

**Where it lives now:** `src/funding_entity_extractor/extract.py:extract_statements` constructs a fresh `httpx.AsyncClient(base_url=..., limits=...)` per call (extract.py ~line 148). With the row-serial driver, that's one open/close per row.

**The pdf-ocr point of comparison:** `pdf-ocr/pdf_ocr/server.py:VLMClient.__init__`:
```python
self._client = AsyncOpenAI(api_key="vllm", base_url=f"{self.base_url}/v1")  # one client, shared
```
Constructed once, reused for every `infer_batch` call across the whole run. TCP connection pool persists.

**The fix:** add an optional `client: httpx.AsyncClient | None = None` parameter to `extract_statements`. If not provided, create one (current behavior). Update the driver to construct one client at run start and pass it through. Same for the benchmark.

**Why this isn't urgent:** with #1 implemented, `extract_statements` is called *once* per run, so the client overhead is amortized over thousands of statements. The reason to do it anyway: tests still call `extract_statements` per-test and create a fresh client each time, and the API gets cleaner.

### 4. **Switch from raw httpx to `openai.AsyncOpenAI`** — cosmetic; we already declared the dep

`pyproject.toml:18` declares `openai>=1.50` but we never import it. Our `_post_chat_completions` (extract.py) hand-builds a JSON body, hand-parses the response. `pdf-ocr/pdf_ocr/server.py:176` calls `await self._client.chat.completions.create(...)` and reads `response.choices[0].message.content`. The openai SDK handles the request shape, parses the response, and gives back typed objects.

**Why it's worth doing eventually:** the SDK gets bug fixes, new fields (e.g., logprobs, structured outputs) automatically; raw JSON poking does not. But there's zero throughput difference and the current code is well-tested, so this is post-#1 polish.

### 5. **Pipeline I/O off the inference path (overkill for v0; relevant for multi-shard)**

`pdf-ocr/pdf_ocr/convert.py:Pipeline` (lines 58-179) runs PDF rendering and batching in *two* `threading.Thread`s with bounded `queue.Queue`s (`page_queue`, `batch_queue`). The renderer thread reads pages off the PDF and queues them; the batcher groups them; the main thread pulls batches and dispatches inference. Inference never blocks on I/O, and rendering never blocks on inference.

**Why it doesn't matter for our v0:** parquet reads are fast (<1s for our largest 1MB file), and once #1 lands the inference phase is itself <30s, so I/O is a non-issue. **When it would matter:** if we move to a multi-shard orchestration where shard N+1's parquet should start downloading while shard N is mid-extraction. At that point, copy the `Pipeline` shape: a downloader thread + a `next_shard` queue + the main loop.

### 6. **Push run_config as its own config (separate `push_to_hub` call)** — fix the metrics-row data loss

**Symptom:** `metrics-first-real-run/train-00000-of-00001.parquet` has the run_config row as `bucket=__run_config__, n_rows=111, <everything else null>`. We lost `run_name`, `gpu_name`, `max_gpu_memory_mb`, `vllm_version`, `lora_repo`, `base_model`, `served_name`, `timestamp`, `concurrency`, `temperature`, `max_tokens`, `max_model_len`, `tensor_parallel_size`.

**Where it lives:** `benchmark/benchmark_hf_job.py:_push_to_hub`:
```python
metrics_combined = [metrics_row, run_config]
Dataset.from_list(metrics_combined).push_to_hub(repo_id, config_name=f"metrics-{run_name}", private=private)
```

**Cause:** `Dataset.from_list` infers schema from the *first* dict (or the union, depending on version), and the resulting parquet keeps only fields present in both, dropping run_config-only keys.

**The fix:** push run_config as its own dataset config (separate `config_name`), so each row is in a parquet with a homogeneous schema:
```python
Dataset.from_list([metrics_row]).push_to_hub(repo_id, config_name=f"metrics-{run_name}", private=private)
Dataset.from_list([run_config]).push_to_hub(repo_id, config_name=f"run-config-{run_name}", private=private)
```
Or stop using `Dataset.from_list` for the metrics-bundle case and write the parquets directly with explicit pyarrow schemas. The first option is simpler and matches the convention already used for `predictions-{run_name}`.

### 7. **Subdivide-on-failure for batched failures — not needed at our scale, but cheap insurance**

`pdf-ocr/pdf_ocr/convert.py:_infer_with_retry` (lines 182-210) catches an exception during a batch's `infer_batch`, halves the batch, and retries. Useful for "the whole batch hit a vLLM OOM" scenarios.

**Why we don't need it:** our retries are at the per-statement level inside `extract_one` (tenacity, exponential backoff on retryable HTTP statuses). If the batch as a whole fails (e.g., `extract_statements` raises) we'd lose the whole row. But `extract_statements` doesn't raise — it embeds errors inside individual `StatementExtraction` results. So the only way the whole call dies is an unhandled programming error, which subdivision wouldn't fix anyway.

### 8. **Live log-tailing of the vLLM subprocess** — UX nicety

`pdf-ocr/pdf_ocr/server.py:60-91` has a `_stream_output` thread that tails stdout/stderr of the vLLM subprocess and prints with a `[vLLM STDOUT]` prefix. Currently our benchmark redirects to `serve.log` and you only see it if you SSH to the job container.

**Action (optional):** add a `--tail-server-logs` flag that spawns two daemon threads to copy the subprocess pipes to the main stdout. Useful when iterating on `--max-model-len` or `--gpu-memory-utilization` and you want to see vLLM's startup chatter without `tail -f`.

### 9. **More tolerant readiness probe**

`pdf-ocr/pdf_ocr/server.py:wait_for_server` (lines 106-120) catches *any* exception and retries. Our `_wait_for_ready` in `benchmark/benchmark_hf_job.py` only catches `httpx.ConnectError, httpx.ReadError, httpx.TimeoutException`. If the server returns an unexpected exception (e.g., transient TLS handshake error), we'd bail. Trivial fix; cosmetic.

---

## Recommended action order

| # | Change | Effort | Impact | Verifiable how |
|---|---|---|---|---|
| 1 | Flatten driver across rows | 1-2 hours, plus update tests | ~20-50× throughput, makes `--concurrency` work as advertised | Re-run benchmark on same parquet; expect <30s wall, >200 stmts/sec, tokens/sec >500 |
| 2 | `--max-tokens` default → 1024 | 5 min, change default in two places | Parse rate 94% → ~99% | Same benchmark; expect 0-2 parse failures instead of 7 |
| 6 | run_config as separate config | 15 min | Recoverable run-config metadata | Re-run; check `run-config-{name}` config has full row |
| 3 | Hoist client to driver scope | 30 min, optional `client=` param + thread through | Marginal once #1 lands; cleaner API | Existing tests still pass |
| 4 | Switch to `openai.AsyncOpenAI` | 1 hour | Cosmetic; SDK fix-feed | Existing tests still pass |
| 5 | I/O pipelining | Future | Multi-shard scaling | N/A until multi-shard |
| 7 | Subdivide-on-failure | Skip | None | — |
| 8 | Tail server logs | 30 min | UX during iteration | Manual |
| 9 | Generic readiness probe | 5 min | Negligible | Manual |

Suggested first PR: **#1 + #2 + #6** together. Re-run on `predictions/arXiv_src_2009_041.parquet` and compare metrics. If throughput jumps as expected, scale to a larger shard or run multiple shards in parallel.

## Reproduction

```bash
# Re-run after fixes (note: HF jobs needs a bash prelude on the pytorch image
# because that image lacks `uv` and `git`; see extract-funding-from-full-text/CLAUDE.md
# §"HF Jobs (H200) profiling" for the canonical incantation).

python3 - <<'PYEOF'
from huggingface_hub import HfApi
import base64, os
token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
SCRIPT = "vllm-funding-entity-extraction/benchmark/benchmark_hf_job.py"
b64 = base64.b64encode(open(SCRIPT, "rb").read()).decode()
argv = "--input-file predictions/arXiv_src_2009_041.parquet --run-name <name>-rerun --push-to-hub cometadata/arxiv-funding-entity-extractions"
cmd = ["bash", "-c",
    "set -euxo pipefail && apt-get update -qq && apt-get install -y -qq git && "
    "pip install --quiet --root-user-action=ignore uv && "
    f"echo {b64} | base64 -d > /tmp/p.py && rm -rf /root/.cache/uv/environments-v2 && "
    f"uv run /tmp/p.py {argv}"]
job = HfApi().run_job(image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
                     command=cmd, secrets={"HF_TOKEN": token},
                     flavor="a100-large", timeout="1h")
print(job.id, job.url)
PYEOF
```
