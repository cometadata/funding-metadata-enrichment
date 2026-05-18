# Funding statement extraction — three-model cascade

A cascade for extracting (and cleaning) the funding-acknowledgment statement from
the markdown text of an arXiv paper. Outputs either the cleaned funding
statement string, or an empty string if the paper has no funding info.

Bench: best@0.85 F1 = **0.953**, strict tsr@0.95 F1 = **0.725** on
`cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test` test split (597 docs).

## Models

All three are CC0 on the Hugging Face Hub:

1. `cometadata/funding-chunk-classifier-modernbert-base` — ModernBERT-base
   binary classifier, decides which 8K-token chunk of a (potentially very long)
   paper to look at. Custom architecture (encoder + mean-pool + 1-linear); the
   repo includes `modeling.py` with the class.
2. `cometadata/funding-extraction-modernbert-base-spanhead` — ModernBERT-base
   span head: predicts start/end token indices + a "no-answer" logit on a
   single chunk. Custom architecture; the repo includes `modeling.py`.
3. `cometadata/funding-cleaning-qwen3-4b-lora` — Qwen3-4B-Instruct-2507 + LoRA
   (r=32, attn+MLP only). Takes a "rough span ± context" and rewrites it to
   match the canonical frontier-labeled form (strips LaTeX markers, joins
   paragraph breaks).

## Input

The cascade expects the **VLM-converted markdown** of an arXiv PDF (i.e. the
`vlm_markdown` field in the source dataset). PymuPDF text is too noisy to feed
in directly. Long docs (> 8K tokens) are fine — stage 1 chunks them.

## Dependencies

```
pip install "transformers>=4.45,<6" "peft>=0.14" "accelerate" "torch>=2.4" \
            "huggingface_hub>=0.26" rapidfuzz
```

Both ModernBERT models also need their custom `modeling.py` from the repo. The
snippets below grab them via `hf_hub_download`.

## End-to-end usage

```python
import importlib.util
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
MAX_TOK = 8192
STRIDE = 4096

CHUNK_REPO = "cometadata/funding-chunk-classifier-modernbert-base"
SPAN_REPO  = "cometadata/funding-extraction-modernbert-base-spanhead"
CLEAN_REPO = "cometadata/funding-cleaning-qwen3-4b-lora"
MB_BASE    = "answerdotai/ModernBERT-base"
QWEN_BASE  = "Qwen/Qwen3-4B-Instruct-2507"


def _load_class(repo: str, class_name: str):
    """Fetch modeling.py from an HF repo and import it as a module."""
    src = hf_hub_download(repo, "modeling.py")
    spec = importlib.util.spec_from_file_location(f"_{class_name}_mod", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


# ----- load the three models once -----
mb_tok = AutoTokenizer.from_pretrained(MB_BASE)

ChunkClassifier = _load_class(CHUNK_REPO, "ChunkClassifier")
chunker = ChunkClassifier(MB_BASE).to(DEVICE).eval()
chunker.load_state_dict(torch.load(
    hf_hub_download(CHUNK_REPO, "pytorch_model.bin"),
    map_location=DEVICE, weights_only=True,
))

SpanHead = _load_class(SPAN_REPO, "SpanHead")
spanner = SpanHead(MB_BASE).to(DEVICE).eval()
spanner.load_state_dict(torch.load(
    hf_hub_download(SPAN_REPO, "pytorch_model.bin"),
    map_location=DEVICE, weights_only=True,
))

qwen_tok = AutoTokenizer.from_pretrained(QWEN_BASE)
cleaner_base = AutoModelForCausalLM.from_pretrained(
    QWEN_BASE, dtype=torch.bfloat16, device_map=DEVICE
)
cleaner = PeftModel.from_pretrained(cleaner_base, CLEAN_REPO).eval()


# ----- single-doc extraction -----
SYSTEM_CLEAN = (
    "You are a funding statement cleaner. Given a rough extracted funding "
    "statement and its surrounding context from an academic paper, output "
    "the exact funding statement as it should appear in a database. Clean "
    "up LaTeX markers ($^{N}$, \\textsuperscript), hyphenated line breaks, "
    "and abnormal whitespace, but DO NOT paraphrase. If the rough span is "
    "not actually a funding statement, output the single word: NONE"
)


@torch.no_grad()
def extract_funding(vlm_markdown: str) -> str:
    """Returns the cleaned funding statement, or '' if none."""

    # --- Stage 1: pick the top-2 chunks ---
    enc = mb_tok(vlm_markdown, add_special_tokens=False,
                  return_offsets_mapping=True, truncation=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]

    if len(ids) <= MAX_TOK:
        chunks = [(0, len(ids))]
    else:
        chunks = []
        for st in range(0, len(ids), STRIDE):
            en = min(st + MAX_TOK, len(ids))
            chunks.append((st, en))
            if en == len(ids):
                break

    chunk_probs = []
    for st, en in chunks:
        c_ids = torch.tensor(ids[st:en]).unsqueeze(0).to(DEVICE)
        c_attn = torch.ones_like(c_ids)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logit = chunker(c_ids, c_attn).float()
        chunk_probs.append((torch.sigmoid(logit).item(), st, en))

    chunk_probs.sort(key=lambda x: -x[0])
    candidates = [c for c in chunk_probs[:2] if c[0] >= 0.4]
    if not candidates:
        return ""  # no funding-likely chunk

    # --- Stage 2: span head on each candidate; pick lowest no-answer prob ---
    best = None  # (no_a_prob, span_text, char_start, char_end)
    for _, st, en in candidates:
        c_offsets = offsets[st:en]
        c_ids = torch.tensor(ids[st:en]).unsqueeze(0).to(DEVICE)
        c_attn = torch.ones_like(c_ids)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            s_log, e_log, na_log = spanner(c_ids, c_attn)
        s_log = s_log.squeeze(0).float().cpu()
        e_log = e_log.squeeze(0).float().cpu()
        no_a = torch.sigmoid(na_log).item()
        if no_a >= 0.5:
            continue
        s = int(s_log.argmax())
        e = s + int(e_log[s : s + 300].argmax())
        if not (0 <= s <= e < len(c_offsets)):
            continue
        char_s = c_offsets[s][0]
        char_e = c_offsets[e][1]
        span_text = vlm_markdown[char_s:char_e].strip()
        if best is None or no_a < best[0]:
            best = (no_a, span_text, char_s, char_e)

    if best is None:
        return ""
    _, rough_span, char_s, char_e = best

    # --- Stage 3: cleanup LoRA ---
    ctx_l = vlm_markdown[max(0, char_s - 400) : char_s]
    ctx_r = vlm_markdown[char_e : char_e + 400]
    user = f"{ctx_l}<ROUGH>{rough_span}</ROUGH>{ctx_r}"
    messages = [
        {"role": "system", "content": SYSTEM_CLEAN},
        {"role": "user", "content": user},
    ]
    prompt_ids = qwen_tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(DEVICE)
    out = cleaner.generate(prompt_ids, max_new_tokens=512, do_sample=False,
                            pad_token_id=qwen_tok.eos_token_id)
    decoded = qwen_tok.decode(
        out[0, prompt_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    if decoded.upper() == "NONE":
        return ""
    return decoded


# Example
funding = extract_funding(open("paper.md").read())
print(funding or "(no funding statement)")
```

## Stage-by-stage details

### Stage 1 — chunk classifier
- Outputs a scalar in `[0, 1]` per 8K-token chunk.
- Decoding: take the top-2 chunks with `prob >= 0.4`. If none, return `""`.
- Calling convention: mean-pooled binary head, **bfloat16 inference is fine**.
- Standalone metrics (doc-level binary): F1 = 0.968 at threshold 0.5.

### Stage 2 — span head
- Outputs three things per chunk: `start_logits[seq]`, `end_logits[seq]`,
  `no_answer` scalar.
- Decoding:
  - If `sigmoid(no_answer) >= 0.5` → this chunk has no funding, skip it.
  - Else `start = argmax(start_logits)`, `end = start + argmax(end_logits[start:start+300])`.
  - Convert back to char offsets via the tokenizer's `return_offsets_mapping`.
- Among the surviving stage-1 chunks, pick the one with the **lowest** no-answer
  probability (most confident there's funding).
- Standalone metrics (with stage 1 retrieval): strict tsr@0.95 F1 = 0.722,
  best@0.85 F1 = 0.956.

### Stage 3 — cleanup LoRA
- Input format is **load-bearing**: the user message must be
  `<context_left><ROUGH>rough_span</ROUGH><context_right>`. The `<ROUGH>` tags
  tell the model what to clean.
- Use the SYSTEM prompt verbatim (above) — the model was trained on it.
- Greedy decoding (`do_sample=False`, `temperature=0` equivalent). Max ~512
  new tokens.
- Output is **either** the cleaned funding-statement string (verbatim or
  near-verbatim from the input) **or** the literal token `"NONE"` →
  interpret as empty.

## Hard ceiling — read this

The training labels were written by frontier models (Claude / GPT) and are
*not* always verbatim substrings of any source representation:

- ~72% of test gold strings are a verbatim substring of `vlm_markdown`.
- ~28% are not, due to frontier normalization (joined paragraphs, stripped
  LaTeX, normalized whitespace, occasional rephrasing).

This **caps any extractive-or-light-cleanup model around strict tsr@0.95 F1
= 0.73**. The cascade hits 0.725 on strict, essentially at the ceiling.

If you score with `best@0.85` (max of token_sort_ratio, token_set_ratio, and
both directions of partial_ratio, threshold 0.85 — paraphrase-tolerant), you
get F1 = 0.953, which is the practical "did we capture the funding info"
number and is what downstream consumers usually want.

## Failure modes

- **Wrong sibling sentence picked** (~14% of test docs). When an
  acknowledgments section has multiple funding-like sentences ("R.P
  acknowledges..." vs "L.V acknowledges..."), the span head can pick the
  wrong one. The cleanup LoRA cannot recover this — it cleans whatever it's
  handed.
- **Cleanup LoRA over-rewrites** (~3% of test docs). The LoRA occasionally
  prepends an unrelated thanks-sentence from the surrounding context.
  Mitigation: if you have a downstream verifier, prefer the stage-2 rough
  span when stage-3 disagrees substantially.
- **False positives on thanks-only acknowledgments** (~2% of test docs). The
  chunker fires on "we thank Dr. X" sections that mention no money. If you
  need to be very precision-conservative, post-filter: emit `""` if the
  cleaned output contains no funding-indicator words
  (`grant|funded|supported by|NSF|NIH|ERC|...`).

## Memory / latency rough numbers (single H100 80GB)

- Stage 1: ~50ms per 8K-token chunk
- Stage 2: ~50ms per 8K-token chunk
- Stage 3: ~200-500ms per call (~512 output tokens, Qwen3-4B in bf16)

Typical 30-50K-token paper → 6-12 chunks for stage 1, 1-2 for stage 2, 1
stage-3 call. End-to-end ~1-2s per paper.

## When to skip stages

- **Skip stage 1**: just use the last 8K tokens of the paper as the only
  chunk. Stage 2 will still work; you lose ~0.5pt F1 (most funding statements
  are near the end of the paper) but skip the chunker entirely.
- **Skip stage 3**: if you only need approximate localization (good enough to
  feed to a regex/LLM downstream), stage 2 output alone is best@0.85 F1
  0.956. The +0.3pt strict gain from stage 3 may not be worth the 4B-param
  inference cost in your use case.

## Eval / reproduction

Evaluation uses fuzzy-similarity at multiple thresholds. The scoring code
lives in this dataset's training repo, but for reproduction:

```python
from rapidfuzz import fuzz

def best_at_85(gold: str, pred: str) -> bool:
    if not gold and not pred: return True
    if not gold or not pred: return False
    g, p = gold.lower(), pred.lower()
    s = max(
        fuzz.token_sort_ratio(g, p),
        fuzz.token_set_ratio(g, p),
        fuzz.partial_ratio(g, p),
        fuzz.partial_ratio(p, g),
    )
    return s >= 85.0

def strict_at_95(gold: str, pred: str) -> bool:
    if not gold and not pred: return True
    if not gold or not pred: return False
    return fuzz.token_sort_ratio(gold.lower(), pred.lower()) >= 95.0
```

A doc is counted as a match if `metric(gold, pred)` is true. F1 is computed at
the document level over the test split.
