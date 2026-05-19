import time
from typing import List, Optional, Tuple

import torch

from funding_statement_extractor_cascade.cascade.loader import CascadeModels
from funding_statement_extractor_cascade.cascade.types import CascadeConfig, CascadeResult
from funding_statement_extractor_cascade.processing.chunking import build_chunks


def _run_stage1(text: str, models: CascadeModels, cfg: CascadeConfig
                ) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[float, int, int]]]:
    """Returns (ids, offsets, surviving_chunks).

    surviving_chunks = top-k (by prob desc) chunks with sigmoid(logit) >= threshold.
    """
    enc = models.mb_tok(text, add_special_tokens=False,
                         return_offsets_mapping=True, truncation=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]

    chunks = build_chunks(token_count=len(ids), max_tok=cfg.max_tok, stride=cfg.stride)
    chunk_probs: List[Tuple[float, int, int]] = []
    for st, en in chunks:
        c_ids = torch.tensor(ids[st:en]).unsqueeze(0).to(cfg.device)
        c_attn = torch.ones_like(c_ids)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logit = models.chunker(c_ids, c_attn).float()
        chunk_probs.append((torch.sigmoid(logit).item(), st, en))

    chunk_probs.sort(key=lambda x: -x[0])
    surviving = [c for c in chunk_probs[: cfg.chunk_top_k]
                 if c[0] >= cfg.chunk_keep_threshold]
    return ids, offsets, surviving


def _run_stage2(
    text: str,
    surviving: List[Tuple[float, int, int]],
    models: CascadeModels,
    cfg: CascadeConfig,
    *,
    ids: List[int],
    offsets: List[Tuple[int, int]],
) -> Optional[Tuple[float, str, int, int]]:
    """Returns (no_answer_prob, rough_span, char_start, char_end) for the best chunk, or None."""
    best: Optional[Tuple[float, str, int, int]] = None
    for _, st, en in surviving:
        c_offsets = offsets[st:en]
        c_ids = torch.tensor(ids[st:en]).unsqueeze(0).to(cfg.device)
        c_attn = torch.ones_like(c_ids)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            s_log, e_log, na_log = models.spanner(c_ids, c_attn)
        s_log = s_log.squeeze(0).float().cpu()
        e_log = e_log.squeeze(0).float().cpu()
        no_a = torch.sigmoid(na_log).item()
        if no_a >= cfg.span_no_answer_threshold:
            continue
        s = int(s_log.argmax())
        e = s + int(e_log[s : s + cfg.span_max_length].argmax())
        if not (0 <= s <= e < len(c_offsets)):
            continue
        char_s = c_offsets[s][0]
        char_e = c_offsets[e][1]
        span_text = text[char_s:char_e].strip()
        if best is None or no_a < best[0]:
            best = (no_a, span_text, char_s, char_e)
    return best


SYSTEM_CLEAN = (
    "You are a funding statement cleaner. Given a rough extracted funding "
    "statement and its surrounding context from an academic paper, output "
    "the exact funding statement as it should appear in a database. Clean "
    "up LaTeX markers ($^{N}$, \\textsuperscript), hyphenated line breaks, "
    "and abnormal whitespace, but DO NOT paraphrase. If the rough span is "
    "not actually a funding statement, output the single word: NONE"
)


def _run_stage3(text: str, rough_span: str, char_s: int, char_e: int,
                models: CascadeModels, cfg: CascadeConfig) -> str:
    ctx_l = text[max(0, char_s - cfg.cleaner_context_chars): char_s]
    ctx_r = text[char_e: char_e + cfg.cleaner_context_chars]
    user = f"{ctx_l}<ROUGH>{rough_span}</ROUGH>{ctx_r}"
    messages = [
        {"role": "system", "content": SYSTEM_CLEAN},
        {"role": "user", "content": user},
    ]
    prompt_ids = models.qwen_tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(cfg.device)
    out = models.cleaner.generate(
        prompt_ids,
        max_new_tokens=cfg.cleaner_max_new_tokens,
        do_sample=False,
        pad_token_id=models.qwen_tok.eos_token_id,
    )
    decoded = models.qwen_tok.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True).strip()
    if decoded.upper() == "NONE":
        return ""
    return decoded


@torch.no_grad()
def extract_funding(text: str, models: CascadeModels, cfg: CascadeConfig) -> CascadeResult:
    t0 = time.perf_counter()
    try:
        ids, offsets, surviving = _run_stage1(text, models, cfg)
        if not surviving:
            return CascadeResult(
                funding_statement="", rough_span="",
                chunk_prob=None, no_answer_prob=None,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        chunk_prob = max(p for p, _, _ in surviving)
        best = _run_stage2(text, surviving, models, cfg, ids=ids, offsets=offsets)
        if best is None:
            return CascadeResult(
                funding_statement="", rough_span="",
                chunk_prob=chunk_prob, no_answer_prob=None,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        no_a, rough_span, char_s, char_e = best

        if cfg.skip_cleanup:
            funding_statement = rough_span
        else:
            funding_statement = _run_stage3(text, rough_span, char_s, char_e, models, cfg)

        return CascadeResult(
            funding_statement=funding_statement,
            rough_span=rough_span,
            chunk_prob=chunk_prob,
            no_answer_prob=no_a,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
    except Exception as exc:
        return CascadeResult(
            funding_statement="", rough_span="",
            chunk_prob=None, no_answer_prob=None,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"{type(exc).__name__}: {exc}",
        )
