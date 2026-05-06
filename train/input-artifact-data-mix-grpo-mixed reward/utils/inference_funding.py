# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "tinker",
#     "tinker-cookbook",
# ]
# ///
"""Run a trained funding-extraction checkpoint against a test JSONL.

Pure inference: loads a Tinker sampler checkpoint, samples completions
for each `funding_statement` in the input file, parses the JSON output
into structured Funder/Award records, and writes the predictions
locally. No scoring, no gold-label pass-through — that's the eval
script's job, joined by `doi` against the original input file.

Per-example output fields:
    doi                 — identifier copied from the input
    funding_statement   — the input text the model saw
    predicted_text      — raw decoded model output
    predicted_funders   — parsed list of funder dicts
    stop_reason         — "stop" or "length"
    n_output_tokens     — number of tokens generated
    parse_ok            — whether the model output parsed as valid JSON

Example (DGPO v2 best checkpoint, step 135):

    python3 inference_funding.py \\
        --checkpoint=tinker://a10d732b-ee5b-5014-b191-09c35e27b986:train:0/sampler_weights/000135 \\
        --input=arxiv_test.jsonl \\
        --output=/tmp/arxiv_test_predictions.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import re
import time
from pathlib import Path
from typing import List

import tinker

from tinker_cookbook import model_info, renderers, tokenizer_utils

from evaluate_predictions import Funder, _ensure_funder_list


# ---------------------------------------------------------------------------
# Prompt template — must match what the SFT/RL model was trained with.
# Kept inline (instead of imported from funding_reward) to keep this script
# self-contained and avoid pulling in training-only dependencies.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert at extracting structured funding metadata from academic papers. "
    "Given a funding statement, extract all funders and their associated awards. "
    "Return a JSON array of funder objects. Each funder has:\n"
    '- "funder_name": string or null\n'
    '- "awards": array of objects with "award_ids" (array of strings), '
    '"funding_scheme" (array of strings), and "award_title" (array of strings)\n'
    "Return ONLY the JSON array, no other text."
)

USER_TEMPLATE = "Extract funding information from the following statement:\n\n{funding_statement}"

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


# Ligatures commonly produced by PDF extractors
_LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}


def preprocess_statement(text: str) -> str:
    """Moderate preprocessing: undo common PDF parser artifacts.

    Reverses the kinds of corruption introduced by PDF column extraction:
    - rejoin hyphenated line breaks (``word-\\n fix`` -> ``wordfix``)
    - replace remaining newlines/tabs with spaces (column wrap)
    - expand common ligatures (\ufb01 -> fi)
    - collapse multiple whitespace runs into a single space
    """
    if not text:
        return text

    # Rejoin words split across line breaks: "word-\n  fix" -> "wordfix".
    # Only when the next character is a lowercase letter (avoid joining IDs).
    text = re.sub(r"(\w)-\s*\n\s*([a-z])", r"\1\2", text)

    # Replace tabs and newlines with spaces (column wrap)
    text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")

    # Expand ligatures
    for lig, rep in _LIGATURE_MAP.items():
        if lig in text:
            text = text.replace(lig, rep)

    # Collapse multi-whitespace
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def extract_funders_from_text(text: str) -> List[Funder]:
    """Parse model output text into a list of Funder objects.

    Handles: raw JSON list, {"funders": [...]}, markdown code blocks,
    single funder dict. Returns [] on any parse failure.
    """
    if not text or not text.strip():
        return []

    m = _CODE_BLOCK_RE.search(text)
    candidate = m.group(1).strip() if m else text.strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict) and "funders" in parsed:
        return _ensure_funder_list(parsed["funders"])
    return _ensure_funder_list(parsed)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="tinker:// sampler_weights path (e.g. tinker://SESSION:train:0/sampler_weights/000135)",
    )
    p.add_argument(
        "--base-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model id used during training (default: %(default)s)",
    )
    p.add_argument(
        "--renderer",
        default=None,
        help="Renderer name; auto-detected from base model if omitted",
    )
    p.add_argument("--input", required=True, help="Input JSONL path")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument(
        "--summary",
        default=None,
        help="Optional summary JSON path; defaults to <output>.summary.json",
    )
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 = greedy (default); use small positive values if your engine requires it",
    )
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit", type=int, default=None, help="Max examples to run (for smoke tests)"
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max concurrent sample calls to Tinker (default: %(default)s)",
    )
    p.add_argument(
        "--preprocess",
        action="store_true",
        help="Apply moderate preprocessing to undo PDF parser artifacts "
             "(column wrap, ligatures, multi-spacing) before sending to model",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-example inference
# ---------------------------------------------------------------------------
async def run_one(
    sampling_client: tinker.SamplingClient,
    renderer,
    ex: dict,
    sampling_params: tinker.SamplingParams,
    preprocess: bool = False,
) -> dict:
    """Run a single test example and return a result dict.

    The returned dict carries only the raw inputs/outputs plus
    inference-quality signals. Scoring is done by a separate evaluator.

    When ``preprocess`` is True, the funding statement is cleaned with
    :func:`preprocess_statement` before being shown to the model.
    """
    raw_statement = ex["funding_statement"]
    statement = preprocess_statement(raw_statement) if preprocess else raw_statement

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(funding_statement=statement),
        },
    ]
    prompt = renderer.build_generation_prompt(messages)

    resp = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=sampling_params,
    )
    sampled = resp.sequences[0]
    tokens = list(sampled.tokens)
    message, parse_ok = renderer.parse_response(tokens)
    predicted_text = renderers.get_text_content(message)
    predicted_funders = [
        dataclasses.asdict(f) for f in extract_funders_from_text(predicted_text)
    ]

    # Always emit the original statement so eval can join by funding_statement.
    return {
        "doi": ex.get("doi"),
        "funding_statement": raw_statement,
        "preprocessed_statement": statement if preprocess else None,
        "predicted_text": predicted_text,
        "predicted_funders": predicted_funders,
        "stop_reason": sampled.stop_reason,
        "n_output_tokens": len(tokens),
        "parse_ok": bool(parse_ok),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = (
        Path(args.summary)
        if args.summary
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    renderer_name = args.renderer or model_info.get_recommended_renderer_name(args.base_model)
    tokenizer = tokenizer_utils.get_tokenizer(args.base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Using renderer: %s", renderer_name)

    logger.info("Loading examples from %s", input_path)
    with open(input_path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    if args.limit is not None:
        examples = examples[: args.limit]
    logger.info("Loaded %d examples", len(examples))

    logger.info("Connecting to Tinker, loading sampler at %s", args.checkpoint)
    service = tinker.ServiceClient()
    sampling_client = await service.create_sampling_client_async(
        model_path=args.checkpoint,
        base_model=args.base_model,
    )

    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        stop=renderer.get_stop_sequences(),
    )
    logger.info(
        "Sampling params: T=%.2f top_p=%.2f top_k=%d max_tokens=%d seed=%d",
        args.temperature,
        args.top_p,
        args.top_k,
        args.max_tokens,
        args.seed,
    )

    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict | None] = [None] * len(examples)

    if args.preprocess:
        logger.info("Preprocessing enabled: undoing PDF parser artifacts before inference")

    async def run_with_progress(i: int, ex: dict) -> None:
        async with sem:
            try:
                results[i] = await run_one(
                    sampling_client, renderer, ex, sampling_params,
                    preprocess=args.preprocess,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error on example %d (doi=%s): %s", i, ex.get("doi"), exc)
                results[i] = {
                    "doi": ex.get("doi"),
                    "funding_statement": ex.get("funding_statement"),
                    "error": str(exc),
                }
            done = sum(1 for x in results if x is not None)
            if done % 10 == 0 or done == len(examples):
                logger.info("  %d/%d processed", done, len(examples))

    t_start = time.monotonic()
    await asyncio.gather(
        *(run_with_progress(i, ex) for i, ex in enumerate(examples))
    )
    wall_time = time.monotonic() - t_start

    # Write per-example output
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d predictions to %s", len(results), output_path)

    # Inference-quality summary (no scoring — that lives in evaluate_predictions.py).
    valid = [r for r in results if r and "error" not in r]
    n_errors = len(results) - len(valid)

    stop_reason_counts: dict[str, int] = {}
    parse_ok_count = 0
    output_tokens: list[int] = []
    for r in valid:
        sr = r.get("stop_reason") or "unknown"
        stop_reason_counts[sr] = stop_reason_counts.get(sr, 0) + 1
        if r.get("parse_ok"):
            parse_ok_count += 1
        if "n_output_tokens" in r:
            output_tokens.append(r["n_output_tokens"])

    def mean(vs: list[float]) -> float:
        return sum(vs) / len(vs) if vs else 0.0

    summary = {
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "input": str(input_path),
        "n_examples": len(results),
        "n_valid": len(valid),
        "n_errors": n_errors,
        "parse_ok_rate": parse_ok_count / max(len(valid), 1),
        "stop_reason_counts": stop_reason_counts,
        "avg_output_tokens": mean(output_tokens),
        "max_output_tokens_observed": max(output_tokens, default=0),
        "total_wall_time_sec": round(wall_time, 3),
        "throughput_examples_per_sec": round(len(valid) / wall_time, 3) if wall_time > 0 else 0.0,
        "sampling_params": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 62)
    print(f"Inference summary  ({len(valid)}/{len(results)} valid, {n_errors} errors)")
    print("=" * 62)
    print(f"  parse_ok_rate           : {summary['parse_ok_rate']:.4f}")
    print(f"  stop_reason_counts      : {stop_reason_counts}")
    print(f"  avg_output_tokens       : {summary['avg_output_tokens']:.1f}")
    print(f"  max_output_tokens       : {summary['max_output_tokens_observed']}")
    print(f"  total_wall_time_sec     : {summary['total_wall_time_sec']:.1f}")
    print(f"  throughput (ex/s)       : {summary['throughput_examples_per_sec']:.2f}")
    print()
    print(f"  predictions: {output_path}")
    print(f"  summary    : {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
