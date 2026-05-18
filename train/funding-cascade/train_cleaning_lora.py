"""Train the stage-3 cleanup LoRA.

Qwen3-4B-Instruct-2507 + LoRA (r=32, attn+MLP only — NOT embed/lm_head).
Trained to map a rough extracted funding-statement span + ±400 char context
to the frontier-cleaned canonical form (or `NONE` for negatives).

Training data construction (per train doc):
    - For positives, locate the gold via verbatim or fuzzy alignment in
      vlm_markdown, jitter both endpoints by uniform ±80 chars snapped to
      whitespace (simulates upstream span-tagger bound noise). Two jittered
      variants per gold for augmentation.
    - For negatives, a random ±400-char window inside vlm_markdown with the
      <ROUGH> tags around a random 100-300 char interior span; target = NONE.

Usage:
    python train_cleaning_lora.py \\
        --hf-dataset cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test \\
        --output-dir checkpoints/cleaning-lora \\
        --epochs 3

Pushes as `cometadata/funding-cleaning-qwen3-4b-lora`.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from align_utils import align_stmt


SYSTEM = (
    "You are a funding statement cleaner. Given a rough extracted funding "
    "statement and its surrounding context from an academic paper, output "
    "the exact funding statement as it should appear in a database. Clean "
    "up LaTeX markers ($^{N}$, \\textsuperscript), hyphenated line breaks, "
    "and abnormal whitespace, but DO NOT paraphrase. If the rough span is "
    "not actually a funding statement, output the single word: NONE"
)


def jitter_span_bounds(text: str, cs: int, ce: int, max_chars: int,
                        rng: random.Random):
    """Randomly extend/contract span bounds by up to max_chars; snap to whitespace."""
    def snap_to_ws(p, forward: bool):
        if not forward:
            for i in range(p, max(0, p - 30), -1):
                if i < len(text) and text[i].isspace():
                    return i + 1
            return max(0, p)
        else:
            for i in range(p, min(len(text), p + 30)):
                if text[i].isspace():
                    return i
            return min(len(text), p)
    new_cs = cs + rng.randint(-max_chars, max_chars)
    new_ce = ce + rng.randint(-max_chars, max_chars)
    new_cs = max(0, min(len(text), new_cs))
    new_ce = max(new_cs + 30, min(len(text), new_ce))
    new_cs = snap_to_ws(new_cs, forward=False)
    new_ce = snap_to_ws(new_ce, forward=True)
    return new_cs, new_ce


def build_example(row, rng: random.Random, jitter_chars: int, context_chars: int):
    """Build one (system, user, assistant) chat triple from a train row."""
    vlm = row["vlm_markdown"]
    stmts = list(row["funding_statements"])
    if stmts:
        gold = stmts[0]
        a = align_stmt(gold, vlm)
        if a is None:
            return None
        cs, ce, _ = a
        cs_j, ce_j = jitter_span_bounds(vlm, cs, ce, jitter_chars, rng)
        ctx_start = max(0, cs_j - context_chars)
        ctx_end = min(len(vlm), ce_j + context_chars)
        rel_s = cs_j - ctx_start
        rel_e = ce_j - ctx_start
        ctx = vlm[ctx_start:ctx_end]
        marked = ctx[:rel_s] + "<ROUGH>" + ctx[rel_s:rel_e] + "</ROUGH>" + ctx[rel_e:]
        target = gold
    else:
        # Negative: random window + random rough region
        if len(vlm) < 500:
            return None
        ctx_start = rng.randint(0, max(0, len(vlm) - 800))
        ctx_end = min(len(vlm), ctx_start + 800)
        rel_s = rng.randint(100, max(200, (ctx_end - ctx_start) // 2))
        rel_e = min(ctx_end - ctx_start, rel_s + rng.randint(100, 300))
        ctx = vlm[ctx_start:ctx_end]
        marked = ctx[:rel_s] + "<ROUGH>" + ctx[rel_s:rel_e] + "</ROUGH>" + ctx[rel_e:]
        target = "NONE"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": marked},
            {"role": "assistant", "content": target},
        ]
    }


def load_train_df(hf_dataset: str, local_parquet: str) -> pd.DataFrame:
    if local_parquet:
        return pd.read_parquet(local_parquet)
    path = hf_hub_download(repo_id=hf_dataset, filename="train.parquet",
                            repo_type="dataset")
    return pd.read_parquet(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--hf-dataset",
                    default="cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test")
    ap.add_argument("--train-parquet", default=None)
    ap.add_argument("--output-dir", default="checkpoints/cleaning-lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--jitter-chars", type=int, default=80,
                    help="Max char jitter on positive span endpoints")
    ap.add_argument("--context-chars", type=int, default=400,
                    help="Chars of context on each side of the rough span")
    ap.add_argument("--pos-augment-passes", type=int, default=2,
                    help="How many jittered variants of each positive to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    df = load_train_df(args.hf_dataset, args.train_parquet)
    rng = random.Random(args.seed)
    rows = []
    n_dropped = 0
    for _, r in tqdm(df.iterrows(), total=len(df), file=sys.stderr,
                      desc="building examples"):
        stmts = list(r["funding_statements"])
        is_pos = len(stmts) > 0
        n_passes = args.pos_augment_passes if is_pos else 1
        for _ in range(n_passes):
            ex = build_example(r, rng, args.jitter_chars, args.context_chars)
            if ex is None:
                n_dropped += 1
            else:
                rows.append(ex)
    print(f"built {len(rows)} examples (dropped {n_dropped} unalignable)")
    ds = Dataset.from_list(rows)

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # CRITICAL: LoRA on attn + MLP only — NOT embed_tokens / lm_head.
    # Including those modules in the LoRA targets caused garbage outputs
    # in the merged checkpoint during earlier experiments.
    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, lcfg)

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        max_length=args.max_length,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="wandb" if "WANDB_API_KEY" in os.environ else "none",
        run_name=os.path.basename(args.output_dir),
        gradient_checkpointing=True,
        packing=False,
        completion_only_loss=True,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                          processing_class=tokenizer)
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    print(f"saved LoRA adapter to {final_dir}")

    merged_dir = os.path.join(args.output_dir, "merged")
    print(f"merging LoRA into base for vLLM inference...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"merged checkpoint saved to {merged_dir}")


if __name__ == "__main__":
    main()
