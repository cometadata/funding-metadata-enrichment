"""Train the stage-1 chunk classifier.

ModernBERT-base + mean-pool + 1-d linear head. Predicts P(chunk contains a
funding statement) for an 8192-token chunk of an academic paper's
vlm_markdown.

Trained chunk labels:
    1 if the gold funding statement (located via verbatim substring or
      rapidfuzz.partial_ratio_alignment >= 0.7) overlaps the chunk's
      character range by more than half its length,
    0 otherwise.

Usage:
    python train_chunk_classifier.py \\
        --hf-dataset cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test \\
        --output-dir checkpoints/chunk-classifier \\
        --epochs 3

Pushes the corresponding pretrained adapter to HF as
`cometadata/funding-chunk-classifier-modernbert-base`.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from align_utils import align_stmt


# ----- model -----

class ChunkClassifier(nn.Module):
    """ModernBERT encoder + mean-pool + binary head."""

    def __init__(self, base="answerdotai/ModernBERT-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)  # one logit per chunk


# ----- data -----

def build_examples(df, tokenizer, max_tokens: int, stride: int):
    examples = []
    for _, row in tqdm(df.iterrows(), total=len(df), file=sys.stderr,
                        desc="chunking"):
        stmts = list(row["funding_statements"])
        gold = stmts[0] if stmts else ""
        text = row["vlm_markdown"]
        enc = tokenizer(text, return_offsets_mapping=True,
                        add_special_tokens=False, truncation=False)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        # Locate gold char range in text via verbatim or fuzzy
        if gold:
            a = align_stmt(gold, text)
            cs, ce = (a[0], a[1]) if a is not None else (-1, -1)
        else:
            cs, ce = -1, -1

        if len(ids) <= max_tokens:
            chunk_ranges = [(0, len(ids))]
        else:
            chunk_ranges = []
            for st in range(0, len(ids), stride):
                en = min(st + max_tokens, len(ids))
                chunk_ranges.append((st, en))
                if en == len(ids):
                    break

        for st, en in chunk_ranges:
            label = 0
            if cs >= 0:
                chunk_char_start = offsets[st][0]
                chunk_char_end = offsets[en - 1][1]
                overlap_start = max(cs, chunk_char_start)
                overlap_end = min(ce, chunk_char_end)
                overlap = max(0, overlap_end - overlap_start)
                if overlap > 0.5 * (ce - cs):
                    label = 1
            examples.append({"ids": ids[st:en], "label": label})
    return examples


class ChunkDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch):
    L = max(len(b["ids"]) for b in batch)
    ids = torch.zeros(len(batch), L, dtype=torch.long)
    attn = torch.zeros(len(batch), L, dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.float32)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, :n] = torch.tensor(b["ids"], dtype=torch.long)
        attn[i, :n] = 1
        labels[i] = b["label"]
    return {"input_ids": ids, "attention_mask": attn, "labels": labels}


def warmup_cosine(opt, warmup, total):
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def load_train_df(hf_dataset: str, local_parquet: str) -> pd.DataFrame:
    if local_parquet:
        return pd.read_parquet(local_parquet)
    path = hf_hub_download(repo_id=hf_dataset, filename="train.parquet",
                            repo_type="dataset")
    return pd.read_parquet(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--hf-dataset",
                    default="cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test",
                    help="HF dataset id to pull train.parquet from")
    ap.add_argument("--train-parquet", default=None,
                    help="Local train.parquet (overrides --hf-dataset)")
    ap.add_argument("--output-dir", default="checkpoints/chunk-classifier")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--stride", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    df = load_train_df(args.hf_dataset, args.train_parquet)
    examples = build_examples(df, tokenizer, args.max_tokens, args.stride)
    n_pos = sum(e["label"] for e in examples)
    print(f"chunks: {len(examples)}, positive: {n_pos}, "
          f"negative: {len(examples) - n_pos}")

    ds = ChunkDataset(examples)
    model = ChunkClassifier(args.base_model).to(device)

    pos_weight = torch.tensor(len(examples) / max(n_pos, 1), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    steps_per_epoch = (len(ds) + args.batch_size - 1) // args.batch_size
    opt_steps = (steps_per_epoch + args.grad_accum - 1) // args.grad_accum
    total = opt_steps * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    sched = warmup_cosine(optim, args.warmup_steps, total)

    for epoch in range(args.epochs):
        model.train()
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate,
                             shuffle=True)
        pbar = tqdm(loader, total=steps_per_epoch, desc=f"epoch {epoch}",
                    file=sys.stderr)
        optim.zero_grad()
        elosses = []
        for i, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, attn)
            loss = loss_fn(logits.float(), labels) / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum == 0 or (i + 1) == steps_per_epoch:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
            elosses.append(loss.item() * args.grad_accum)
        pbar.close()
        print(f"epoch {epoch}: mean loss = {sum(elosses)/len(elosses):.4f}")
        ckpt_path = Path(args.output_dir) / f"epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  saved {ckpt_path}")

    # Convenience: save last checkpoint as pytorch_model.bin (HF-style)
    final = Path(args.output_dir) / "pytorch_model.bin"
    torch.save(model.state_dict(), final)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved final weights + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
