"""Train the stage-2 span-extraction head.

ModernBERT-base + linear start head + linear end head + linear no-answer head.
Predicts (start_token, end_token) of the funding statement within an
8192-token chunk, plus a no-answer logit indicating whether the chunk
contains a funding statement at all.

The training chunk for each positive doc is the 8192-token sliding window
(stride 4096) that contains the gold funding statement; the gold is located
via verbatim substring or `rapidfuzz.partial_ratio_alignment >= 0.7`. For
negative docs the training chunk is the last 8192-token window and the
no-answer label is 1.

Usage:
    python train_span_head.py \\
        --hf-dataset cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test \\
        --output-dir checkpoints/span-head \\
        --epochs 4

Pushes as `cometadata/funding-extraction-modernbert-base-spanhead`.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from align_utils import align_stmt


# ----- model -----

class SpanHead(nn.Module):
    """ModernBERT encoder + start/end/no-answer heads."""

    def __init__(self, base="answerdotai/ModernBERT-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        h = self.encoder.config.hidden_size
        self.start_head = nn.Linear(h, 1)
        self.end_head = nn.Linear(h, 1)
        self.no_answer_head = nn.Linear(h, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(out.last_hidden_state)
        start_logits = self.start_head(hidden).squeeze(-1)
        end_logits = self.end_head(hidden).squeeze(-1)
        # Mean-pool for no-answer
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        no_answer = self.no_answer_head(pooled).squeeze(-1)
        return start_logits, end_logits, no_answer


# ----- data -----

def find_best_chunk(text, gold, tokenizer, max_tokens, stride):
    """Pick the chunk that best contains the gold span.

    Returns (chunk_ids, chunk_offsets, gold_start_tok, gold_end_tok, has_gold).
    For negatives, takes the last chunk and returns (-1, -1, False).
    """
    enc = tokenizer(text, return_offsets_mapping=True,
                    add_special_tokens=False, truncation=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    gold_start_tok = gold_end_tok = -1
    if gold:
        a = align_stmt(gold, text)
        if a is not None:
            cs, ce, _ = a
            for i, (s, e) in enumerate(offsets):
                if e > cs and s < ce:
                    if gold_start_tok == -1:
                        gold_start_tok = i
                    gold_end_tok = i

    if len(ids) <= max_tokens:
        chunk_ranges = [(0, len(ids))]
    else:
        chunk_ranges = []
        for st in range(0, len(ids), stride):
            en = min(st + max_tokens, len(ids))
            chunk_ranges.append((st, en))
            if en == len(ids):
                break

    if gold_start_tok < 0:
        # Negative: use the last chunk
        st, en = chunk_ranges[-1]
        return ids[st:en], offsets[st:en], -1, -1, False

    # Positive: prefer chunk fully containing the gold
    for st, en in chunk_ranges:
        if gold_start_tok >= st and gold_end_tok < en:
            return (ids[st:en], offsets[st:en],
                    gold_start_tok - st, gold_end_tok - st, True)
    # Fallback: pick chunk with max overlap
    best = max(chunk_ranges,
               key=lambda r: min(r[1], gold_end_tok + 1) - max(r[0], gold_start_tok))
    st, en = best
    s = max(0, gold_start_tok - st)
    e = min(en - st - 1, gold_end_tok - st)
    return ids[st:en], offsets[st:en], s, e, True


class SpanDataset(Dataset):
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
    starts = torch.zeros(len(batch), dtype=torch.long)
    ends = torch.zeros(len(batch), dtype=torch.long)
    no_answer = torch.zeros(len(batch), dtype=torch.float32)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, :n] = torch.tensor(b["ids"], dtype=torch.long)
        attn[i, :n] = 1
        starts[i] = b["start"] if b["start"] >= 0 else 0
        ends[i] = b["end"] if b["end"] >= 0 else 0
        no_answer[i] = 1.0 if b["start"] < 0 else 0.0
    return {"input_ids": ids, "attention_mask": attn, "starts": starts,
            "ends": ends, "no_answer": no_answer}


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
                    default="cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test")
    ap.add_argument("--train-parquet", default=None)
    ap.add_argument("--output-dir", default="checkpoints/span-head")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--stride", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--no-answer-loss-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    df = load_train_df(args.hf_dataset, args.train_parquet)
    print(f"building span examples for {len(df)} docs...")
    examples = []
    n_pos = n_neg = n_unaligned = 0
    for _, row in tqdm(df.iterrows(), total=len(df), file=sys.stderr):
        stmts = list(row["funding_statements"])
        gold = stmts[0] if stmts else ""
        ids, offsets, s, e, has_g = find_best_chunk(
            row["vlm_markdown"], gold, tokenizer, args.max_tokens, args.stride,
        )
        if gold and not has_g:
            n_unaligned += 1
            continue
        if has_g:
            n_pos += 1
        else:
            n_neg += 1
        examples.append({"ids": ids, "start": s, "end": e})
    print(f"  pos={n_pos} neg={n_neg} unaligned_dropped={n_unaligned} "
          f"-> {len(examples)} examples")

    ds = SpanDataset(examples)
    model = SpanHead(args.base_model).to(device)

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
            starts = batch["starts"].to(device)
            ends = batch["ends"].to(device)
            no_a = batch["no_answer"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                slog, elog, nalog = model(ids, attn)
            mask = attn.bool()
            slog = slog.masked_fill(~mask, -1e4)
            elog = elog.masked_fill(~mask, -1e4)
            ans_mask = (no_a == 0)
            if ans_mask.any():
                start_loss = F.cross_entropy(slog[ans_mask].float(),
                                              starts[ans_mask], reduction="mean")
                end_loss = F.cross_entropy(elog[ans_mask].float(),
                                            ends[ans_mask], reduction="mean")
            else:
                start_loss = end_loss = 0.0 * slog.sum()
            no_answer_loss = F.binary_cross_entropy_with_logits(nalog.float(), no_a)
            loss = (start_loss + end_loss
                    + args.no_answer_loss_weight * no_answer_loss) / args.grad_accum
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

    final = Path(args.output_dir) / "pytorch_model.bin"
    torch.save(model.state_dict(), final)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved final weights + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
