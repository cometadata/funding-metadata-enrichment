"""Split train.jsonl + synthetic.jsonl into SFT / RL / eval by unique ID.

Every row sharing the same `id` field lands in exactly one split.
Outputs separate train and synthetic files per split so that downstream
scripts can apply their own upsampling (e.g. SFT upsamples synthetic 2x).

Default split ratios: 70% SFT, 20% RL-train, 10% RL-eval.

Outputs:
    sft_train.jsonl       — real train rows for SFT IDs
    sft_synthetic.jsonl   — synthetic rows for SFT IDs
    rl_train.jsonl        — real train rows for RL IDs
    rl_synthetic.jsonl    — synthetic rows for RL IDs
    rl_eval_train.jsonl   — real train rows for eval IDs
    rl_eval_synthetic.jsonl — synthetic rows for eval IDs
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _load_by_id(path: str) -> dict[str, list[dict]]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_id[row["id"]].append(row)
    return by_id


def _write_jsonl(rows: list[dict], path: Path) -> int:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="train.jsonl")
    parser.add_argument("--synthetic", default="synthetic.jsonl")
    parser.add_argument("--sft-frac", type=float, default=0.70)
    parser.add_argument("--rl-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    eval_frac = 1.0 - args.sft_frac - args.rl_frac
    assert eval_frac > 0, f"sft_frac + rl_frac must be < 1.0, got {args.sft_frac + args.rl_frac}"

    train_by_id = _load_by_id(args.train)
    synth_by_id = _load_by_id(args.synthetic)

    all_ids = sorted(set(train_by_id.keys()) | set(synth_by_id.keys()))
    rng = random.Random(args.seed)
    rng.shuffle(all_ids)

    n = len(all_ids)
    sft_end = int(n * args.sft_frac)
    rl_end = sft_end + int(n * args.rl_frac)

    split_ids = {
        "sft": all_ids[:sft_end],
        "rl": all_ids[sft_end:rl_end],
        "rl_eval": all_ids[rl_end:],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ids in split_ids.items():
        train_rows = [row for id_ in ids for row in train_by_id.get(id_, [])]
        synth_rows = [row for id_ in ids for row in synth_by_id.get(id_, [])]

        prefix = split_name
        t_path = out_dir / f"{prefix}_train.jsonl"
        s_path = out_dir / f"{prefix}_synthetic.jsonl"
        nt = _write_jsonl(train_rows, t_path)
        ns = _write_jsonl(synth_rows, s_path)

        print(f"{split_name}: {len(ids)} IDs → {nt} train rows ({t_path.name}), {ns} synthetic rows ({s_path.name})")

    print(f"\nTotal: {n} IDs split into {args.sft_frac:.0%} / {args.rl_frac:.0%} / {eval_frac:.0%}")


if __name__ == "__main__":
    main()
