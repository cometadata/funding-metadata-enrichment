# Funding statement extraction cascade — training code

Training scripts for the three-stage cascade that extracts and cleans
funding statements from arXiv PDFs. Each stage corresponds to one published
model:

| Stage | Script | HF model |
|---|---|---|
| 1 — chunk classifier | `train_chunk_classifier.py` | [`cometadata/funding-chunk-classifier-modernbert-base`](https://huggingface.co/cometadata/funding-chunk-classifier-modernbert-base) |
| 2 — span head | `train_span_head.py` | [`cometadata/funding-extraction-modernbert-base-spanhead`](https://huggingface.co/cometadata/funding-extraction-modernbert-base-spanhead) |
| 3 — cleaning LoRA | `train_cleaning_lora.py` | [`cometadata/funding-cleaning-qwen3-4b-lora`](https://huggingface.co/cometadata/funding-cleaning-qwen3-4b-lora) |

See [`CASCADE_USAGE.md`](./CASCADE_USAGE.md) for the full inference pipeline,
load-and-run code, and benchmark numbers.

## Setup

```bash
pip install "torch>=2.4" "transformers>=4.45,<6" "trl>=0.29" "peft>=0.14" \
            "accelerate" "datasets" "pandas" "rapidfuzz" \
            "huggingface_hub>=0.26" "tqdm" "wandb"
```

Each script pulls the training data from
[`cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test`](https://huggingface.co/datasets/cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test)
automatically (or accepts `--train-parquet /local/path.parquet` to use a
cached copy).

Set `HF_TOKEN` and (optionally) `WANDB_API_KEY` env vars before training.

## Stage 1 — chunk classifier

Single 1× H100, ~30 min:

```bash
python train_chunk_classifier.py \
    --output-dir checkpoints/chunk-classifier \
    --epochs 3 --batch-size 2 --grad-accum 8
```

## Stage 2 — span head

Single 1× H100, ~45 min:

```bash
python train_span_head.py \
    --output-dir checkpoints/span-head \
    --epochs 4 --batch-size 4 --grad-accum 4
```

## Stage 3 — cleaning LoRA

Single 1× H100, ~1 h (Qwen3-4B + LoRA, bf16, gradient checkpointing):

```bash
python train_cleaning_lora.py \
    --output-dir checkpoints/cleaning-lora \
    --epochs 3 --batch-size 2 --grad-accum 8
```

Each script's `--help` lists all hyperparameters.

## Output layout

Each script writes:

- `epoch{N}.pt` per epoch (state_dict for stages 1-2, TRL `SFTTrainer`
  checkpoints for stage 3)
- A final `pytorch_model.bin` (stages 1-2) or `final/` + `merged/` directories
  (stage 3)
- The tokenizer config

For stages 1-2, the saved state dicts load via the model classes defined at
the top of each script (also bundled as `modeling.py` in the HF repos). For
stage 3, `final/` is the LoRA adapter and `merged/` is the merged
full-precision model that vLLM can serve.

## Notes on training-data construction

All three scripts use `align_utils.align_stmt` (verbatim substring first,
then anchor-based fuzzy alignment via `rapidfuzz.partial_ratio_alignment`) to
locate the gold funding statement inside `vlm_markdown`. About 28% of gold
strings are not verbatim substrings (frontier labelers normalize whitespace,
strip LaTeX markers, occasionally rephrase), so the fuzzy fallback is
essential — without it ~28% of positive training rows would be dropped.

The cleaning LoRA augments each positive example with two jittered variants
(`±80` char endpoint noise, snapped to whitespace) to make the model robust
to upstream span-tagger boundary errors.

## Reproducibility

All scripts accept `--seed` (default 42). Hyperparameters in the per-script
docstrings match the ones used to train the published models.
