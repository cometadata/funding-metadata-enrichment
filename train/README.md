# Funding Parsing LoRA — Training Code

Training pipeline for [cometadata/funding-parsing-lora-Llama_3.1_8B-instruct-ep2-r128-a32-grpo](https://huggingface.co/cometadata/funding-parsing-lora-Llama_3.1_8B-instruct-ep2-r128-a32-grpo), a LoRA adapter that extracts structured funding information from funding statements.

Training data (real + synthetic funding statements) is available at [cometadata/funding-extraction-sft-data](https://huggingface.co/datasets/cometadata/funding-extraction-sft-data).


## Requirements

All scripts use [PEP 723](https://peps.python.org/pep-0723/) inline metadata and can be run directly with a compatible tool (e.g., `uv run`). The main dependencies are:

- [tinker](https://tinker.dev/) — GPU training/inference via Tinker API
- [tinker-cookbook](https://github.com/thinkingmachines/tinker-cookbook) — Training recipes and utilities
- [chz](https://github.com/openai/chz) — CLI argument parsing

You need a `TINKER_API_KEY` environment variable set. Training runs locally on CPU but GPU work runs remotely on Tinker's platform.

## Files

| File | Description |
|------|-------------|
| `sft_funding_llama.py` | Stage 1: Supervised fine-tuning on Llama 3.1 8B Instruct |
| `rl_funding.py` | Stage 2: GRPO reinforcement learning from SFT checkpoint |
| `funding_reward.py` | RL environment, reward function, and system/user prompts |
| `evaluate_predictions.py` | Hierarchical matching metrics (Hungarian algorithm) |
| `export_checkpoint.py` | Export best Tinker checkpoint to HuggingFace as PEFT adapter |
| `train.jsonl` | 1,316 real funding statements with DOIs ([HF dataset](https://huggingface.co/datasets/cometadata/funding-extraction-sft-data)) |
| `synthetic.jsonl` | 2,531 synthetic funding statements ([HF dataset](https://huggingface.co/datasets/cometadata/funding-extraction-sft-data)) |

## Usage

### Stage 1: SFT

```bash
nohup python3 sft_funding_llama.py \
    --log_path=/tmp/tinker-funding-sft-llama/run-name \
    --behavior_if_log_dir_exists=delete \
    > /tmp/tinker-funding-sft-llama/run-name.log 2>&1 &
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `meta-llama/Llama-3.1-8B-Instruct` | Base model |
| `--lora_rank` | 128 | LoRA rank |
| `--num_epochs` | 2 | Training epochs |
| `--synthetic_upsample` | 2 | Synthetic data duplication factor |
| `--batch_size` | 128 | Batch size |
| `--max_length` | 4096 | Max sequence length |
| `--learning_rate` | 0.0 (auto) | 0 = auto from Tinker's model-specific formula |

### Stage 2: GRPO

```bash
nohup python3 rl_funding.py \
    --load_checkpoint_path="tinker://<session>/weights/final" \
    --log_path=/tmp/tinker-funding-rl/run-name \
    --behavior_if_log_dir_exists=delete \
    > /tmp/tinker-funding-rl/run-name.log 2>&1 &
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning_rate` | 3e-5 | RL learning rate |
| `--temperature` | 0.8 | Sampling temperature |
| `--batch_size` | 16 | Batch size |
| `--group_size` | 8 | Rollouts per prompt |
| `--kl_penalty_coef` | 0.03 | KL penalty vs SFT reference |
| `--load_checkpoint_path` | — | SFT checkpoint to start from |

### Export best checkpoint

```bash
python3 export_checkpoint.py \
    --checkpoints-file /tmp/tinker-funding-rl/run-name/checkpoints.jsonl \
    --best \
    --base-model "meta-llama/Llama-3.1-8B-Instruct" \
    --repo-prefix "cometadata/funding-extraction"
```

## Monitoring Runs

```bash
# Check if process is alive
ps -p $PID

# Tail raw logs
tail -f /tmp/tinker-funding-rl/run-name.log

# Count completed steps
wc -l /tmp/tinker-funding-rl/run-name/metrics.jsonl

# Formatted metrics
cat /tmp/tinker-funding-rl/run-name/metrics.jsonl | python3 -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l)
    step = d.get('step', '?')
    rw = d.get('env/all/reward/total', 0)
    erw = d.get('test/env/all/reward/total', '')
    kl = d.get('kl_policy_base', 0)
    done = d.get('progress/done_frac', 0)
    erw_s = f'{erw:.3f}' if isinstance(erw, (int, float)) else '—'
    print(f'step={step:>3}  reward={rw:.3f}  eval={erw_s:>7}  kl={kl:.4f}  done={done:.1%}')
"
```

## Reward Function

The reward uses gated, hierarchical matching with the Hungarian algorithm for 1:1 funder pairing of the sub-fields:

- Funder name: Weight = 0.50, Fuzzy matching with token-sort, acronym, and containment boosts; F0.5 score
- Award IDs: Weight = 0.50, Normalized exact matching; F0.5 score
- Funding scheme: Not weighted
- Award title: Not weighted

## Published Results

| Step | Eval Reward | Notes |
|------|-------------|-------|
| 0 | 0.945 | SFT baseline |
| 30 | 0.958 | |
| **70** | **0.959** | **Best checkpoint (published LoRA)** |
| 217 | 0.950 | Final step |
