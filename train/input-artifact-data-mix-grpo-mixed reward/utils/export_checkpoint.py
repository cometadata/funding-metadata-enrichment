# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "tinker",
#     "transformers>=4.45",
#     "huggingface-hub>=0.25",
# ]
# ///
"""Export a Tinker RL checkpoint to HuggingFace as a PEFT LoRA adapter.

Downloads the adapter from Tinker, patches adapter_config with the base model,
adds the tokenizer, and uploads to HuggingFace Hub.

Usage:
    python export_checkpoint.py --checkpoints-file /tmp/tinker-funding-rl/grpo-v1/checkpoints.jsonl
    python export_checkpoint.py --tinker-path tinker://run-id/sampler_weights/000050
    python export_checkpoint.py --checkpoints-file /path/to/checkpoints.jsonl --name 000100
    python export_checkpoint.py --tinker-path tinker://... --repo-prefix cometadata/funding-extraction
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer


def load_checkpoint_path(checkpoints_file: str, name: str | None) -> str:
    """Read a checkpoints.jsonl and return the sampler_path for the given name (or last)."""
    records = []
    with open(checkpoints_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No checkpoints found in {checkpoints_file}")
    if name:
        for r in records:
            if r["name"] == name:
                return r["sampler_path"]
        raise ValueError(f"Checkpoint '{name}' not found. Available: {[r['name'] for r in records]}")
    return records[-1]["sampler_path"]


def download_adapter(tinker_path: str, output_dir: Path) -> Path:
    """Download adapter from Tinker and return the extracted directory."""
    import subprocess

    result = subprocess.run(
        ["tinker", "checkpoint", "download", tinker_path, "--output", str(output_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tinker checkpoint download failed:\n{result.stderr}")

    # Find the extracted subdirectory containing adapter_model.safetensors
    for p in output_dir.rglob("adapter_model.safetensors"):
        return p.parent
    raise FileNotFoundError(f"No adapter_model.safetensors found in {output_dir}")


def patch_adapter_config(adapter_dir: Path, base_model: str):
    """Set base_model_name_or_path in adapter_config.json if missing."""
    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text())
    if not config.get("base_model_name_or_path"):
        config["base_model_name_or_path"] = base_model
        config_path.write_text(json.dumps(config, indent=2) + "\n")


def add_tokenizer(adapter_dir: Path, base_model: str):
    """Save the base model's tokenizer into the adapter directory."""
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.save_pretrained(str(adapter_dir))


def build_repo_id(repo_prefix: str, base_model: str, step_name: str) -> str:
    """Build a HuggingFace repo ID from components."""
    model_short = base_model.split("/")[-1].lower()
    return f"{repo_prefix}-{model_short}-grpo-step{step_name}"


def find_best_checkpoint(metrics_file: str, checkpoints_file: str, metric: str = "test/env/all/reward/total") -> str:
    """Find the checkpoint with the highest eval metric."""
    # Load metrics
    metrics_by_step = {}
    with open(metrics_file) as f:
        for line in f:
            m = json.loads(line)
            step = m.get("step")
            val = m.get(metric)
            if step is not None and val is not None:
                metrics_by_step[step] = val

    if not metrics_by_step:
        raise ValueError(f"No metric '{metric}' found in {metrics_file}")

    # Load checkpoints with sampler_path
    checkpoints = []
    with open(checkpoints_file) as f:
        for line in f:
            r = json.loads(line)
            if r.get("sampler_path") and not r.get("rolling"):
                checkpoints.append(r)

    if not checkpoints:
        raise ValueError(f"No exportable checkpoints in {checkpoints_file}")

    # Match checkpoints to metrics (checkpoint at step N reflects training up to step N)
    best_name = None
    best_val = -float("inf")
    best_path = None
    for ckpt in checkpoints:
        # Find the closest eval step <= checkpoint batch
        batch = ckpt.get("batch", 0)
        candidates = [(s, v) for s, v in metrics_by_step.items() if s <= batch]
        if candidates:
            step, val = max(candidates, key=lambda x: x[1])
            if val > best_val:
                best_val = val
                best_name = ckpt["name"]
                best_path = ckpt["sampler_path"]

    if best_path is None:
        raise ValueError("Could not match any checkpoint to eval metrics")

    print(f"Best checkpoint: {best_name} ({metric}={best_val:.4f})")
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Export Tinker checkpoint to HuggingFace")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tinker-path", help="Direct tinker:// sampler_weights path")
    source.add_argument("--checkpoints-file", help="Path to checkpoints.jsonl")
    parser.add_argument("--name", help="Checkpoint name (default: latest)", default=None)
    parser.add_argument("--best", action="store_true", help="Export the best checkpoint by eval reward")
    parser.add_argument("--best-metric", default="test/env/all/reward/total", help="Metric to rank checkpoints by")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B", help="Base model for tokenizer and adapter config")
    parser.add_argument("--repo-prefix", default="adambuttrick/funding-extraction", help="HF repo prefix")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    parser.add_argument("--local-only", action="store_true", help="Download and prepare but don't upload")
    parser.add_argument("--output-dir", help="Local output directory (default: temp dir)")
    args = parser.parse_args()

    # Resolve tinker path
    if args.tinker_path:
        tinker_path = args.tinker_path
        step_name = tinker_path.rstrip("/").split("/")[-1]
    elif args.best:
        metrics_file = str(Path(args.checkpoints_file).parent / "metrics.jsonl")
        tinker_path = find_best_checkpoint(metrics_file, args.checkpoints_file, args.best_metric)
        step_name = "best"
    else:
        tinker_path = load_checkpoint_path(args.checkpoints_file, args.name)
        step_name = args.name or tinker_path.rstrip("/").split("/")[-1]

    print(f"Source: {tinker_path}")

    # Download
    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="tinker-export-"))
    print(f"Downloading adapter to {output_dir}...")
    adapter_dir = download_adapter(tinker_path, output_dir)
    print(f"Adapter at: {adapter_dir}")

    # Patch adapter config
    patch_adapter_config(adapter_dir, args.base_model)

    # Add tokenizer
    print(f"Adding tokenizer from {args.base_model}...")
    add_tokenizer(adapter_dir, args.base_model)

    # List final contents
    files = sorted(p.name for p in adapter_dir.iterdir() if p.is_file())
    print(f"Files: {files}")

    if args.local_only:
        print(f"Local export complete: {adapter_dir}")
        return

    # Upload
    repo_id = build_repo_id(args.repo_prefix, args.base_model, step_name)
    print(f"Uploading to {repo_id}...")
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(adapter_dir), repo_id=repo_id)
    print(f"Done: https://huggingface.co/{repo_id}")

    # Clean up temp dir if we created one
    if not args.output_dir:
        shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
