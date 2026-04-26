"""CLI entry point: `funding-extract serve` and `funding-extract run`."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

from .serve import ServeConfig, run_serve

# run_extraction is defined in this module (Task 11 fills it in); reference it
# here so tests can monkeypatch it cleanly.
try:
    from .run_driver import run_extraction, RunConfig  # noqa: F401  (filled in Task 11)
except ImportError:
    run_extraction = None  # type: ignore[assignment]
    RunConfig = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class _StripDoubleDashAction(argparse.Action):
    """argparse REMAINDER preserves a leading '--' separator; strip it."""

    def __call__(self, parser, namespace, values, option_string=None):
        if values and values[0] == "--":
            values = values[1:]
        setattr(namespace, self.dest, values)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="funding-extract",
        description="Online vLLM extraction utility for funding statements.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # serve
    s = sub.add_parser("serve", help="Run vLLM OpenAI-compatible server with the LoRA preloaded.")
    s.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    s.add_argument("--lora-repo", default="cometadata/funding-extraction-llama-3.1-8b-instruct-artifact-data-mix-grpo-mixed-reward")
    s.add_argument("--served-name", default="funding-extraction")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    s.add_argument("--max-model-len", type=int, default=4096)
    s.add_argument("--tensor-parallel-size", type=int, default=1)
    s.add_argument("--max-lora-rank", type=int, default=64)
    s.add_argument("--max-loras", type=int, default=1)
    s.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        action=_StripDoubleDashAction,
        help="Extra args passed to `vllm serve` after `--`.",
    )

    # run
    r = sub.add_parser("run", help="Run async extraction over a parquet file against a vLLM server.")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--text-field", default="predicted_statements")
    r.add_argument(
        "--row-id-fields",
        default="arxiv_id,row_idx",
        help="Comma-separated columns forming the resume key (must be present in input).",
    )
    r.add_argument("--vllm-url", default="http://localhost:8000")
    r.add_argument("--served-name", default="funding-extraction")
    r.add_argument("--concurrency", type=int, default=256)
    r.add_argument("--write-batch-size", type=int, default=256)
    r.add_argument("--request-timeout", type=float, default=60.0)
    r.add_argument("--max-retries", type=int, default=3)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--top-p", type=float, default=1.0)
    r.add_argument("--max-tokens", type=int, default=2048)
    r.add_argument("--max-input-chars", type=int, default=8000,
                   help="Skip statements longer than this with InputTooLong error (no HTTP call). 0 disables.")
    r.add_argument("--no-resume", action="store_true")
    r.add_argument("--log-every", type=int, default=50)

    return p


def _serve_cfg_from_args(args: argparse.Namespace) -> ServeConfig:
    passthrough = list(args.passthrough or [])
    # argparse REMAINDER preserves the leading '--'; strip it.
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return ServeConfig(
        base_model=args.base_model,
        lora_repo=args.lora_repo,
        served_name=args.served_name,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        max_lora_rank=args.max_lora_rank,
        max_loras=args.max_loras,
        passthrough=passthrough,
    )


@dataclass
class _RunCfg:
    """Minimal run config used by `cli.main` for dispatch.

    The full RunConfig with extraction-loop fields lives in run_driver.py
    (added in Task 11). This shim lets cli tests pass before run_driver
    exists.
    """
    input: str
    output: str
    text_field: str
    row_id_fields: list[str]
    vllm_url: str
    served_name: str
    concurrency: int
    write_batch_size: int
    request_timeout: float
    max_retries: int
    temperature: float
    top_p: float
    max_tokens: int
    no_resume: bool
    log_every: int
    max_input_chars: int | None = None


def _run_cfg_from_args(args: argparse.Namespace):
    # Use RunConfig if Task 11 has populated it; else the shim above.
    cls = RunConfig if RunConfig is not None else _RunCfg
    return cls(
        input=args.input,
        output=args.output,
        text_field=args.text_field,
        row_id_fields=[s.strip() for s in args.row_id_fields.split(",") if s.strip()],
        vllm_url=args.vllm_url,
        served_name=args.served_name,
        concurrency=args.concurrency,
        write_batch_size=args.write_batch_size,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        no_resume=args.no_resume,
        log_every=args.log_every,
        max_input_chars=args.max_input_chars if args.max_input_chars > 0 else None,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        return run_serve(_serve_cfg_from_args(args))
    if args.command == "run":
        if run_extraction is None:
            logger.error("run_extraction is not available — run_driver.py missing.")
            return 1
        return run_extraction(_run_cfg_from_args(args))

    parser.error(f"unknown command: {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
