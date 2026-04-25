"""Local orchestrator for ArXiv funding-statement extraction HF Jobs.

Manages a pool of concurrent A100 jobs over the 17.5k-file
`cometadata/arxiv-latex-extract-full-text/results-2026-04-24/` corpus
with manifest-backed resume, retry, and EMA-based rebalancing.
"""
from __future__ import annotations

import base64
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi


_DONE_RE = re.compile(
    r"\[done file=(?P<file>\S+) rows=(?P<rows>\d+) elapsed_s=(?P<elapsed>[\d.]+)\]"
)


@dataclass
class ManifestRow:
    input_file: str
    size_bytes: int
    est_seconds: float
    status: str  # pending | assigned | done | failed
    attempts: int
    job_id: Optional[str]
    assigned_at: Optional[float]
    completed_at: Optional[float]
    output_path: Optional[str]
    last_error: Optional[str]
    worker_elapsed_s: Optional[float]
    row_count: Optional[int] = None  # filled when worker reports done


Manifest = List[ManifestRow]


def write_manifest(rows: Manifest, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        tbl = pa.Table.from_pylist([asdict(r) for r in rows])
    else:
        # Build empty table with the right schema so reads round-trip cleanly.
        schema = pa.schema([
            ("input_file", pa.string()),
            ("size_bytes", pa.int64()),
            ("est_seconds", pa.float64()),
            ("status", pa.string()),
            ("attempts", pa.int64()),
            ("job_id", pa.string()),
            ("assigned_at", pa.float64()),
            ("completed_at", pa.float64()),
            ("output_path", pa.string()),
            ("last_error", pa.string()),
            ("worker_elapsed_s", pa.float64()),
            ("row_count", pa.int64()),
        ])
        tbl = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(tbl, tmp)
    os.replace(tmp, path)


def read_manifest(path: Path) -> Manifest:
    tbl = pq.read_table(str(path))
    return [ManifestRow(**r) for r in tbl.to_pylist()]


def pick_next_batch(rows, *, target_seconds, max_files):
    pending = [r for r in rows if r.status == "pending"]
    pending.sort(key=lambda r: r.est_seconds, reverse=True)
    batch: list = []
    total = 0.0
    for r in pending:
        if not batch:
            batch.append(r)
            total += r.est_seconds
            continue
        if total + r.est_seconds > target_seconds:
            break
        if len(batch) >= max_files:
            break
        batch.append(r)
        total += r.est_seconds
    return batch


def update_ema(prev, sample, alpha):
    if prev is None:
        return sample
    return (1 - alpha) * prev + alpha * sample


def parse_done_line(line):
    m = _DONE_RE.search(line)
    if not m:
        return None
    return {
        "file": m.group("file"),
        "rows": int(m.group("rows")),
        "elapsed_s": float(m.group("elapsed")),
    }


def read_row_count_local(path):
    """Footer-only row-count read of a local parquet (diagnostic helper)."""
    return pq.ParquetFile(str(path)).metadata.num_rows


def list_input_files_with_sizes(repo_id, subdir):
    """Return [(in_repo_path, size_bytes), ...] for all .parquet files under repo_id/subdir.

    Uses HfApi.list_repo_tree(recursive=True) so we get one indexed listing call
    instead of 17k+ per-file footer reads.
    """
    api = HfApi()
    out = []
    for entry in api.list_repo_tree(
        repo_id, path_in_repo=subdir, repo_type="dataset", recursive=True
    ):
        path = getattr(entry, "path", None)
        size = getattr(entry, "size", None)
        if path is None or size is None:
            continue
        if not path.endswith(".parquet"):
            continue
        out.append((path, int(size)))
    out.sort(key=lambda t: t[0])
    return out


def seed_or_merge_manifest(existing, listing, *, seconds_per_byte):
    by_file = {r.input_file: r for r in existing}
    for input_file, size_bytes in listing:
        if input_file in by_file:
            continue
        by_file[input_file] = ManifestRow(
            input_file=input_file,
            size_bytes=int(size_bytes),
            est_seconds=float(size_bytes) * seconds_per_byte,
            status="pending",
            attempts=0,
            job_id=None,
            assigned_at=None,
            completed_at=None,
            output_path=None,
            last_error=None,
            worker_elapsed_s=None,
            row_count=None,
        )
    return [by_file[k] for k in sorted(by_file)]


def reconcile_against_outputs(rows, existing_outputs):
    import time as _time
    out = []
    for r in rows:
        candidate = f"predictions/{Path(r.input_file).name}"
        if candidate in existing_outputs and r.status != "done":
            r = ManifestRow(**{
                **asdict(r),
                "status": "done",
                "output_path": candidate,
                "completed_at": _time.time(),
            })
        out.append(r)
    return out


def list_existing_outputs(repo_id):
    """Return set of in-repo paths like {'predictions/x.parquet', ...}.

    Returns an empty set if the repo or the predictions/ subdir does not yet
    exist. The list_repo_tree() call returns a generator that defers HTTP
    requests until iteration, so the try must wrap the for-loop, not the call.
    """
    api = HfApi()
    out = set()
    try:
        entries = api.list_repo_tree(
            repo_id, path_in_repo="predictions",
            repo_type="dataset", recursive=True,
        )
        for entry in entries:
            path = getattr(entry, "path", None)
            if path and path.endswith(".parquet"):
                out.add(path)
    except Exception:
        return set()
    return out


IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
FLAVOR = "a100-large"


def submit_a100_job(*, script_path, worker_argv, token, timeout="2h", max_retries=10):
    """Submit one HF Job on the a100-large flavor; retry on 429.

    The worker script is base64-encoded and shipped via the `bash -c ... uv run`
    prelude required for the pytorch image (see CLAUDE.md HF Jobs invariants).
    """
    b64 = base64.b64encode(Path(script_path).read_bytes()).decode()
    argv_str = " ".join(worker_argv)
    cmd = ["bash", "-c",
           "set -euxo pipefail && apt-get update -qq && apt-get install -y -qq git && "
           "pip install --quiet --root-user-action=ignore uv && "
           f"echo {b64} | base64 -d > /tmp/p.py && rm -rf /root/.cache/uv/environments-v2 && "
           f"uv run /tmp/p.py {argv_str}"]
    api = HfApi()
    last_exc: Optional[Exception] = None
    retry_markers = ("429", "500", "502", "503", "504",
                     "timeout", "connection", "Connection")
    for attempt in range(max_retries):
        try:
            job = api.run_job(
                image=IMAGE, command=cmd,
                secrets={"HF_TOKEN": token},
                flavor=FLAVOR, timeout=timeout,
            )
            return job.id
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if any(m in msg for m in retry_markers):
                time.sleep(60)
                continue
            raise
    assert last_exc is not None
    raise last_exc


@dataclass
class JobState:
    stage: str           # e.g. RUNNING / COMPLETED / ERROR / CANCELED
    done_files: list     # parsed [done ...] events
    last_log_ts: Optional[float]


def poll_job_state(job_id):
    """Inspect job + tail its logs for [done ...] events.

    Uses huggingface_hub.HfApi.inspect_job + fetch_job_logs (confirmed available
    in the installed version). Transient log-fetch errors are swallowed so the
    orchestrator can retry on its next poll.
    """
    api = HfApi()
    info = api.inspect_job(job_id=job_id)
    stage = getattr(info.status, "stage", None) or getattr(info, "status", "UNKNOWN")
    done = []
    last_ts: Optional[float] = None
    # fetch_job_logs re-streams the entire job log on every poll (no cursor),
    # so caps must accommodate cumulative log volume across the whole run, not
    # just per-poll deltas. Workers emit ~30-50 lines per processed file plus
    # cold-start chatter, so a 2h job can easily hit 100k lines.
    max_lines = 500_000
    max_seconds = 90.0
    n_lines = 0
    bailed_early = False
    start = time.monotonic()
    try:
        for entry in api.fetch_job_logs(job_id=job_id):
            n_lines += 1
            # fetch_job_logs yields plain strings in this huggingface_hub version,
            # but older/newer versions may yield objects with a .data attribute.
            line = entry if isinstance(entry, str) else getattr(entry, "data", "") or ""
            parsed = parse_done_line(line)
            if parsed:
                done.append(parsed)
            last_ts = time.time()
            if n_lines >= max_lines:
                bailed_early = True
                break
            if time.monotonic() - start >= max_seconds:
                bailed_early = True
                break
    except Exception:
        pass
    if bailed_early:
        logger.info(
            "[poll] bailed early on job=%s after %d lines / %.1fs",
            job_id, n_lines, time.monotonic() - start,
        )
    return JobState(stage=str(stage), done_files=done, last_log_ts=last_ts)


import argparse
import json
import logging
import sys
import time as _time
from collections import Counter
from datetime import datetime, timezone

logger = logging.getLogger("orchestrate_extractions")


def _render_readme(summary):
    return f"""# arxiv-funding-statement-extractions

Funding-statement extractions over `cometadata/arxiv-latex-extract-full-text/results-2026-04-24/`.

- Extractor: `funding_statement_extractor` @ `statement-only-extraction`
- Model: `lightonai/GTE-ModernColBERT-v1`
- Config: Tier 2 (paragraph prefilter, regex floor 11.0, top_k=5, threshold=10.0)
- Hardware: A100-large bf16, batch_size 512
- Files processed: {summary['n_files']}
- Status: {summary['status_counts']}
- Total rows: {summary['total_rows']:,}
- Total worker seconds: {summary['total_worker_seconds']:.0f}
- Est cost @ $5/hr: ${summary['estimated_cost_usd']:.2f}
- Completed: {summary['completed_at']}

## Schema

Per row in `predictions/*.parquet`:
- `arxiv_id`, `doc_id`, `input_file`, `row_idx`
- `predicted_statements`: list[str]
- `predicted_details`: list[struct{{statement, score, query, paragraph_idx}}]
- `text_length`, `latency_ms`, `error`
"""


def write_run_summary(manifest, output_repo, manifest_path):
    """Write local run summary JSON and push summary + manifest snapshot + README to hub."""
    manifest_path = Path(manifest_path)
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_files": len(manifest),
        "status_counts": dict(Counter(r.status for r in manifest)),
        "total_rows": sum(
            (r.row_count or 0) for r in manifest if r.status == "done"
        ),
        "total_worker_seconds": sum(
            (r.worker_elapsed_s or 0) for r in manifest
        ),
        "estimated_cost_usd": sum(
            (r.worker_elapsed_s or 0) for r in manifest
        ) / 3600.0 * 5.0,
        "failed_files": [
            {"input_file": r.input_file, "last_error": r.last_error}
            for r in manifest if r.status == "failed"
        ],
    }
    summary_path = manifest_path.parent / (manifest_path.stem + "-summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(summary_path),
        path_in_repo="run_metadata/summary.json",
        repo_id=output_repo,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo="run_metadata/manifest-snapshot.parquet",
        repo_id=output_repo,
        repo_type="dataset",
    )
    readme = _render_readme(summary)
    readme_path = manifest_path.parent / "README.md"
    readme_path.write_text(readme)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=output_repo,
        repo_type="dataset",
    )
    return summary


def parse_orch_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-repo", default="cometadata/arxiv-latex-extract-full-text")
    p.add_argument("--input-subdir", default="results-2026-04-24")
    p.add_argument("--output-repo",
                   default="cometadata/arxiv-funding-statement-extractions")
    p.add_argument("--manifest",
                   default="manifests/arxiv-extractions-2026-04-24.parquet")
    p.add_argument("--max-in-flight", type=int, default=8)
    p.add_argument("--target-seconds", type=int, default=5400,
                   help="Target wall-clock per job batch (default 90 min).")
    p.add_argument("--max-files-per-batch", type=int, default=50)
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--stuck-min", type=int, default=15,
                   help="Minutes of log silence before cancelling a RUNNING job.")
    p.add_argument("--poll-interval-s", type=int, default=60)
    p.add_argument("--seed-seconds-per-byte", type=float, default=9e-7,
                   help="Initial estimate; overridden by EMA after first jobs complete.")
    p.add_argument("--worker-script", default="scripts/extract_funding_job.py")
    p.add_argument("--job-timeout", default="2h")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't actually submit; mark batches done immediately.")
    return p.parse_args(argv)


def main(argv=None, *, submit_fn=None, poll_fn=None) -> int:
    args = parse_orch_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
        logger.info("loaded manifest with %d rows", len(manifest))
    else:
        manifest = []

    logger.info("listing inputs from %s/%s", args.input_repo, args.input_subdir)
    listing = list_input_files_with_sizes(args.input_repo, args.input_subdir)
    logger.info("found %d input files (%d total bytes)",
                len(listing), sum(s for _, s in listing))
    manifest = seed_or_merge_manifest(
        manifest, listing, seconds_per_byte=args.seed_seconds_per_byte
    )

    logger.info("listing existing outputs in %s", args.output_repo)
    existing = list_existing_outputs(args.output_repo)
    logger.info("found %d existing output files", len(existing))
    manifest = reconcile_against_outputs(manifest, existing)
    write_manifest(manifest, manifest_path)

    if args.dry_run:
        token = "dry-run"
    else:
        token = os.environ.get("HF_TOKEN") or open(
            os.path.expanduser("~/.cache/huggingface/token")
        ).read().strip()

    # In dry-run mode synthesize submit/poll so the assigned->done state
    # machine + EMA update + release path actually run.
    if args.dry_run and submit_fn is None and poll_fn is None:
        seed_spb = args.seed_seconds_per_byte
        _dry_jobs: dict = {}

        def _arg_value(argv, name):
            for i, tok in enumerate(argv):
                if tok == name and i + 1 < len(argv):
                    return argv[i + 1]
            return None

        def _dry_submit(*, script_path, worker_argv, token, timeout,
                        max_retries=10):
            files = (_arg_value(worker_argv, "--input-files") or "").split(",")
            files = [f for f in files if f]
            job_tag = _arg_value(worker_argv, "--job-tag") or "x"
            jid = f"dry-{job_tag}"
            _dry_jobs[jid] = files
            return jid

        # We need access to the manifest to look up size_bytes; stash a
        # closure-captured callable that the loop wires up.
        _manifest_lookup: dict = {"by_file": lambda: {}}

        def _dry_poll(job_id):
            files = _dry_jobs.get(job_id, [])
            mb = _manifest_lookup["by_file"]()
            done = []
            for f in files:
                row = mb.get(f)
                size = row.size_bytes if row else 4096
                done.append({
                    "file": f,
                    "rows": max(1, size // 4096),
                    "elapsed_s": size * seed_spb,
                })
            return JobState(stage="COMPLETED", done_files=done,
                            last_log_ts=_time.time())

        submit_fn = _dry_submit
        poll_fn = _dry_poll
    else:
        _manifest_lookup = None

    if submit_fn is None:
        submit_fn = submit_a100_job
    if poll_fn is None:
        poll_fn = poll_job_state

    in_flight: dict = {}
    # Resume support: rebuild in_flight from manifest rows already in 'assigned'
    # state (e.g. from a prior orchestrator process that was killed mid-run).
    # Group by job_id so the poll loop can drive these to terminal state.
    if not args.dry_run:
        resumed_groups: dict = {}
        for r in manifest:
            if r.status == "assigned" and r.job_id:
                resumed_groups.setdefault(r.job_id, []).append(r)
        for jid, rows in resumed_groups.items():
            submitted_at = min((r.assigned_at or _time.time()) for r in rows)
            in_flight[jid] = {
                "files": [r.input_file for r in rows],
                "submitted_at": submitted_at,
                "last_log_ts": _time.time(),  # reset clock so stuck-detection is fair
            }
        if in_flight:
            logger.info(
                "resumed %d in-flight jobs covering %d assigned files",
                len(in_flight), sum(len(v["files"]) for v in in_flight.values()),
            )
    seconds_per_byte_ema: Optional[float] = None

    def manifest_by_file():
        return {r.input_file: r for r in manifest}

    if _manifest_lookup is not None:
        _manifest_lookup["by_file"] = manifest_by_file

    while True:
        # --- Poll in-flight jobs ---
        for job_id in list(in_flight):
            try:
                state = poll_fn(job_id)
            except Exception as exc:
                logger.warning("poll failed for %s: %s", job_id, exc)
                continue
            mb = manifest_by_file()
            for ev in state.done_files:
                row = mb.get(ev["file"])
                if row and row.status == "assigned" and row.job_id == job_id:
                    row.status = "done"
                    row.completed_at = _time.time()
                    row.output_path = f"predictions/{Path(ev['file']).name}"
                    row.worker_elapsed_s = ev["elapsed_s"]
                    row.row_count = ev["rows"]
                    if row.size_bytes > 0:
                        sample = ev["elapsed_s"] / row.size_bytes
                        seconds_per_byte_ema = update_ema(
                            seconds_per_byte_ema, sample, 0.3
                        )
                    logger.info(
                        "[done] file=%s rows=%d elapsed=%.1fs ema=%s s/byte",
                        ev["file"], ev["rows"], ev["elapsed_s"],
                        f"{seconds_per_byte_ema:.3e}" if seconds_per_byte_ema else "n/a",
                    )
            if state.last_log_ts:
                in_flight[job_id]["last_log_ts"] = state.last_log_ts

            silent_min = (_time.time() - in_flight[job_id]["last_log_ts"]) / 60.0
            stuck_running = silent_min > args.stuck_min and state.stage == "RUNNING"
            # SCHEDULING: HF can leave a submitted job waiting for a GPU slot
            # for hours during contention. The pool slot is wasted, so cancel
            # and let the orchestrator resubmit.
            sched_age_min = (
                _time.time() - in_flight[job_id]["submitted_at"]
            ) / 60.0
            stuck_scheduling = (
                state.stage == "SCHEDULING" and sched_age_min > args.stuck_min
            )
            # Track whether WE cancelled this job for infra reasons (vs the
            # job ending naturally due to ERROR/COMPLETED). Infra cancellations
            # should NOT consume the row's retry budget — those files never got
            # a real attempt at the work.
            self_cancelled_infra = False
            if stuck_running or stuck_scheduling:
                reason = "running silent" if stuck_running else "stuck in SCHEDULING"
                logger.warning(
                    "job %s %s for %.1fmin -- cancelling (no attempt charged)",
                    job_id, reason,
                    silent_min if stuck_running else sched_age_min,
                )
                if not args.dry_run:
                    try:
                        HfApi().cancel_job(job_id=job_id)
                    except Exception as exc:
                        logger.warning("cancel failed: %s", exc)
                state = JobState(stage="CANCELED",
                                 done_files=state.done_files,
                                 last_log_ts=state.last_log_ts)
                self_cancelled_infra = True

            if state.stage in ("COMPLETED", "ERROR", "CANCELED"):
                for f in in_flight[job_id]["files"]:
                    row = mb.get(f)
                    if row and row.status == "assigned" and row.job_id == job_id:
                        if self_cancelled_infra:
                            # Infra cancel: just release back to pending,
                            # don't bump attempts — file never had a real chance.
                            row.status = "pending"
                            row.job_id = None
                            row.assigned_at = None
                            row.last_error = (
                                f"job {job_id} cancelled (infra: "
                                f"{'sched-stuck' if stuck_scheduling else 'silent-stuck'})"
                            )
                            logger.info(
                                "[released-infra] file=%s attempts=%d (unchanged)",
                                f, row.attempts,
                            )
                            continue
                        row.attempts += 1
                        if row.attempts >= args.max_attempts:
                            row.status = "failed"
                            row.last_error = f"job {job_id} stage={state.stage}"
                            logger.error("[failed] file=%s after %d attempts",
                                         f, row.attempts)
                        else:
                            row.status = "pending"
                            row.job_id = None
                            row.assigned_at = None
                            row.last_error = f"job {job_id} stage={state.stage}"
                            logger.warning("[released] file=%s attempts=%d",
                                           f, row.attempts)
                del in_flight[job_id]

        # Recompute pending estimates with EMA
        if seconds_per_byte_ema:
            for r in manifest:
                if r.status == "pending":
                    r.est_seconds = r.size_bytes * seconds_per_byte_ema

        # --- Refill ---
        while len(in_flight) < args.max_in_flight:
            batch = pick_next_batch(
                manifest,
                target_seconds=args.target_seconds,
                max_files=args.max_files_per_batch,
            )
            if not batch:
                break
            job_tag = f"orch-{int(_time.time())}-{len(in_flight)}"
            input_files_arg = ",".join(r.input_file for r in batch)
            worker_argv = [
                "--input-repo", args.input_repo,
                "--input-files", input_files_arg,
                "--output-repo", args.output_repo,
                "--job-tag", job_tag,
            ]
            if args.dry_run:
                logger.info(
                    "[dry-run] submitting synthetic batch (%d files, est %.1fs)",
                    len(batch), sum(r.est_seconds for r in batch),
                )

            try:
                job_id = submit_fn(
                    script_path=args.worker_script,
                    worker_argv=worker_argv,
                    token=token,
                    timeout=args.job_timeout,
                )
            except Exception as exc:
                logger.error(
                    "submit failed: %s -- leaving %d files pending",
                    exc, len(batch),
                )
                break
            now = _time.time()
            for r in batch:
                r.status = "assigned"
                r.job_id = job_id
                r.assigned_at = now
            in_flight[job_id] = {
                "files": [r.input_file for r in batch],
                "submitted_at": now,
                "last_log_ts": now,
            }
            logger.info(
                "[submitted] job=%s files=%d est_total=%.0fs",
                job_id, len(batch), sum(r.est_seconds for r in batch),
            )

        write_manifest(manifest, manifest_path)

        n_pending = sum(1 for r in manifest if r.status == "pending")
        n_assigned = sum(1 for r in manifest if r.status == "assigned")
        n_done = sum(1 for r in manifest if r.status == "done")
        n_failed = sum(1 for r in manifest if r.status == "failed")
        logger.info(
            "status: pending=%d assigned=%d done=%d failed=%d in_flight=%d",
            n_pending, n_assigned, n_done, n_failed, len(in_flight),
        )
        if n_pending == 0 and n_assigned == 0 and not in_flight:
            logger.info("all done")
            break

        if args.dry_run:
            # In dry-run we never actually have in_flight jobs to wait on; loop will
            # exit on the next iteration once everything is marked done.
            continue
        time.sleep(args.poll_interval_s)

    if not args.dry_run:
        try:
            write_run_summary(manifest, args.output_repo, manifest_path)
        except Exception as exc:
            logger.warning("write_run_summary failed: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
