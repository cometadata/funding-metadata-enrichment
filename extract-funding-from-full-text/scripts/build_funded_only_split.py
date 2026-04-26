"""Build a filtered split of funding-statement extractions.

Loads all output parquets from `cometadata/arxiv-funding-statement-extractions`,
keeps only rows whose `predicted_statements` is non-empty, and re-packs them
into deterministic 10k-row shards under `funded_only/shard-NNNNN.parquet` on
the same dataset.

Run from repo root:

    .venv/bin/python3 scripts/build_funded_only_split.py \
        --output-repo cometadata/arxiv-funding-statement-extractions \
        --rows-per-shard 10000
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import CommitOperationAdd, HfApi

logger = logging.getLogger("build_funded_only_split")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output-repo",
                   default="cometadata/arxiv-funding-statement-extractions")
    p.add_argument("--predictions-glob", default="predictions/**/*.parquet",
                   nargs="+",
                   help="One or more glob patterns. Defaults match both flat "
                        "predictions/*.parquet (legacy) and predictions/<year>/*.parquet.")
    p.add_argument("--shard-prefix", default="funded_only/shard-")
    p.add_argument("--rows-per-shard", type=int, default=10_000)
    p.add_argument("--max-statement-chars", type=int, default=8000,
                   help="Drop rows where the joined predicted statements exceed "
                        "this many characters (filters out runaway extractions).")
    p.add_argument("--commit-batch-size", type=int, default=10,
                   help="Shards uploaded per hub commit.")
    p.add_argument("--num-proc", type=int, default=8,
                   help="Parallel workers for streaming filter.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build shards locally but skip the hub upload.")
    p.add_argument("--staging-dir", default=None,
                   help="Where to write intermediate shards (default: tmp).")
    p.add_argument("--snapshot-dir", default=None,
                   help="Where to mirror the predictions/ tree from hub "
                        "(default: tmp; reuse path to skip re-download).")
    return p.parse_args(argv)


def stream_filtered_rows(repo_id: str, globs, num_proc: int,
                         max_statement_chars: int, snapshot_dir: Path):
    """Bulk-fetch predictions/ once via direct HTTPS resolve/main URLs.

    snapshot_download() goes through xet-read-token (1 API call per file),
    which trips the 3000 req / 5-min quota at our scale (17.5k files).
    Direct https://huggingface.co/datasets/<repo>/resolve/main/<path> downloads
    are served from storage edges and don't count against the API quota.
    """
    import os
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import HfApi
    if isinstance(globs, str):
        globs = [globs]
    api = HfApi()
    all_files = api.list_repo_files(repo_id, repo_type="dataset")
    pred_files = [
        f for f in all_files
        if f.startswith("predictions/") and f.endswith(".parquet")
    ]
    logger.info("repo has %d prediction parquets", len(pred_files))

    token_path = os.path.expanduser("~/.cache/huggingface/token")
    headers = {}
    if os.path.exists(token_path):
        with open(token_path) as f:
            headers["Authorization"] = f"Bearer {f.read().strip()}"

    def fetch_one(in_repo_path):
        local = snapshot_dir / in_repo_path
        if local.exists() and local.stat().st_size > 0:
            return in_repo_path, "cached"
        local.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{in_repo_path}"
        for attempt in range(8):
            try:
                r = requests.get(url, headers=headers, stream=True, timeout=60)
                if r.status_code == 200:
                    tmp = local.with_suffix(local.suffix + ".tmp")
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=64 * 1024):
                            fh.write(chunk)
                    tmp.rename(local)
                    return in_repo_path, "fetched"
                if r.status_code == 429:
                    time.sleep(30 * (attempt + 1))
                    continue
                r.raise_for_status()
            except (requests.ConnectionError, requests.Timeout,
                    requests.exceptions.ChunkedEncodingError):
                time.sleep(2 * (attempt + 1))
                continue
        raise RuntimeError(f"give up fetching {in_repo_path}")

    fetched = cached = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, p): p for p in pred_files}
        for i, fut in enumerate(as_completed(futures), 1):
            _, status = fut.result()
            if status == "cached":
                cached += 1
            else:
                fetched += 1
            if i % 1000 == 0:
                logger.info("fetched=%d cached=%d / %d", fetched, cached, len(pred_files))
    logger.info("snapshot complete: fetched=%d cached=%d total=%d",
                fetched, cached, len(pred_files))

    local_files = sorted(str(p) for p in snapshot_dir.rglob("*.parquet"))
    logger.info("snapshot has %d local parquet files", len(local_files))

    # Read each parquet with pyarrow (skips the `datasets` library's strict
    # schema-inference path that trips on subtle nullable/null-type drift
    # across 17k+ files), filter rows in-memory, accumulate.
    import pyarrow as pa
    import pyarrow.parquet as pq
    keep_batches = []
    n_total = 0
    n_kept = 0
    for i, path in enumerate(local_files, 1):
        tbl = pq.read_table(path)
        n = tbl.num_rows
        n_total += n
        if n == 0:
            continue
        stmts_col = tbl.column("predicted_statements").to_pylist()
        mask = pa.array([
            bool(s) and sum(len(x) for x in s) <= max_statement_chars
            for s in stmts_col
        ])
        kept = tbl.filter(mask)
        if kept.num_rows:
            keep_batches.append(kept)
            n_kept += kept.num_rows
        if i % 2000 == 0:
            logger.info("scanned %d/%d files; kept=%d/%d rows",
                        i, len(local_files), n_kept, n_total)
    logger.info("filter done: kept %d / %d rows (%.1f%%)",
                n_kept, n_total, 100.0 * n_kept / max(1, n_total))
    if not keep_batches:
        return pa.Table.from_pylist([])
    combined = pa.concat_tables(keep_batches, promote_options="default")
    return combined


def write_shards(filtered_table, *, rows_per_shard: int, shard_prefix: str,
                 staging: Path) -> list[tuple[str, Path, int]]:
    """Slice the filtered pyarrow Table into rows_per_shard chunks and write
    each to disk.

    Returns [(path_in_repo, local_path, n_rows), ...].
    """
    import pyarrow.parquet as pq
    n_total = filtered_table.num_rows
    n_shards = math.ceil(n_total / rows_per_shard) if n_total else 0
    logger.info("writing %d shards (%d rows each, last possibly smaller)",
                n_shards, rows_per_shard)
    out = []
    for i in range(n_shards):
        start = i * rows_per_shard
        end = min(start + rows_per_shard, n_total)
        shard = filtered_table.slice(start, end - start)
        local = staging / f"shard-{i:05d}.parquet"
        pq.write_table(shard, str(local), compression="zstd")
        path_in_repo = f"{shard_prefix}{i:05d}.parquet"
        out.append((path_in_repo, local, end - start))
        logger.info("[shard %d] rows=%d local=%s", i, end - start, local)
    return out


def commit_in_batches(api: HfApi, *, repo_id: str, shards,
                      commit_batch_size: int) -> int:
    """Push shards to hub in groups of commit_batch_size to stay under rate limits."""
    n_commits = 0
    for i in range(0, len(shards), commit_batch_size):
        chunk = shards[i:i + commit_batch_size]
        ops = [
            CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(local))
            for p, local, _ in chunk
        ]
        msg = (f"funded_only: add shards {chunk[0][0].rsplit('-',1)[1].split('.')[0]}"
               f" .. {chunk[-1][0].rsplit('-',1)[1].split('.')[0]}"
               f" ({len(chunk)} files)")
        api.create_commit(
            repo_id=repo_id, repo_type="dataset",
            operations=ops, commit_message=msg,
        )
        n_commits += 1
        logger.info("[commit %d] uploaded %d shards", n_commits, len(chunk))
    return n_commits


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        stream=sys.stdout)

    snapshot = Path(args.snapshot_dir) if args.snapshot_dir else \
        Path(tempfile.mkdtemp(prefix="funded_snapshot_"))
    snapshot.mkdir(parents=True, exist_ok=True)
    logger.info("snapshot dir: %s", snapshot)

    filtered = stream_filtered_rows(
        args.output_repo, args.predictions_glob,
        args.num_proc, args.max_statement_chars, snapshot,
    )

    staging = Path(args.staging_dir) if args.staging_dir else \
        Path(tempfile.mkdtemp(prefix="funded_only_"))
    staging.mkdir(parents=True, exist_ok=True)
    logger.info("staging dir: %s", staging)

    shards = write_shards(filtered,
                          rows_per_shard=args.rows_per_shard,
                          shard_prefix=args.shard_prefix,
                          staging=staging)

    if args.dry_run:
        logger.info("[dry-run] %d shards written locally; skipping hub upload",
                    len(shards))
        return 0

    api = HfApi()
    n_commits = commit_in_batches(
        api, repo_id=args.output_repo, shards=shards,
        commit_batch_size=args.commit_batch_size,
    )
    logger.info("done: %d shards in %d commits, total rows %d",
                len(shards), n_commits,
                sum(n for _, _, n in shards))
    return 0


if __name__ == "__main__":
    sys.exit(main())
