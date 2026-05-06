import argparse
import logging
import multiprocessing

from funding_statement_extractor.statements import cli as statements_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract-funding-statements",
        description="Extract funding statements from research documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single markdown file
  %(prog)s -i document.md -o results.json

  # Process a directory of markdown files
  %(prog)s -i /path/to/md/files -o results.json

  # Stream a directory of parquet chunks (text column auto-detects, prefers "markdown")
  %(prog)s -i /path/to/parquet-chunks --input-format parquet \\
    --parquet-text-column markdown --parquet-id-column source_id -o results.json

  # Normalize extracted statements
  %(prog)s -i docs/ -o results.json --normalize
        """,
    )
    statements_cli.add_arguments(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.WARNING
    if getattr(args, "verbose", False):
        log_level = logging.INFO
    logging.basicConfig(level=log_level)

    statements_cli.run(args)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
