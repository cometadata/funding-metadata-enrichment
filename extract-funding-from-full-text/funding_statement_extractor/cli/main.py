import argparse
import logging
import multiprocessing
import sys

from funding_statement_extractor.statements import cli as statements_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract-funding-statements",
        description="Extract funding statements from research documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    statements_cli.add_arguments(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.WARNING
    if getattr(args, "verbose", False):
        log_level = logging.INFO
    if getattr(args, "debug", False):
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)

    statements_cli.run(args)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
