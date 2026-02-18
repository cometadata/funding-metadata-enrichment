import argparse
import copy
import logging
import multiprocessing
import sys
from pathlib import Path

from funding_extractor.statements import cli as statements_cli
from funding_extractor.entities import cli as entities_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract-funding",
        description="Extract funding information from research documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  statements    Extract funding statements from documents (stage 1)
  entities      Extract structured entities from statements (stage 2)
  pipeline      Run both stages end-to-end
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    # statements subcommand
    stmt_parser = subparsers.add_parser(
        "statements",
        help="Extract funding statements from documents (stage 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    statements_cli.add_arguments(stmt_parser)

    # entities subcommand
    entity_parser = subparsers.add_parser(
        "entities",
        help="Extract structured entities from funding statements (stage 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    entities_cli.add_arguments(entity_parser)

    # pipeline subcommand — resolve conflicts because both stages share
    # flags like -i, -o, -v, etc.  The last add_arguments call wins.
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run full extraction pipeline (statements + entities)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        conflict_handler="resolve",
    )
    statements_cli.add_arguments(pipeline_parser)
    entities_cli.add_arguments(pipeline_parser)
    pipeline_parser.add_argument(
        "--intermediate-file",
        help="Path for intermediate JSONL (default: auto-generated temp file)",
    )

    return parser


def main_with_args(args: argparse.Namespace) -> None:
    if args.command == "statements":
        statements_cli.run(args)

    elif args.command == "entities":
        entities_cli.run(args)

    elif args.command == "pipeline":
        # Determine intermediate file path
        intermediate = getattr(args, "intermediate_file", None)
        if intermediate:
            intermediate_path = Path(intermediate)
        else:
            intermediate_path = Path(str(args.output) + ".statements.jsonl")

        # Stage 1: statements extraction
        stage1_args = copy.copy(args)
        stage1_args.output = str(intermediate_path)
        statements_cli.run(stage1_args)

        # Stage 2: entity extraction
        stage2_args = copy.copy(args)
        stage2_args.input = str(intermediate_path)
        entities_cli.run(stage2_args)

    else:
        build_parser().print_help()
        sys.exit(1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    log_level = logging.WARNING
    if getattr(args, "verbose", False):
        log_level = logging.INFO
    if getattr(args, "debug", False):
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)

    main_with_args(args)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
