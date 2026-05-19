import argparse
import logging

from funding_statement_extractor_cascade.cascade import cli as cascade_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="funding-statement-extractor-cascade",
        description="Extract cleaned funding statements from VLM-converted arXiv markdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s -i paper.md -o results.json
  %(prog)s -i docs/ --skip-cleanup --dtype bf16
  %(prog)s -i bench.parquet --parquet-text-column vlm_markdown --parquet-id-column arxiv_id
""",
    )
    cascade_cli.add_arguments(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    cascade_cli.run(args)


if __name__ == "__main__":
    main()
