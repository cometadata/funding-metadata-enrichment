# heal-markdown

A Rust utility for restoring PDF-extracted markdown into readable format.

## Build

```bash
cargo build --release
```

The release binary will be at `target/release/heal-markdown`.

## Usage

### Process a single file

```bash
heal-markdown path/to/file.md
```

Output is written alongside the original as `file.healed.md`.

### Process a directory

```bash
heal-markdown path/to/dir/ --out-dir output/
```

Recursively finds all `.md` files and writes cleaned versions to `output/`, preserving directory structure.

### Process in place

```bash
heal-markdown path/to/dir/ --in-place
```

Overwrites original files with cleaned versions.

### Process Parquet input

```bash
heal-markdown input.parquet --out-dir output/ --output-format parquet
```

Reads a Parquet file with `filename` and `content` columns, processes each document, and writes results to a new Parquet file.

### Process Parquet to both formats

```bash
heal-markdown input.parquet --out-dir output/ --output-format both
```

Writes both a Parquet file with cleaned content and individual markdown files.

## CLI Options

```
Restore PDF-extracted Markdown into readable CommonMark.

Usage: heal-markdown [OPTIONS] <PATH>

Arguments:
  <PATH>  Markdown file, directory, or Parquet file to process

Options:
      --in-place
          Overwrite files instead of writing alongside originals
      --out-dir <OUT_DIR>
          Directory to write cleaned files
      --page-lines <PAGE_LINES>
          Lines per page for header/footer detection [default: 50]
      --repeat-threshold <REPEAT_THRESHOLD>
          Frequency threshold for repeated line removal [default: 0.8]
      --max-header-words <MAX_HEADER_WORDS>
          Max words for repeated line candidates [default: 12]
      --skip-tables
          No-op, accepted for compatibility (table repair deferred)
      --strip-citations
          Remove [12]-style bracketed citations
      --exclude-failed
          Exclude files that convert to zero bytes
      --output-format <OUTPUT_FORMAT>
          Output format: markdown, parquet, or both [possible values: markdown, parquet, both]
      --failed-output <FAILED_OUTPUT>
          Path for failed records Parquet file
      --threads <THREADS>
          Number of parallel threads
  -h, --help
          Print help
```

## Notes

- `--skip-tables` is accepted for CLI compatibility with the Python version but is a no-op. Table repair is deferred to a future release.
- `--threads` controls the number of Rayon worker threads used for parallel processing. By default, Rayon uses one thread per logical CPU core.
- When processing Parquet input, the tool expects columns named `filename` and `content`.
- The `--failed-output` option writes a separate Parquet file containing documents that failed validation or produced zero-byte output, along with failure categorization metadata.
