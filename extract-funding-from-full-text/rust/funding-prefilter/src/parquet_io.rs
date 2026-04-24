//! Parquet ingestion and output.
//!
//! * [`read_shard_rows`] — iterate over `(arxiv_id, text, status)` rows from an
//!   input shard with column projection so the unused columns
//!   (`stage_timings_us`, `peak_memory_bytes`, etc.) are never decoded.
//! * [`write_candidates`] — write one `CandidateRow` per matched document to a
//!   small Snappy-compressed parquet file. If `rows` is empty we still emit a
//!   0-row file with the correct schema so resumability can distinguish
//!   "shard was processed, had no candidates" from "shard never ran".
//! * [`read_candidate_arxiv_ids`] — helper for the `merge` subcommand that
//!   reads only the `arxiv_id` column from a candidate shard written above.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use arrow_array::{Array, LargeStringArray, RecordBatch, RecordBatchReader, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

/// Single row of the input parquet shard, projected down to the three columns
/// we actually use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputRow {
    pub arxiv_id: String,
    pub text: String,
    pub status: String,
}

/// Single row of the output candidate parquet shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateRow {
    pub arxiv_id: String,
    pub num_matched_paragraphs: u32,
}

/// Locate the arrow field index for `name`, or return a clear error naming the
/// missing column and the path we were reading.
fn column_index(schema: &Schema, name: &str, path: &Path) -> Result<usize> {
    schema.index_of(name).map_err(|_| {
        anyhow!(
            "input parquet at {} is missing required column {:?}",
            path.display(),
            name
        )
    })
}

/// Read a string cell from an arrow array, transparently handling both
/// `Utf8` (i32 offsets, [`StringArray`]) and `LargeUtf8` (i64 offsets,
/// [`LargeStringArray`]). Real shards in
/// `cometadata/arxiv-latex-extract-full-text` use `LargeUtf8` for the `text`
/// column, so a single-type downcast (e.g. only `StringArray`) silently fails
/// for every row. Callers should funnel any string column read through this
/// helper to stay defensive against future schema variations.
fn read_string_cell(
    col: &dyn Array,
    row: usize,
    column_label: &str,
    path: &Path,
) -> Result<String> {
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        if arr.is_null(row) {
            return Err(anyhow!(
                "null `{}` at row {} in {}",
                column_label,
                row,
                path.display()
            ));
        }
        return Ok(arr.value(row).to_string());
    }
    if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        if arr.is_null(row) {
            return Err(anyhow!(
                "null `{}` at row {} in {}",
                column_label,
                row,
                path.display()
            ));
        }
        return Ok(arr.value(row).to_string());
    }
    Err(anyhow!(
        "column `{}` in {} has unsupported type {:?} (expected Utf8 or LargeUtf8)",
        column_label,
        path.display(),
        col.data_type()
    ))
}

/// Like [`read_string_cell`] but treats null cells as empty strings instead of
/// erroring. Used for columns that are nullable in the upstream schema —
/// notably `text`, which is `null` whenever extraction failed (`status != "ok"`).
/// We still want to read the row so the pipeline's `status == "ok"` filter can
/// drop it; an empty `text` is harmless because the gate returns `false` on
/// empty input.
fn read_optional_string_cell(
    col: &dyn Array,
    row: usize,
    column_label: &str,
    path: &Path,
) -> Result<String> {
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        if arr.is_null(row) {
            return Ok(String::new());
        }
        return Ok(arr.value(row).to_string());
    }
    if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        if arr.is_null(row) {
            return Ok(String::new());
        }
        return Ok(arr.value(row).to_string());
    }
    Err(anyhow!(
        "column `{}` in {} has unsupported type {:?} (expected Utf8 or LargeUtf8)",
        column_label,
        path.display(),
        col.data_type()
    ))
}

/// Iterator that walks a `ParquetRecordBatchReader`, flattening the current
/// `RecordBatch` into individual [`InputRow`] values and fetching the next
/// batch when the current one is exhausted.
///
/// Rather than materializing all rows up front, we hold a single
/// `RecordBatch` at a time plus a row cursor — the projected columns keep the
/// per-batch footprint small even for large shards.
struct ShardRowIter {
    reader: ParquetRecordBatchReader,
    path: std::path::PathBuf,
    arxiv_col: usize,
    text_col: usize,
    status_col: usize,
    current: Option<RecordBatch>,
    row_cursor: usize,
}

impl ShardRowIter {
    fn load_next_batch(&mut self) -> Result<bool> {
        match self.reader.next() {
            None => {
                self.current = None;
                Ok(false)
            }
            Some(Ok(batch)) => {
                self.current = Some(batch);
                self.row_cursor = 0;
                Ok(true)
            }
            Some(Err(e)) => Err(anyhow::Error::new(e).context(format!(
                "reading batch from parquet at {}",
                self.path.display()
            ))),
        }
    }

    fn row_from_current(&mut self) -> Result<InputRow> {
        let batch = self.current.as_ref().expect("row_from_current called with no current batch");
        let row = self.row_cursor;

        // Funnel every string column through `read_string_cell` so we
        // transparently accept both `Utf8` and `LargeUtf8` (real shards have
        // `text: LargeUtf8`). Null checks happen inside the helper.
        let arxiv_id = read_string_cell(
            batch.column(self.arxiv_col).as_ref(),
            row,
            "arxiv_id",
            &self.path,
        )?;
        // `text` is nullable in real shards (null when `status != "ok"`).
        // Tolerate null here; the pipeline filters by `status` so empty text
        // for a non-ok row never reaches the gate.
        let text = read_optional_string_cell(
            batch.column(self.text_col).as_ref(),
            row,
            "text",
            &self.path,
        )?;
        let status = read_string_cell(
            batch.column(self.status_col).as_ref(),
            row,
            "status",
            &self.path,
        )?;

        Ok(InputRow {
            arxiv_id,
            text,
            status,
        })
    }
}

impl Iterator for ShardRowIter {
    type Item = Result<InputRow>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Drain the current batch if we still have rows in it.
            if let Some(batch) = &self.current {
                if self.row_cursor < batch.num_rows() {
                    let res = self.row_from_current();
                    self.row_cursor += 1;
                    return Some(res);
                }
            }
            // Otherwise fetch the next batch; `Ok(false)` means EOF.
            match self.load_next_batch() {
                Ok(true) => continue,
                Ok(false) => return None,
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// Open `path` as a parquet file and return an iterator over
/// `(arxiv_id, text, status)` rows.
///
/// Uses [`ProjectionMask::leaves`] so the unused columns
/// (`stage_timings_us`, `peak_memory_bytes`, etc.) are skipped entirely. The
/// arrow schema is assumed flat for our inputs — for each required column we
/// look up its top-level index via [`Schema::index_of`] and pass it as a leaf
/// index, which matches the `ProjectionMask::leaves` example in the parquet
/// crate (the three string columns have exactly one leaf each, so the
/// arrow-field index equals the leaf index).
pub fn read_shard_rows(
    path: &Path,
) -> Result<Box<dyn Iterator<Item = Result<InputRow>>>> {
    let file = File::open(path)
        .with_context(|| format!("opening parquet file {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading parquet metadata from {}", path.display()))?;

    // Look up indices in the PRE-projection arrow schema to compute the leaf
    // mask. These are positions in the on-disk schema, which is how
    // ProjectionMask::leaves expects to be addressed.
    let pre_schema = builder.schema().clone();
    let arxiv_col_pre = column_index(&pre_schema, "arxiv_id", path)?;
    let text_col_pre = column_index(&pre_schema, "text", path)?;
    let status_col_pre = column_index(&pre_schema, "status", path)?;

    let mask = ProjectionMask::leaves(
        builder.parquet_schema(),
        [arxiv_col_pre, text_col_pre, status_col_pre],
    );

    let reader = builder
        .with_projection(mask)
        .with_batch_size(2048)
        .build()
        .with_context(|| {
            format!("building parquet record batch reader for {}", path.display())
        })?;

    // After projection, the yielded RecordBatches contain ONLY the projected
    // columns, re-indexed contiguously from 0..N-1 in the original schema's
    // relative order. The pre-projection indices (e.g. 5 for `text` on the
    // real schema) would point past the end of a 3-column projected batch and
    // panic, so re-resolve every column by name against the post-projection
    // schema before iterating.
    let projected_schema = reader.schema();
    let arxiv_col = column_index(&projected_schema, "arxiv_id", path)?;
    let text_col = column_index(&projected_schema, "text", path)?;
    let status_col = column_index(&projected_schema, "status", path)?;

    let iter = ShardRowIter {
        reader,
        path: path.to_path_buf(),
        arxiv_col,
        text_col,
        status_col,
        current: None,
        row_cursor: 0,
    };

    Ok(Box::new(iter))
}

/// Build the output arrow schema. Kept as a helper so both writer + reader
/// helpers (and callers like `merge`) can share it.
fn candidate_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("arxiv_id", DataType::Utf8, false),
        Field::new("shard_id", DataType::Utf8, false),
        Field::new("num_matched_paragraphs", DataType::UInt32, false),
    ]))
}

/// Write `rows` to a Snappy-compressed parquet file at `path`, stamped with
/// `shard_id`. Creates the parent directory if missing. When `rows` is empty
/// we still emit a 0-row file with the correct schema — downstream
/// resumability checks `path.exists()` to decide whether a shard has been
/// processed, so the empty file is the "processed, no candidates" marker.
pub fn write_candidates(
    path: &Path,
    shard_id: &str,
    rows: &[CandidateRow],
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("creating parent directory for {}", path.display())
        })?;
    }

    let schema = candidate_schema();

    let arxiv_ids: Vec<&str> = rows.iter().map(|r| r.arxiv_id.as_str()).collect();
    let shard_ids: Vec<&str> = vec![shard_id; rows.len()];
    let counts: Vec<u32> = rows.iter().map(|r| r.num_matched_paragraphs).collect();

    let arxiv_arr = Arc::new(StringArray::from(arxiv_ids));
    let shard_arr = Arc::new(StringArray::from(shard_ids));
    let count_arr = Arc::new(UInt32Array::from(counts));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![arxiv_arr, shard_arr, count_arr],
    )
    .with_context(|| format!("building candidate RecordBatch for {}", path.display()))?;

    let file = File::create(path)
        .with_context(|| format!("creating output parquet file {}", path.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .with_context(|| format!("opening ArrowWriter for {}", path.display()))?;

    // Only write the batch if non-empty — ArrowWriter::write with 0 rows is
    // valid, but skipping it keeps the row-group layout clean and the empty
    // case obviously zero-row at the parquet level.
    if !rows.is_empty() {
        writer
            .write(&batch)
            .with_context(|| format!("writing candidate batch to {}", path.display()))?;
    }
    writer
        .close()
        .with_context(|| format!("closing ArrowWriter for {}", path.display()))?;

    Ok(())
}

/// Read back the `arxiv_id` column from a previously-written candidate shard.
///
/// Used by the `merge` subcommand in `main.rs` to stitch per-shard outputs
/// into a single candidate set. Projects only `arxiv_id` so we avoid decoding
/// `shard_id` / `num_matched_paragraphs`.
pub fn read_candidate_arxiv_ids(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path)
        .with_context(|| format!("opening candidate parquet {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading candidate parquet metadata from {}", path.display()))?;

    // Pre-projection index drives the leaf mask; the post-projection index
    // (resolved below from `reader.schema()`) is what addresses the projected
    // batch — same C2 pattern as in `read_shard_rows`.
    let pre_schema = builder.schema().clone();
    let arxiv_col_pre = column_index(&pre_schema, "arxiv_id", path)?;
    let mask = ProjectionMask::leaves(builder.parquet_schema(), [arxiv_col_pre]);

    let reader = builder
        .with_projection(mask)
        .with_batch_size(2048)
        .build()
        .with_context(|| {
            format!("building reader for candidate parquet {}", path.display())
        })?;

    let projected_schema = reader.schema();
    let arxiv_col = column_index(&projected_schema, "arxiv_id", path)?;

    let mut out: Vec<String> = Vec::new();
    for batch_res in reader {
        let batch = batch_res.with_context(|| {
            format!("reading batch from candidate parquet {}", path.display())
        })?;
        let arr = batch.column(arxiv_col);
        for i in 0..arr.len() {
            out.push(read_string_cell(arr.as_ref(), i, "arxiv_id", path)?);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn write_input_parquet(
        path: &Path,
        schema: Arc<Schema>,
        columns: Vec<Arc<dyn Array>>,
    ) -> Result<()> {
        let batch = RecordBatch::try_new(schema.clone(), columns)?;
        let file = File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    #[test]
    fn writer_roundtrip_with_rows() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("candidates.parquet");
        let rows = vec![
            CandidateRow {
                arxiv_id: "2401.00001".to_string(),
                num_matched_paragraphs: 2,
            },
            CandidateRow {
                arxiv_id: "2401.00002".to_string(),
                num_matched_paragraphs: 1,
            },
            CandidateRow {
                arxiv_id: "2401.00003".to_string(),
                num_matched_paragraphs: 4,
            },
        ];

        write_candidates(&path, "shard-000", &rows).unwrap();
        assert!(path.exists(), "writer must produce an output file");

        let ids = read_candidate_arxiv_ids(&path).unwrap();
        assert_eq!(
            ids,
            vec![
                "2401.00001".to_string(),
                "2401.00002".to_string(),
                "2401.00003".to_string(),
            ],
            "round-trip must preserve row order"
        );
    }

    #[test]
    fn writer_empty_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.parquet");

        write_candidates(&path, "shard-empty", &[]).unwrap();
        assert!(
            path.exists(),
            "empty candidate shard must still produce a file for resumability"
        );

        let ids = read_candidate_arxiv_ids(&path).unwrap();
        assert!(ids.is_empty(), "empty shard must round-trip to empty vec");
    }

    #[test]
    fn reader_rejects_missing_column() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("missing.parquet");

        // Only arxiv_id — no text, no status.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "arxiv_id",
            DataType::Utf8,
            false,
        )]));
        let arxiv = Arc::new(StringArray::from(vec!["2401.00001"])) as Arc<dyn Array>;
        write_input_parquet(&path, schema, vec![arxiv]).unwrap();

        let err = read_shard_rows(&path).err().expect(
            "read_shard_rows must fail when required columns are missing",
        );
        let msg = format!("{err:#}");
        assert!(
            msg.contains("text") || msg.contains("status"),
            "error should mention missing column (text/status), got: {msg}"
        );
    }

    #[test]
    fn reader_reads_full_row() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("input.parquet");

        // Include `extra_col` so the projection must skip it.
        let schema = Arc::new(Schema::new(vec![
            Field::new("arxiv_id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("status", DataType::Utf8, false),
            Field::new("extra_col", DataType::Utf8, false),
        ]));
        let arxiv = Arc::new(StringArray::from(vec!["2401.00001"])) as Arc<dyn Array>;
        let text = Arc::new(StringArray::from(vec![
            "This work was supported by the NSF.",
        ])) as Arc<dyn Array>;
        let status = Arc::new(StringArray::from(vec!["ok"])) as Arc<dyn Array>;
        let extra = Arc::new(StringArray::from(vec!["ignored-payload"])) as Arc<dyn Array>;
        write_input_parquet(&path, schema, vec![arxiv, text, status, extra]).unwrap();

        let iter = read_shard_rows(&path).unwrap();
        let rows: Vec<InputRow> = iter.collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].arxiv_id, "2401.00001");
        assert_eq!(rows[0].text, "This work was supported by the NSF.");
        assert_eq!(rows[0].status, "ok");
    }

    /// Regression for C1: real shards in
    /// `cometadata/arxiv-latex-extract-full-text` use `LargeUtf8` for the
    /// `text` column, which is backed by [`LargeStringArray`] (i64 offsets)
    /// rather than [`StringArray`] (i32 offsets). A naive `downcast_ref::
    /// <StringArray>` on those columns silently returns `None` and the
    /// reader errors on every row. Defensively we also test `arxiv_id` as
    /// `LargeUtf8` so a future shard variant doesn't regress us either.
    #[test]
    fn reader_reads_full_row_with_large_utf8_text() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("input_large_utf8.parquet");

        // `arxiv_id` and `text` as LargeUtf8 — the bug was that text in
        // particular regressed; arxiv_id is included to lock down the
        // defensive coverage on every string column.
        let schema = Arc::new(Schema::new(vec![
            Field::new("arxiv_id", DataType::LargeUtf8, false),
            Field::new("text", DataType::LargeUtf8, false),
            Field::new("status", DataType::Utf8, false),
        ]));
        let arxiv = Arc::new(LargeStringArray::from(vec!["2401.00001"])) as Arc<dyn Array>;
        let text = Arc::new(LargeStringArray::from(vec![
            "This work was supported by the NSF (LargeUtf8).",
        ])) as Arc<dyn Array>;
        let status = Arc::new(StringArray::from(vec!["ok"])) as Arc<dyn Array>;
        write_input_parquet(&path, schema, vec![arxiv, text, status]).unwrap();

        let iter = read_shard_rows(&path).unwrap();
        let rows: Vec<InputRow> = iter.collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].arxiv_id, "2401.00001");
        assert_eq!(
            rows[0].text,
            "This work was supported by the NSF (LargeUtf8)."
        );
        assert_eq!(rows[0].status, "ok");
    }

    /// Regression for C2: column projection re-indexes contiguously, so the
    /// stored column indices must be resolved against the post-projection
    /// schema. With the original schema arranged as
    /// `[arxiv_id, dummy, text, dummy2, status]`, the pre-projection index
    /// for `status` is 4 — but the projected batch only has 3 columns, so
    /// `batch.column(4)` would panic. The fix re-resolves indices against
    /// `reader.schema()`.
    #[test]
    fn reader_reads_full_row_with_extra_column_in_middle() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("input_extra_middle.parquet");

        let schema = Arc::new(Schema::new(vec![
            Field::new("arxiv_id", DataType::Utf8, false),
            Field::new("dummy", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("dummy2", DataType::Utf8, false),
            Field::new("status", DataType::Utf8, false),
        ]));
        let arxiv = Arc::new(StringArray::from(vec!["2401.99999"])) as Arc<dyn Array>;
        let dummy = Arc::new(StringArray::from(vec!["pad-1"])) as Arc<dyn Array>;
        let text = Arc::new(StringArray::from(vec![
            "This work was supported by NIH grant R01-12345.",
        ])) as Arc<dyn Array>;
        let dummy2 = Arc::new(StringArray::from(vec!["pad-2"])) as Arc<dyn Array>;
        let status = Arc::new(StringArray::from(vec!["ok"])) as Arc<dyn Array>;
        write_input_parquet(
            &path,
            schema,
            vec![arxiv, dummy, text, dummy2, status],
        )
        .unwrap();

        let iter = read_shard_rows(&path).unwrap();
        let rows: Vec<InputRow> = iter.collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].arxiv_id, "2401.99999");
        assert_eq!(
            rows[0].text,
            "This work was supported by NIH grant R01-12345."
        );
        assert_eq!(rows[0].status, "ok");
    }

    /// Regression for the null-text case: real shards have `text: null`
    /// whenever extraction failed (`status != "ok"`). The reader must yield
    /// these rows with empty text rather than erroring, so the pipeline can
    /// drop them via the `status == "ok"` filter.
    #[test]
    fn reader_tolerates_null_text_for_failed_rows() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("input_null_text.parquet");

        let schema = Arc::new(Schema::new(vec![
            Field::new("arxiv_id", DataType::Utf8, false),
            Field::new("text", DataType::LargeUtf8, true),
            Field::new("status", DataType::Utf8, false),
        ]));
        let arxiv =
            Arc::new(StringArray::from(vec!["2401.00001", "2401.00002"])) as Arc<dyn Array>;
        let text = Arc::new(LargeStringArray::from(vec![
            Some("This work was supported by the NSF."),
            None,
        ])) as Arc<dyn Array>;
        let status = Arc::new(StringArray::from(vec!["ok", "error"])) as Arc<dyn Array>;
        write_input_parquet(&path, schema, vec![arxiv, text, status]).unwrap();

        let iter = read_shard_rows(&path).unwrap();
        let rows: Vec<InputRow> = iter.collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].status, "ok");
        assert_eq!(rows[0].text, "This work was supported by the NSF.");
        assert_eq!(rows[1].status, "error");
        assert_eq!(rows[1].text, "");
    }
}
