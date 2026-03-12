use std::fs;
use std::path::{Path, PathBuf};

use arrow::array::{AsArray, BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use std::sync::Arc;
use walkdir::WalkDir;

use crate::types::RestorationResult;

/// Candidate column names for filenames, checked in priority order (case-insensitive).
const FILENAME_CANDIDATES: &[&str] = &["file_name", "filename", "relative_path", "name", "path"];

/// Candidate column names for content, checked in priority order (case-insensitive).
const CONTENT_CANDIDATES: &[&str] = &["content", "text", "markdown", "md", "body"];

/// Find the first column in the schema matching any of the candidates (case-insensitive).
/// Returns the column index if found.
fn find_column(schema: &Schema, candidates: &[&str]) -> Option<usize> {
    for candidate in candidates {
        for (i, field) in schema.fields().iter().enumerate() {
            if field.name().eq_ignore_ascii_case(candidate) {
                return Some(i);
            }
        }
    }
    None
}

/// Read a Parquet file and auto-detect filename and content columns.
///
/// Returns `(filenames, contents)` where each is a `Vec<String>`.
/// If no filename column is found among candidates, generates `row_0.md`, `row_1.md`, etc.
/// If no content column is found, returns an error.
pub fn read_parquet_input(
    path: &Path,
) -> Result<(Vec<String>, Vec<String>), Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.build()?;

    // Collect all record batches
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;

    let filename_col_idx = find_column(&schema, FILENAME_CANDIDATES);
    let content_col_idx = find_column(&schema, CONTENT_CANDIDATES)
        .ok_or("No content column found. Expected one of: content, text, markdown, md, body")?;

    let mut filenames = Vec::new();
    let mut contents = Vec::new();
    let mut row_counter: usize = 0;

    for batch in &batches {
        let content_array = batch.column(content_col_idx).as_string::<i32>();

        let filename_array = filename_col_idx.map(|idx| batch.column(idx).as_string::<i32>());

        for i in 0..batch.num_rows() {
            let content = content_array.value(i).to_string();
            contents.push(content);

            let filename = if let Some(ref fnames) = filename_array {
                fnames.value(i).to_string()
            } else {
                let name = format!("row_{}.md", row_counter);
                name
            };
            filenames.push(filename);
            row_counter += 1;
        }
    }

    Ok((filenames, contents))
}

/// Write successful restoration results to a Parquet file.
///
/// Schema: file_name (Utf8), content (Utf8), warnings (Utf8), success (Boolean),
///         original_size (Int64), output_size (Int64)
pub fn write_parquet_output(
    results: &[RestorationResult],
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("file_name", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("warnings", DataType::Utf8, true),
        Field::new("success", DataType::Boolean, false),
        Field::new("original_size", DataType::Int64, false),
        Field::new("output_size", DataType::Int64, false),
    ]));

    let file_names: Vec<&str> = results.iter().map(|r| r.source.as_str()).collect();
    let contents: Vec<String> = results
        .iter()
        .map(|r| r.cleaned_content.clone().unwrap_or_default())
        .collect();
    let contents_ref: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();
    let warnings: Vec<String> = results
        .iter()
        .map(|r| r.warnings.join("; "))
        .collect();
    let warnings_ref: Vec<&str> = warnings.iter().map(|s| s.as_str()).collect();
    let successes: Vec<bool> = results.iter().map(|r| r.success).collect();
    let original_sizes: Vec<i64> = results.iter().map(|r| r.original_size as i64).collect();
    let output_sizes: Vec<i64> = results.iter().map(|r| r.output_size as i64).collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(file_names)),
            Arc::new(StringArray::from(contents_ref)),
            Arc::new(StringArray::from(warnings_ref)),
            Arc::new(BooleanArray::from(successes)),
            Arc::new(Int64Array::from(original_sizes)),
            Arc::new(Int64Array::from(output_sizes)),
        ],
    )?;

    // Create parent directories if needed
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = fs::File::create(output_path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

/// Convert a RecoveryConfidence to a string representation.
fn confidence_to_string(conf: &crate::types::RecoveryConfidence) -> &'static str {
    match conf {
        crate::types::RecoveryConfidence::None_ => "none",
        crate::types::RecoveryConfidence::Low => "low",
        crate::types::RecoveryConfidence::Medium => "medium",
        crate::types::RecoveryConfidence::High => "high",
    }
}

/// Write failed restoration results to a Parquet file.
///
/// Schema: file_name (Utf8), original_content (Utf8), cleaned_content (Utf8),
///         failure_category (Utf8), issues (Utf8), recovery_attempted (Boolean),
///         recovery_confidence (Utf8), recoverable (Boolean), validation_error (Utf8)
pub fn write_failed_parquet(
    results: &[RestorationResult],
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("file_name", DataType::Utf8, false),
        Field::new("original_content", DataType::Utf8, true),
        Field::new("cleaned_content", DataType::Utf8, true),
        Field::new("failure_category", DataType::Utf8, true),
        Field::new("issues", DataType::Utf8, true),
        Field::new("recovery_attempted", DataType::Boolean, false),
        Field::new("recovery_confidence", DataType::Utf8, true),
        Field::new("recoverable", DataType::Boolean, false),
        Field::new("validation_error", DataType::Utf8, true),
    ]));

    let file_names: Vec<&str> = results.iter().map(|r| r.source.as_str()).collect();

    let original_contents: Vec<Option<&str>> = results
        .iter()
        .map(|r| r.original_content.as_deref())
        .collect();

    let cleaned_contents: Vec<Option<&str>> = results
        .iter()
        .map(|r| r.cleaned_content.as_deref())
        .collect();

    let failure_categories: Vec<Option<String>> = results
        .iter()
        .map(|r| r.failure_analysis.as_ref().map(|fa| fa.category.to_string()))
        .collect();
    let failure_categories_ref: Vec<Option<&str>> = failure_categories
        .iter()
        .map(|s| s.as_deref())
        .collect();

    let issues: Vec<Option<String>> = results
        .iter()
        .map(|r| r.failure_analysis.as_ref().map(|fa| fa.issues.join("; ")))
        .collect();
    let issues_ref: Vec<Option<&str>> = issues.iter().map(|s| s.as_deref()).collect();

    let recovery_attempted: Vec<bool> = results.iter().map(|r| r.recovery_attempted).collect();

    let recovery_confidence: Vec<Option<&str>> = results
        .iter()
        .map(|r| {
            r.failure_analysis
                .as_ref()
                .and_then(|fa| fa.recovery_confidence.as_ref())
                .map(confidence_to_string)
        })
        .collect();

    let recoverable: Vec<bool> = results
        .iter()
        .map(|r| {
            r.failure_analysis
                .as_ref()
                .map(|fa| fa.recoverable)
                .unwrap_or(false)
        })
        .collect();

    let validation_errors: Vec<Option<&str>> = results
        .iter()
        .map(|r| r.validation_error.as_deref())
        .collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(file_names)),
            Arc::new(StringArray::from(original_contents)),
            Arc::new(StringArray::from(cleaned_contents)),
            Arc::new(StringArray::from(failure_categories_ref)),
            Arc::new(StringArray::from(issues_ref)),
            Arc::new(BooleanArray::from(recovery_attempted)),
            Arc::new(StringArray::from(recovery_confidence)),
            Arc::new(BooleanArray::from(recoverable)),
            Arc::new(StringArray::from(validation_errors)),
        ],
    )?;

    // Create parent directories if needed
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = fs::File::create(output_path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

/// Check if a path refers to Parquet input.
///
/// Returns true if the path has a `.parquet` extension, or if it is a directory
/// containing at least one `.parquet` file.
pub fn is_parquet_input(path: &Path) -> bool {
    if path.is_file() {
        return path
            .extension()
            .map(|ext| ext == "parquet")
            .unwrap_or(false);
    }

    if path.is_dir() {
        return fs::read_dir(path)
            .map(|entries| {
                entries.filter_map(|e| e.ok()).any(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "parquet")
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);
    }

    false
}

/// Gather Parquet file paths from a file or directory.
///
/// If `path` is a file, returns a single-element vector containing that path.
/// If `path` is a directory, recursively walks it for `*.parquet` files,
/// excluding any entries where a path component starts with `.`.
/// Results are sorted alphabetically.
pub fn gather_parquet_paths(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }

    let mut paths: Vec<PathBuf> = WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "parquet")
                .unwrap_or(false)
        })
        .filter(|e| {
            let rel = e.path().strip_prefix(path).unwrap_or(e.path());
            !rel.components().any(|c| c.as_os_str().to_string_lossy().starts_with('.'))
        })
        .map(|e| e.into_path())
        .collect();

    paths.sort();
    paths
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::fs::File;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn create_test_parquet(dir: &std::path::Path, filename: &str) -> std::path::PathBuf {
        let path = dir.join(filename);
        let schema = Arc::new(Schema::new(vec![
            Field::new("file_name", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["doc1.md", "doc2.md"])),
                Arc::new(StringArray::from(vec!["# Doc 1\nContent", "# Doc 2\nContent"])),
            ],
        )
        .unwrap();
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        path
    }

    fn create_test_parquet_with_columns(
        dir: &std::path::Path,
        filename: &str,
        filename_col: Option<&str>,
        content_col: &str,
    ) -> std::path::PathBuf {
        let path = dir.join(filename);
        let mut fields = Vec::new();
        if let Some(fname_col) = filename_col {
            fields.push(Field::new(fname_col, DataType::Utf8, false));
        }
        fields.push(Field::new(content_col, DataType::Utf8, false));
        let schema = Arc::new(Schema::new(fields));

        let mut columns: Vec<Arc<dyn arrow::array::Array>> = Vec::new();
        if filename_col.is_some() {
            columns.push(Arc::new(StringArray::from(vec!["a.md", "b.md"])));
        }
        columns.push(Arc::new(StringArray::from(vec![
            "# A\nContent A",
            "# B\nContent B",
        ])));

        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        path
    }

    #[test]
    fn test_read_parquet_detects_columns() {
        let dir = TempDir::new().unwrap();
        let path = create_test_parquet(dir.path(), "test.parquet");
        let (filenames, contents) = read_parquet_input(&path).unwrap();
        assert_eq!(filenames.len(), 2);
        assert_eq!(contents.len(), 2);
        assert_eq!(filenames[0], "doc1.md");
        assert_eq!(filenames[1], "doc2.md");
        assert_eq!(contents[0], "# Doc 1\nContent");
        assert_eq!(contents[1], "# Doc 2\nContent");
    }

    #[test]
    fn test_read_parquet_alternate_content_column() {
        let dir = TempDir::new().unwrap();
        let path =
            create_test_parquet_with_columns(dir.path(), "test.parquet", Some("filename"), "text");
        let (filenames, contents) = read_parquet_input(&path).unwrap();
        assert_eq!(filenames.len(), 2);
        assert_eq!(filenames[0], "a.md");
        assert_eq!(contents[0], "# A\nContent A");
    }

    #[test]
    fn test_read_parquet_no_filename_column() {
        let dir = TempDir::new().unwrap();
        let path =
            create_test_parquet_with_columns(dir.path(), "test.parquet", None, "markdown");
        let (filenames, contents) = read_parquet_input(&path).unwrap();
        assert_eq!(filenames.len(), 2);
        assert_eq!(filenames[0], "row_0.md");
        assert_eq!(filenames[1], "row_1.md");
        assert_eq!(contents.len(), 2);
    }

    #[test]
    fn test_read_parquet_no_content_column_errors() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("other", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["1"])),
                Arc::new(StringArray::from(vec!["data"])),
            ],
        )
        .unwrap();
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let result = read_parquet_input(&path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No content column found"));
    }

    #[test]
    fn test_is_parquet_input_file() {
        let dir = TempDir::new().unwrap();
        let path = create_test_parquet(dir.path(), "test.parquet");
        assert!(is_parquet_input(&path));
    }

    #[test]
    fn test_is_parquet_input_nonexistent() {
        let dir = TempDir::new().unwrap();
        assert!(!is_parquet_input(&dir.path().join("nonexistent.md")));
    }

    #[test]
    fn test_is_parquet_input_md_file() {
        let dir = TempDir::new().unwrap();
        let md_path = dir.path().join("test.md");
        fs::write(&md_path, "# Hello").unwrap();
        assert!(!is_parquet_input(&md_path));
    }

    #[test]
    fn test_is_parquet_input_directory_with_parquet() {
        let dir = TempDir::new().unwrap();
        create_test_parquet(dir.path(), "data.parquet");
        assert!(is_parquet_input(dir.path()));
    }

    #[test]
    fn test_is_parquet_input_directory_without_parquet() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("data.csv"), "a,b\n1,2").unwrap();
        assert!(!is_parquet_input(dir.path()));
    }

    #[test]
    fn test_gather_parquet_paths_single_file() {
        let dir = TempDir::new().unwrap();
        let path = create_test_parquet(dir.path(), "single.parquet");
        let paths = gather_parquet_paths(&path);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], path);
    }

    #[test]
    fn test_gather_parquet_paths_directory() {
        let dir = TempDir::new().unwrap();
        create_test_parquet(dir.path(), "a.parquet");
        create_test_parquet(dir.path(), "b.parquet");
        fs::write(dir.path().join("c.csv"), "not parquet").unwrap();
        let paths = gather_parquet_paths(dir.path());
        assert_eq!(paths.len(), 2);
        // Should be sorted
        assert!(paths[0] < paths[1]);
    }

    #[test]
    fn test_gather_parquet_paths_excludes_dotdirs() {
        let dir = TempDir::new().unwrap();
        let hidden = dir.path().join(".hidden");
        fs::create_dir(&hidden).unwrap();
        create_test_parquet(&hidden, "secret.parquet");
        create_test_parquet(dir.path(), "visible.parquet");
        let paths = gather_parquet_paths(dir.path());
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn test_gather_parquet_paths_recursive() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("subdir");
        fs::create_dir(&sub).unwrap();
        create_test_parquet(dir.path(), "root.parquet");
        create_test_parquet(&sub, "nested.parquet");
        let paths = gather_parquet_paths(dir.path());
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_write_parquet_output() {
        let dir = TempDir::new().unwrap();
        let output_path = dir.path().join("output.parquet");
        let results = vec![crate::types::RestorationResult {
            source: "doc1.md".into(),
            destination: None,
            warnings: vec!["warn1".into()],
            success: true,
            original_size: 100,
            output_size: 90,
            validation_error: None,
            failure_analysis: None,
            recovery_attempted: false,
            cleaned_content: Some("cleaned".into()),
            original_content: Some("original".into()),
        }];
        write_parquet_output(&results, &output_path).unwrap();
        assert!(output_path.exists());

        // Read it back and verify
        let (filenames, contents) = {
            let file = File::open(&output_path).unwrap();
            let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
            let reader = builder.build().unwrap();
            let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().unwrap();
            assert_eq!(batches.len(), 1);
            let batch = &batches[0];
            let fnames = batch.column(0).as_string::<i32>();
            let conts = batch.column(1).as_string::<i32>();
            (
                fnames.value(0).to_string(),
                conts.value(0).to_string(),
            )
        };
        assert_eq!(filenames, "doc1.md");
        assert_eq!(contents, "cleaned");
    }

    #[test]
    fn test_write_parquet_output_empty_cleaned_content() {
        let dir = TempDir::new().unwrap();
        let output_path = dir.path().join("output.parquet");
        let results = vec![crate::types::RestorationResult {
            source: "doc.md".into(),
            destination: None,
            warnings: vec![],
            success: false,
            original_size: 50,
            output_size: 0,
            validation_error: None,
            failure_analysis: None,
            recovery_attempted: false,
            cleaned_content: None,
            original_content: Some("original".into()),
        }];
        write_parquet_output(&results, &output_path).unwrap();
        assert!(output_path.exists());
    }

    #[test]
    fn test_write_failed_parquet() {
        let dir = TempDir::new().unwrap();
        let output_path = dir.path().join("failed.parquet");
        let results = vec![crate::types::RestorationResult {
            source: "bad.md".into(),
            destination: None,
            warnings: vec!["problem".into()],
            success: false,
            original_size: 200,
            output_size: 0,
            validation_error: Some("invalid structure".into()),
            failure_analysis: Some(crate::types::FailureAnalysis {
                category: crate::types::FailureCategory::MalformedStructure,
                issues: vec!["broken headers".into(), "missing content".into()],
                recoverable: false,
                recovery_attempted: true,
                recovery_confidence: Some(crate::types::RecoveryConfidence::Low),
            }),
            recovery_attempted: true,
            cleaned_content: Some("partial".into()),
            original_content: Some("original bad content".into()),
        }];
        write_failed_parquet(&results, &output_path).unwrap();
        assert!(output_path.exists());

        // Read it back and verify schema
        let file = File::open(&output_path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let schema = builder.schema().clone();
        assert_eq!(schema.fields().len(), 9);
        assert_eq!(schema.field(0).name(), "file_name");
        assert_eq!(schema.field(3).name(), "failure_category");
        assert_eq!(schema.field(5).name(), "recovery_attempted");
        assert_eq!(schema.field(7).name(), "recoverable");

        let reader = builder.build().unwrap();
        let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 1);

        // Verify some values
        let category = batch
            .column(3)
            .as_string::<i32>();
        assert_eq!(category.value(0), "malformed");

        let issues = batch
            .column(4)
            .as_string::<i32>();
        assert_eq!(issues.value(0), "broken headers; missing content");

        let confidence = batch
            .column(6)
            .as_string::<i32>();
        assert_eq!(confidence.value(0), "low");
    }

    #[test]
    fn test_write_failed_parquet_no_failure_analysis() {
        let dir = TempDir::new().unwrap();
        let output_path = dir.path().join("failed.parquet");
        let results = vec![crate::types::RestorationResult {
            source: "doc.md".into(),
            destination: None,
            warnings: vec![],
            success: false,
            original_size: 100,
            output_size: 0,
            validation_error: None,
            failure_analysis: None,
            recovery_attempted: false,
            cleaned_content: None,
            original_content: None,
        }];
        write_failed_parquet(&results, &output_path).unwrap();
        assert!(output_path.exists());
    }
}
