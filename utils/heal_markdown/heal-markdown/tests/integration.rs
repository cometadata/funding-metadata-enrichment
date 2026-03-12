use std::fs;
use std::process::Command;
use std::sync::Arc;
use tempfile::TempDir;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;

fn create_test_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("file_name", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["doc1.md", "doc2.md"])),
            Arc::new(StringArray::from(vec![
                "# Doc 1\n\nThe re-\nsearch was good.\n\n\u{2022} Item one\n\u{25cf} Item two\n",
                "# Doc 2\n\nNormal content here.\n\nPage 5 of 10\n",
            ])),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn test_process_simple_markdown_file() {
    let dir = TempDir::new().unwrap();
    let output_dir = dir.path().join("output");

    let status = Command::new(env!("CARGO_BIN_EXE_heal-markdown"))
        .args(&[
            "tests/fixtures/simple.md",
            "--out-dir",
            output_dir.to_str().unwrap(),
        ])
        .status()
        .unwrap();

    assert!(status.success());
    let output_file = output_dir.join("simple.md");
    assert!(output_file.exists());
    let content = fs::read_to_string(&output_file).unwrap();
    assert!(content.contains("Title"));
    assert!(content.contains("paragraph"));
}

#[test]
fn test_process_messy_markdown_file() {
    let dir = TempDir::new().unwrap();
    let output_dir = dir.path().join("output");

    let status = Command::new(env!("CARGO_BIN_EXE_heal-markdown"))
        .args(&[
            "tests/fixtures/messy.md",
            "--out-dir",
            output_dir.to_str().unwrap(),
        ])
        .status()
        .unwrap();

    assert!(status.success());
    let output_file = output_dir.join("messy.md");
    assert!(output_file.exists());
    let content = fs::read_to_string(&output_file).unwrap();
    // Hyphenation should be fixed
    assert!(
        content.contains("research"),
        "Expected 'research' in output, got:\n{}",
        content
    );
    // Bullets should be normalized (pulldown-cmark may use * instead of -)
    assert!(
        content.contains("First item") || content.contains("Second item"),
        "Expected bullet items in output, got:\n{}",
        content
    );
    // Footer should be removed
    assert!(
        !content.contains("Page 3 of 10"),
        "Expected footer 'Page 3 of 10' to be removed, but found it in:\n{}",
        content
    );
}

#[test]
fn test_process_directory() {
    let dir = TempDir::new().unwrap();
    let output_dir = dir.path().join("output");

    let status = Command::new(env!("CARGO_BIN_EXE_heal-markdown"))
        .args(&["tests/fixtures/", "--out-dir", output_dir.to_str().unwrap()])
        .status()
        .unwrap();

    assert!(status.success());
}

#[test]
fn test_exclude_failed() {
    let dir = TempDir::new().unwrap();
    let output_dir = dir.path().join("output");

    let status = Command::new(env!("CARGO_BIN_EXE_heal-markdown"))
        .args(&[
            "tests/fixtures/",
            "--out-dir",
            output_dir.to_str().unwrap(),
            "--exclude-failed",
        ])
        .status()
        .unwrap();

    assert!(status.success());
}

#[test]
fn test_parquet_to_parquet() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("input.parquet");
    create_test_parquet(&input);
    let output_dir = dir.path().join("output");

    let status = Command::new(env!("CARGO_BIN_EXE_heal-markdown"))
        .args(&[
            input.to_str().unwrap(),
            "--out-dir",
            output_dir.to_str().unwrap(),
            "--output-format",
            "parquet",
        ])
        .status()
        .unwrap();

    assert!(status.success());
    // The CLI writes {stem}-healed.parquet into the output directory
    let entries: Vec<_> = std::fs::read_dir(&output_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "parquet"))
        .collect();
    assert!(
        !entries.is_empty(),
        "Expected at least one parquet output file"
    );
}

#[test]
fn test_parquet_to_both() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("input.parquet");
    create_test_parquet(&input);
    let output_dir = dir.path().join("output");

    let status = Command::new(env!("CARGO_BIN_EXE_heal-markdown"))
        .args(&[
            input.to_str().unwrap(),
            "--out-dir",
            output_dir.to_str().unwrap(),
            "--output-format",
            "both",
        ])
        .status()
        .unwrap();

    assert!(status.success());
    // Check parquet output exists
    let parquet_files: Vec<_> = std::fs::read_dir(&output_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "parquet"))
        .collect();
    assert!(!parquet_files.is_empty(), "Expected parquet output");

    // Check markdown output exists
    assert!(
        output_dir.join("doc1.md").exists(),
        "Expected doc1.md markdown output"
    );
    assert!(
        output_dir.join("doc2.md").exists(),
        "Expected doc2.md markdown output"
    );
}
