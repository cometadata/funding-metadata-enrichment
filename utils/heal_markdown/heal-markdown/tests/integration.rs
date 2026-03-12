use std::fs;
use std::process::Command;
use tempfile::TempDir;

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
        .args(&[
            "tests/fixtures/",
            "--out-dir",
            output_dir.to_str().unwrap(),
        ])
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
