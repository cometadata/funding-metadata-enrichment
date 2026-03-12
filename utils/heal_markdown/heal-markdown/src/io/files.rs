use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Gather markdown file paths from a file or directory.
///
/// If `path` is a file, returns a single-element vector containing that path.
/// If `path` is a directory, recursively walks it for `*.md` files,
/// excluding any entries where a path component starts with `.`.
/// Results are sorted alphabetically.
pub fn gather_markdown_paths(path: &Path) -> Vec<PathBuf> {
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
                .map(|ext| ext == "md")
                .unwrap_or(false)
        })
        .filter(|e| {
            // Exclude entries where any component (relative to the walk root) starts with '.'
            let rel = e.path().strip_prefix(path).unwrap_or(e.path());
            !rel.components().any(|c| {
                c.as_os_str()
                    .to_string_lossy()
                    .starts_with('.')
            })
        })
        .map(|e| e.into_path())
        .collect();

    paths.sort();
    paths
}

/// Write healed (or original) content to the appropriate destination.
///
/// Routing logic:
/// - If `content` is empty and `exclude_failed` is true, returns `None` (skip writing).
/// - If `content` is empty and `exclude_failed` is false, writes `original_content` instead.
/// - Destination is determined by:
///   - `in_place`: overwrite the source file
///   - `out_dir` with optional `failure_category`: write to structured output directory
///   - Otherwise: write to `{source_stem}-clean.md` in the same directory as source
pub fn write_output(
    source: &Path,
    base_path: &Path,
    content: &str,
    original_content: &str,
    in_place: bool,
    out_dir: Option<&Path>,
    exclude_failed: bool,
    failure_category: Option<&str>,
) -> Option<PathBuf> {
    let output_content = if content.is_empty() {
        if exclude_failed {
            return None;
        }
        original_content
    } else {
        content
    };

    let destination = if in_place {
        source.to_path_buf()
    } else if let Some(out) = out_dir {
        let relative = source.strip_prefix(base_path).unwrap_or(source);
        if let Some(category) = failure_category {
            out.join("failed").join(category).join(relative)
        } else {
            out.join(relative)
        }
    } else {
        // Write to {source_stem}-clean.md in the same directory as source
        let parent = source.parent().unwrap_or_else(|| Path::new("."));
        let stem = source
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        parent.join(format!("{}-clean.md", stem))
    };

    // Create parent directories if needed
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).ok();
    }

    fs::write(&destination, output_content).ok();
    Some(destination)
}

/// Read a file with lossy UTF-8 conversion.
///
/// Reads the raw bytes and converts to a `String` using `from_utf8_lossy`,
/// replacing invalid UTF-8 sequences with the replacement character.
pub fn read_file_lossy(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_default();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_gather_markdown_single_file() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.md");
        fs::write(&file, "# Test").unwrap();
        let paths = gather_markdown_paths(&file);
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn test_gather_markdown_directory() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.md"), "# A").unwrap();
        fs::write(dir.path().join("b.md"), "# B").unwrap();
        fs::write(dir.path().join("c.txt"), "Not markdown").unwrap();
        let paths = gather_markdown_paths(dir.path());
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_gather_excludes_dotdirs() {
        let dir = TempDir::new().unwrap();
        let hidden = dir.path().join(".hidden");
        fs::create_dir(&hidden).unwrap();
        fs::write(hidden.join("secret.md"), "# Hidden").unwrap();
        fs::write(dir.path().join("visible.md"), "# Visible").unwrap();
        let paths = gather_markdown_paths(dir.path());
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn test_gather_sorted_order() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("z.md"), "# Z").unwrap();
        fs::write(dir.path().join("a.md"), "# A").unwrap();
        fs::write(dir.path().join("m.md"), "# M").unwrap();
        let paths = gather_markdown_paths(dir.path());
        assert_eq!(paths.len(), 3);
        assert!(paths[0] < paths[1]);
        assert!(paths[1] < paths[2]);
    }

    #[test]
    fn test_write_output_in_place() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.md");
        fs::write(&file, "original").unwrap();
        let result = write_output(
            &file,
            dir.path(),
            "healed",
            "original",
            true,
            None,
            false,
            None,
        );
        assert_eq!(result, Some(file.clone()));
        assert_eq!(fs::read_to_string(&file).unwrap(), "healed");
    }

    #[test]
    fn test_write_output_out_dir() {
        let dir = TempDir::new().unwrap();
        let source = dir.path().join("sub").join("test.md");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        fs::write(&source, "original").unwrap();
        let out_dir = dir.path().join("output");
        let result = write_output(
            &source,
            dir.path(),
            "healed",
            "original",
            false,
            Some(&out_dir),
            false,
            None,
        );
        let expected = out_dir.join("sub").join("test.md");
        assert_eq!(result, Some(expected.clone()));
        assert_eq!(fs::read_to_string(&expected).unwrap(), "healed");
    }

    #[test]
    fn test_write_output_out_dir_with_failure() {
        let dir = TempDir::new().unwrap();
        let source = dir.path().join("test.md");
        fs::write(&source, "original").unwrap();
        let out_dir = dir.path().join("output");
        let result = write_output(
            &source,
            dir.path(),
            "healed",
            "original",
            false,
            Some(&out_dir),
            false,
            Some("parse_error"),
        );
        let expected = out_dir.join("failed").join("parse_error").join("test.md");
        assert_eq!(result, Some(expected.clone()));
        assert_eq!(fs::read_to_string(&expected).unwrap(), "healed");
    }

    #[test]
    fn test_write_output_default_clean_suffix() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.md");
        fs::write(&file, "original").unwrap();
        let result = write_output(
            &file,
            dir.path(),
            "healed",
            "original",
            false,
            None,
            false,
            None,
        );
        let expected = dir.path().join("test-clean.md");
        assert_eq!(result, Some(expected.clone()));
        assert_eq!(fs::read_to_string(&expected).unwrap(), "healed");
    }

    #[test]
    fn test_write_output_exclude_failed_empty_content() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.md");
        fs::write(&file, "original").unwrap();
        let result = write_output(
            &file,
            dir.path(),
            "",
            "original",
            false,
            None,
            true,
            None,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_write_output_fallback_to_original() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.md");
        fs::write(&file, "original").unwrap();
        let result = write_output(
            &file,
            dir.path(),
            "",
            "original",
            false,
            None,
            false,
            None,
        );
        let expected = dir.path().join("test-clean.md");
        assert_eq!(result, Some(expected.clone()));
        assert_eq!(fs::read_to_string(&expected).unwrap(), "original");
    }

    #[test]
    fn test_read_file_lossy() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.md");
        fs::write(&file, "Hello, world!").unwrap();
        let content = read_file_lossy(&file);
        assert_eq!(content, "Hello, world!");
    }

    #[test]
    fn test_read_file_lossy_invalid_utf8() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.bin");
        // Write bytes with invalid UTF-8 sequence
        fs::write(&file, b"Hello \xff world").unwrap();
        let content = read_file_lossy(&file);
        assert!(content.contains("Hello"));
        assert!(content.contains("world"));
    }
}
