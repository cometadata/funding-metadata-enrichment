mod dictionary;
mod failure;
mod heuristics;
mod io;
mod markdown;
mod pipeline;
mod recovery;
mod types;
mod validation;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use clap::Parser;
use rayon::prelude::*;

use failure::categorize_failure;
use io::files::{gather_markdown_paths, read_file_lossy, write_output};
use io::parquet_io::{
    gather_parquet_paths, is_parquet_input, read_parquet_input, write_failed_parquet,
    write_parquet_output,
};
use pipeline::restore_markdown;
use recovery::split_concatenated_text;
use types::{RecoveryConfidence, RestorationOptions, RestorationResult};
use validation::validate_input_quality;

#[derive(Parser)]
#[command(name = "heal-markdown")]
#[command(about = "Restore PDF-extracted Markdown into readable CommonMark.")]
struct Cli {
    /// Markdown file, directory, or Parquet file to process
    path: PathBuf,

    /// Overwrite files instead of writing alongside originals
    #[arg(long)]
    in_place: bool,

    /// Directory to write cleaned files
    #[arg(long)]
    out_dir: Option<PathBuf>,

    /// Lines per page for header/footer detection
    #[arg(long, default_value_t = 50)]
    page_lines: usize,

    /// Frequency threshold for repeated line removal
    #[arg(long, default_value_t = 0.8)]
    repeat_threshold: f64,

    /// Max words for repeated line candidates
    #[arg(long, default_value_t = 12)]
    max_header_words: usize,

    /// No-op, accepted for compatibility (table repair deferred)
    #[arg(long)]
    skip_tables: bool,

    /// Remove [12]-style bracketed citations
    #[arg(long)]
    strip_citations: bool,

    /// Exclude files that convert to zero bytes
    #[arg(long)]
    exclude_failed: bool,

    /// Output format: markdown, parquet, or both
    #[arg(long, value_parser = ["markdown", "parquet", "both"])]
    output_format: Option<String>,

    /// Path for failed records Parquet file
    #[arg(long)]
    failed_output: Option<PathBuf>,

    /// Number of parallel threads
    #[arg(long)]
    threads: Option<usize>,
}

/// Process a single document through validation, recovery, and restoration.
fn process_document(
    filename: &str,
    content: &str,
    options: &RestorationOptions,
    exclude_failed: bool,
) -> RestorationResult {
    let original_size = content.len();

    // Step 1: Validate input quality
    let (is_valid, validation_msg) = validate_input_quality(content);

    if !is_valid {
        // Step 2: Try word boundary recovery
        let recovery_result = split_concatenated_text(content, RecoveryConfidence::Medium);

        // Step 3: Categorize the failure
        let failure_analysis = categorize_failure(
            content,
            Some(validation_msg.clone()),
            Some(&recovery_result),
        );

        if recovery_result.changes_made > 0 {
            // Recovery made some changes -- use recovered text through the pipeline
            let (cleaned, warnings) = restore_markdown(&recovery_result.text, options);
            let output_size = cleaned.len();

            if exclude_failed && output_size == 0 {
                return RestorationResult {
                    source: filename.to_string(),
                    destination: None,
                    warnings,
                    success: false,
                    original_size,
                    output_size: 0,
                    validation_error: Some(validation_msg),
                    failure_analysis: Some(failure_analysis),
                    recovery_attempted: true,
                    cleaned_content: None,
                    original_content: Some(content.to_string()),
                };
            }

            return RestorationResult {
                source: filename.to_string(),
                destination: None,
                warnings,
                success: true,
                original_size,
                output_size,
                validation_error: Some(validation_msg),
                failure_analysis: Some(failure_analysis),
                recovery_attempted: true,
                cleaned_content: Some(cleaned),
                original_content: Some(content.to_string()),
            };
        }

        // Recovery did not help
        return RestorationResult {
            source: filename.to_string(),
            destination: None,
            warnings: vec![],
            success: false,
            original_size,
            output_size: 0,
            validation_error: Some(validation_msg),
            failure_analysis: Some(failure_analysis),
            recovery_attempted: true,
            cleaned_content: None,
            original_content: Some(content.to_string()),
        };
    }

    // Validation passed -- run the restoration pipeline
    let (cleaned, warnings) = restore_markdown(content, options);
    let output_size = cleaned.len();

    if exclude_failed && output_size == 0 {
        return RestorationResult {
            source: filename.to_string(),
            destination: None,
            warnings,
            success: false,
            original_size,
            output_size: 0,
            validation_error: None,
            failure_analysis: None,
            recovery_attempted: false,
            cleaned_content: None,
            original_content: Some(content.to_string()),
        };
    }

    RestorationResult {
        source: filename.to_string(),
        destination: None,
        warnings,
        success: true,
        original_size,
        output_size,
        validation_error: None,
        failure_analysis: None,
        recovery_attempted: false,
        cleaned_content: Some(cleaned),
        original_content: Some(content.to_string()),
    }
}

/// Print a summary of all processing results.
fn print_summary(results: &[RestorationResult]) {
    let total = results.len();
    let successful = results.iter().filter(|r| r.success).count();
    let failed = total - successful;
    let total_warnings: usize = results.iter().map(|r| r.warnings.len()).sum();

    println!("\n=== Summary ===");
    println!("Total files: {}", total);
    println!("Successful: {}", successful);
    println!("Failed: {}", failed);
    println!("Warnings: {}", total_warnings);

    if failed > 0 {
        let mut category_counts: HashMap<String, usize> = HashMap::new();
        for r in results.iter().filter(|r| !r.success) {
            let category = r
                .failure_analysis
                .as_ref()
                .map(|fa| fa.category.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            *category_counts.entry(category).or_insert(0) += 1;
        }

        println!("\nFailure categories:");
        let mut sorted_categories: Vec<_> = category_counts.into_iter().collect();
        sorted_categories.sort_by(|a, b| b.1.cmp(&a.1));
        for (category, count) in sorted_categories {
            println!("  {}: {}", category, count);
        }
    }
}

fn main() {
    let cli = Cli::parse();

    // Configure thread pool if specified
    if let Some(n) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .unwrap();
    }

    let options = RestorationOptions {
        page_length_estimate: cli.page_lines,
        repeat_threshold: cli.repeat_threshold,
        max_header_words: cli.max_header_words,
        strip_citations: cli.strip_citations,
    };

    let all_results: Vec<RestorationResult>;

    if is_parquet_input(&cli.path) {
        // === Parquet branch ===
        let output_format = cli
            .output_format
            .clone()
            .unwrap_or_else(|| "parquet".to_string());

        // Validate: if output includes markdown, --out-dir is required
        if (output_format == "markdown" || output_format == "both") && cli.out_dir.is_none() {
            eprintln!("Error: --out-dir is required when output format includes markdown");
            std::process::exit(1);
        }

        let parquet_files = gather_parquet_paths(&cli.path);
        if parquet_files.is_empty() {
            eprintln!("No parquet files found at {:?}", cli.path);
            std::process::exit(1);
        }

        let mut collected_results: Vec<RestorationResult> = Vec::new();
        let progress = AtomicUsize::new(0);

        for parquet_path in &parquet_files {
            eprintln!("Processing: {}", parquet_path.display());

            let (filenames, contents) = match read_parquet_input(parquet_path) {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("Error reading {}: {}", parquet_path.display(), e);
                    continue;
                }
            };

            let results: Vec<RestorationResult> = filenames
                .par_iter()
                .zip(contents.par_iter())
                .map(|(f, c)| {
                    let result = process_document(f, c, &options, cli.exclude_failed);
                    let count = progress.fetch_add(1, Ordering::Relaxed) + 1;
                    if count % 1000 == 0 {
                        eprintln!("Processed {} documents...", count);
                    }
                    result
                })
                .collect();

            // Write parquet output for success results
            if output_format == "parquet" || output_format == "both" {
                let success_results: Vec<_> =
                    results.iter().filter(|r| r.success).cloned().collect();
                if !success_results.is_empty() {
                    let stem = parquet_path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy();
                    let out_path = if let Some(ref out_dir) = cli.out_dir {
                        out_dir.join(format!("{}-healed.parquet", stem))
                    } else {
                        let parent = parquet_path
                            .parent()
                            .unwrap_or_else(|| std::path::Path::new("."));
                        parent.join(format!("{}-healed.parquet", stem))
                    };
                    if let Err(e) = write_parquet_output(&success_results, &out_path) {
                        eprintln!("Error writing output parquet: {}", e);
                    } else {
                        eprintln!("Wrote: {}", out_path.display());
                    }
                }
            }

            // Write failed parquet if requested
            if let Some(ref failed_path) = cli.failed_output {
                let failed_results: Vec<_> =
                    results.iter().filter(|r| !r.success).cloned().collect();
                if !failed_results.is_empty() {
                    if let Err(e) = write_failed_parquet(&failed_results, failed_path) {
                        eprintln!("Error writing failed parquet: {}", e);
                    } else {
                        eprintln!("Wrote failed records: {}", failed_path.display());
                    }
                }
            }

            // Write individual markdown files if requested
            if output_format == "markdown" || output_format == "both" {
                if let Some(ref out_dir) = cli.out_dir {
                    for r in results.iter().filter(|r| r.success) {
                        if let Some(ref content) = r.cleaned_content {
                            let md_path = out_dir.join(&r.source);
                            if let Some(parent) = md_path.parent() {
                                std::fs::create_dir_all(parent).ok();
                            }
                            std::fs::write(&md_path, content).ok();
                        }
                    }
                }
            }

            // Print per-file results
            for r in &results {
                let status = if r.success { "OK" } else { "FAIL" };
                let warn_count = r.warnings.len();
                if warn_count > 0 {
                    println!("{}: {} ({} warnings)", r.source, status, warn_count);
                } else {
                    println!("{}: {}", r.source, status);
                }
            }

            collected_results.extend(results);
        }

        all_results = collected_results;
    } else {
        // === Markdown branch ===
        let md_paths = gather_markdown_paths(&cli.path);
        if md_paths.is_empty() {
            eprintln!("No markdown files found at {:?}", cli.path);
            std::process::exit(1);
        }

        let base_path = if cli.path.is_dir() {
            cli.path.clone()
        } else {
            cli.path
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .to_path_buf()
        };

        let progress = AtomicUsize::new(0);

        all_results = md_paths
            .par_iter()
            .map(|path| {
                let content = read_file_lossy(path);
                let filename = path.to_string_lossy().to_string();
                let mut result = process_document(&filename, &content, &options, cli.exclude_failed);

                // Determine failure category string for write_output
                let failure_cat = if !result.success {
                    result
                        .failure_analysis
                        .as_ref()
                        .map(|fa| fa.category.to_string())
                } else {
                    None
                };

                // Write output
                let destination = write_output(
                    path,
                    &base_path,
                    result.cleaned_content.as_deref().unwrap_or(""),
                    &content,
                    cli.in_place,
                    cli.out_dir.as_deref(),
                    cli.exclude_failed,
                    failure_cat.as_deref(),
                );
                result.destination = destination;

                let count = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if count % 1000 == 0 {
                    eprintln!("Processed {} files...", count);
                }

                // Print per-file result
                let status = if result.success { "OK" } else { "FAIL" };
                let warn_count = result.warnings.len();
                if warn_count > 0 {
                    println!("{}: {} ({} warnings)", result.source, status, warn_count);
                } else {
                    println!("{}: {}", result.source, status);
                }

                result
            })
            .collect();
    }

    print_summary(&all_results);
}
