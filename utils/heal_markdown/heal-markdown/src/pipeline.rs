use plsfix::fix_text;

use crate::heuristics::{
    collapse_blank_lines, collapse_multiple_spaces, deduplicate_unicode_math,
    drop_short_line_runs, fix_hyphenation, merge_broken_lines, merge_broken_references,
    merge_short_fragments, normalize_bullets, remove_arxiv_watermarks,
    remove_blank_between_short_lines, remove_citations_and_noise, remove_footer_patterns,
    renumber_references, strip_repeated_lines, strip_spurious_tables, trim_leading_noise,
};
use crate::markdown::{format_markdown, unescape_brackets, validate_markdown};
use crate::types::RestorationOptions;

pub fn restore_markdown(text: &str, options: &RestorationOptions) -> (String, Vec<String>) {
    let mut warnings = Vec::new();

    // Step 1: Fix encoding issues
    let mut text = fix_text(text, None);

    // Step 2: Trim leading noise
    let (t, trimmed) = trim_leading_noise(&text);
    text = t;
    if trimmed {
        warnings.push("Trimmed leading noise before first paragraph".to_string());
    }

    // Step 3: Strip repeated lines (headers/footers)
    let (t, removed_lines) = strip_repeated_lines(
        &text,
        options.page_length_estimate,
        options.repeat_threshold,
        options.max_header_words,
    );
    text = t;
    if !removed_lines.is_empty() {
        let preview: Vec<_> = removed_lines.iter().take(5).cloned().collect();
        warnings.push(format!("Removed repeated lines: {}", preview.join(", ")));
    }

    // Step 4: Remove footer patterns
    text = remove_footer_patterns(&text);

    // Step 5: Remove arXiv/viXra watermark lines
    text = remove_arxiv_watermarks(&text);

    // Step 6: Strip spurious table markup
    let (t, table_warnings) = strip_spurious_tables(&text);
    text = t;
    if !table_warnings.is_empty() {
        let preview: Vec<_> = table_warnings.iter().take(3).cloned().collect();
        warnings.push(format!("Stripped spurious tables: {}", preview.join("; ")));
    }

    // Step 7: Drop short line runs
    let (t, short_runs) = drop_short_line_runs(&text, 3, 5);
    text = t;
    if !short_runs.is_empty() {
        let preview: Vec<_> = short_runs.iter().take(5).cloned().collect();
        warnings.push(format!("Dropped short-line runs: {}", preview.join(", ")));
    }

    // Step 8: Remove citations and noise
    text = remove_citations_and_noise(&text, options.strip_citations);

    // Step 9: Remove blank lines between short lines
    text = remove_blank_between_short_lines(&text, 24);

    // Step 10: Merge short fragments
    let (t, merged_fragments) = merge_short_fragments(&text, 12, 6);
    text = t;
    if merged_fragments {
        warnings.push("Merged clusters of short fragments".to_string());
    }

    // Step 11: Normalize bullets
    text = normalize_bullets(&text);

    // Step 12: Fix hyphenation
    text = fix_hyphenation(&text);

    // Step 13: Merge broken lines (with figure/table caption awareness)
    text = merge_broken_lines(&text);

    // Step 14: Merge broken reference entries
    text = merge_broken_references(&text);

    // Step 15: Collapse multiple consecutive spaces
    text = collapse_multiple_spaces(&text);

    // Step 16: Collapse blank lines
    text = collapse_blank_lines(&text);

    // Step 17: Deduplicate Unicode math characters
    text = deduplicate_unicode_math(&text);

    // Step 18: Format markdown
    text = format_markdown(&text);

    // Step 19: Unescape brackets from pulldown-cmark
    text = unescape_brackets(&text);

    // Step 20: Renumber duplicate references
    text = renumber_references(&text);

    // Step 21: Validate markdown
    warnings.extend(validate_markdown(&text));

    (text, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RestorationOptions;

    #[test]
    fn test_restore_markdown_basic() {
        let text = "# Title\n\nSome paragraph text.\n\nAnother paragraph.\n";
        let options = RestorationOptions::default();
        let (result, _warnings) = restore_markdown(text, &options);
        assert!(result.contains("Title"));
        assert!(result.contains("paragraph"));
    }

    #[test]
    fn test_restore_markdown_fixes_hyphenation() {
        // Use a longer first line so trim_leading_noise doesn't strip it
        // (requires 4+ consecutive alpha chars on the line)
        let text = "This important re-\nsearch was very good.\n";
        let options = RestorationOptions::default();
        let (result, _) = restore_markdown(text, &options);
        assert!(result.contains("research"));
    }

    #[test]
    fn test_restore_markdown_removes_footers() {
        let text = "Content here\n  42  \nMore content\nPage 3 of 10\nEnd";
        let options = RestorationOptions::default();
        let (result, _) = restore_markdown(text, &options);
        assert!(!result.contains("Page 3 of 10"));
    }

    #[test]
    fn test_restore_markdown_normalizes_bullets() {
        // Use longer bullet items so they aren't merged as short fragments,
        // and check for either "- " or "* " since format_markdown may convert dashes
        let text = "\u{2022} Item one with some detail\n\u{25cf} Item two with some detail\n";
        let options = RestorationOptions::default();
        let (result, _) = restore_markdown(text, &options);
        // normalize_bullets converts unicode bullets to "- ", then
        // format_markdown (pulldown-cmark) may convert to "* "
        assert!(
            result.contains("- Item") || result.contains("* Item"),
            "Expected normalized bullet markers, got: {}",
            result
        );
    }
}
