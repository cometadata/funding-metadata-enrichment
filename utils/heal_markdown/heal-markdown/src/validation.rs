use std::collections::HashSet;
use std::sync::OnceLock;

use once_cell::sync::Lazy;
use regex::Regex;

use crate::heuristics::CAPTION_PATTERN;

const MAX_REASONABLE_LINE: usize = 1000;
const LONG_LINE_FRACTION: f64 = 0.20;
const MIN_WHITESPACE_RATIO: f64 = 0.05;
const MAX_AVG_LINE_LENGTH: f64 = 1000.0;
const MAX_HEADER_LENGTH: usize = 100;
const MIN_SENTENCE_LENGTH_FOR_PERIOD: usize = 10;
const TITLE_CASE_RATIO: f64 = 0.6;
const MAX_TITLE_CASE_WORDS: usize = 6;

static NUMBERED_SECTION_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\d+\.(\d+\.?)?\s+[A-Z]").unwrap());

static COMMON_HEADERS: OnceLock<HashSet<&'static str>> = OnceLock::new();

fn common_headers() -> &'static HashSet<&'static str> {
    COMMON_HEADERS.get_or_init(|| {
        let mut set = HashSet::new();
        for h in &[
            "abstract",
            "introduction",
            "background",
            "related work",
            "methodology",
            "methods",
            "experimental setup",
            "experiment",
            "experiments",
            "results",
            "discussion",
            "conclusion",
            "conclusions",
            "acknowledgments",
            "acknowledgements",
            "references",
            "bibliography",
            "appendix",
            "supplementary",
            "future work",
            "limitations",
            "summary",
        ] {
            set.insert(*h);
        }
        set
    })
}

/// Validates input text quality, returning `(is_valid, message)`.
///
/// Checks for empty text, excessive long lines, low whitespace ratio,
/// and excessively long average line length.
pub fn validate_input_quality(text: &str) -> (bool, String) {
    if text.trim().is_empty() {
        return (false, "Empty input".to_string());
    }

    let lines: Vec<&str> = text.lines().collect();
    let total_lines = lines.len();

    if total_lines == 0 {
        return (false, "Empty input".to_string());
    }

    // Check long lines
    let long_count = lines
        .iter()
        .filter(|l| l.len() > MAX_REASONABLE_LINE)
        .count();
    let long_fraction = long_count as f64 / total_lines as f64;
    if long_fraction > LONG_LINE_FRACTION {
        return (
            false,
            format!(
                "Too many long lines: {:.0}% of lines exceed {} chars",
                long_fraction * 100.0,
                MAX_REASONABLE_LINE
            ),
        );
    }

    // Check whitespace ratio
    let total_chars = text.len();
    if total_chars > 0 {
        let whitespace_count = text.chars().filter(|c| c.is_whitespace()).count();
        let ws_ratio = whitespace_count as f64 / total_chars as f64;
        if ws_ratio < MIN_WHITESPACE_RATIO {
            return (false, format!("Low whitespace ratio: {:.3}", ws_ratio));
        }
    }

    // Check average line length
    let total_line_len: usize = lines.iter().map(|l| l.len()).sum();
    let avg_line_len = total_line_len as f64 / total_lines as f64;
    if avg_line_len > MAX_AVG_LINE_LENGTH {
        return (
            false,
            format!("Average line length too high: {:.0}", avg_line_len),
        );
    }

    (true, "OK".to_string())
}

/// Checks whether a line looks like an academic section header.
///
/// Matches against common academic headers, numbered section patterns
/// (e.g. "1. Introduction", "2.1 Background"), all-uppercase short lines,
/// and title-case short lines.
pub fn is_section_header(line: &str) -> bool {
    let trimmed = line.trim();

    if trimmed.is_empty() || trimmed.len() > MAX_HEADER_LENGTH {
        return false;
    }

    // Check against common headers (case-insensitive)
    let lower = trimmed.to_lowercase();
    if common_headers().contains(lower.as_str()) {
        return true;
    }

    // Check numbered section pattern: "1. Introduction", "2.1 Background"
    if NUMBERED_SECTION_RE.is_match(trimmed) {
        return true;
    }

    // All-uppercase and under 50 chars
    if trimmed.len() < 50
        && trimmed.chars().any(|c| c.is_alphabetic())
        && trimmed
            .chars()
            .filter(|c| c.is_alphabetic())
            .all(|c| c.is_uppercase())
    {
        return true;
    }

    // Title case heuristic: >60% of words are title case, in lines <= 6 words
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    if !words.is_empty() && words.len() <= MAX_TITLE_CASE_WORDS {
        let title_case_count = words
            .iter()
            .filter(|w| {
                let mut chars = w.chars();
                if let Some(first) = chars.next() {
                    first.is_uppercase()
                } else {
                    false
                }
            })
            .count();
        let ratio = title_case_count as f64 / words.len() as f64;
        if ratio > TITLE_CASE_RATIO {
            return true;
        }
    }

    false
}

/// Checks whether a line looks like a complete sentence.
///
/// A line is considered complete if it ends with `!` or `?`, or ends with `.`
/// and is at least 10 characters long (to avoid matching abbreviations like "Fig.").
pub fn is_complete_sentence(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }

    if trimmed.ends_with('!') || trimmed.ends_with('?') {
        return true;
    }

    if trimmed.ends_with('.') && trimmed.len() >= MIN_SENTENCE_LENGTH_FOR_PERIOD {
        return true;
    }

    false
}

/// Determines whether a blank line between `prev` and `next` should be preserved.
///
/// Preserves blank lines:
/// - Between section headers
/// - After short lines (<50 chars)
/// - After complete sentences >40 chars
/// - Between a complete sentence and a line starting with an uppercase letter
pub fn should_preserve_blank_line(prev: &str, next: &str) -> bool {
    let prev_trimmed = prev.trim();
    let next_trimmed = next.trim();

    // Between section headers
    if is_section_header(prev_trimmed) || is_section_header(next_trimmed) {
        return true;
    }

    // Around figure/table captions
    if CAPTION_PATTERN.is_match(prev_trimmed) || CAPTION_PATTERN.is_match(next_trimmed) {
        return true;
    }

    // After short lines (<50 chars)
    if !prev_trimmed.is_empty() && prev_trimmed.len() < 50 {
        return true;
    }

    // After complete sentences >40 chars
    if prev_trimmed.len() > 40 && is_complete_sentence(prev_trimmed) {
        return true;
    }

    // Between complete sentence + uppercase start
    if is_complete_sentence(prev_trimmed) {
        if let Some(first_char) = next_trimmed.chars().next() {
            if first_char.is_uppercase() {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_empty() {
        let (valid, msg) = validate_input_quality("");
        assert!(!valid);
        assert!(msg.contains("Empty"));
    }

    #[test]
    fn test_validate_normal_text() {
        let text = "This is a normal paragraph.\nIt has multiple lines.\nAnd good spacing.\n";
        let (valid, _) = validate_input_quality(text);
        assert!(valid);
    }

    #[test]
    fn test_validate_low_whitespace() {
        let text = "a".repeat(100);
        let (valid, msg) = validate_input_quality(&text);
        assert!(!valid);
        assert!(msg.to_lowercase().contains("whitespace ratio"));
    }

    #[test]
    fn test_validate_long_lines() {
        let long_line = "x".repeat(1500);
        let text = format!("{}\n{}\nshort\nshort\n{}", long_line, long_line, long_line);
        let (valid, msg) = validate_input_quality(&text);
        assert!(!valid);
        assert!(msg.to_lowercase().contains("long lines"));
    }

    #[test]
    fn test_validate_whitespace_only() {
        let (valid, msg) = validate_input_quality("   \n  \n   ");
        assert!(!valid);
        assert!(msg.contains("Empty"));
    }

    #[test]
    fn test_validate_ok_message() {
        let text = "This is a normal paragraph.\nWith multiple lines.\n";
        let (valid, msg) = validate_input_quality(text);
        assert!(valid);
        assert_eq!(msg, "OK");
    }

    #[test]
    fn test_is_section_header_common() {
        assert!(is_section_header("Abstract"));
        assert!(is_section_header("INTRODUCTION"));
        assert!(is_section_header("Related Work"));
        assert!(is_section_header("1. Introduction"));
        assert!(is_section_header("2.1 Background"));
    }

    #[test]
    fn test_is_section_header_all_common_headers() {
        let headers = [
            "abstract",
            "introduction",
            "background",
            "related work",
            "methodology",
            "methods",
            "experimental setup",
            "experiment",
            "experiments",
            "results",
            "discussion",
            "conclusion",
            "conclusions",
            "acknowledgments",
            "acknowledgements",
            "references",
            "bibliography",
            "appendix",
            "supplementary",
            "future work",
            "limitations",
            "summary",
        ];
        for h in &headers {
            assert!(
                is_section_header(h),
                "Expected '{}' to be a section header",
                h
            );
        }
    }

    #[test]
    fn test_is_section_header_case_insensitive() {
        assert!(is_section_header("ABSTRACT"));
        assert!(is_section_header("abstract"));
        assert!(is_section_header("Abstract"));
        assert!(is_section_header("REFERENCES"));
    }

    #[test]
    fn test_is_section_header_numbered() {
        assert!(is_section_header("1. Introduction"));
        assert!(is_section_header("2.1 Background"));
        assert!(is_section_header("3.2. Methods"));
    }

    #[test]
    fn test_is_section_header_all_uppercase() {
        assert!(is_section_header("SOME NEW HEADING"));
        assert!(is_section_header("DATA ANALYSIS"));
    }

    #[test]
    fn test_is_section_header_title_case() {
        assert!(is_section_header("Data Analysis"));
        assert!(is_section_header("Key Findings"));
    }

    #[test]
    fn test_is_section_header_negative() {
        assert!(!is_section_header(""));
        assert!(!is_section_header(
            "this is a regular sentence that is quite long and not a header at all"
        ));
        assert!(!is_section_header(&"x".repeat(101)));
    }

    #[test]
    fn test_is_section_header_long_line_rejected() {
        let long = "A".repeat(101);
        assert!(!is_section_header(&long));
    }

    #[test]
    fn test_is_complete_sentence() {
        assert!(is_complete_sentence("This is a complete sentence."));
        assert!(is_complete_sentence("Is this a question?"));
        assert!(is_complete_sentence("What an exclamation!"));
        assert!(!is_complete_sentence("Not complete"));
        assert!(!is_complete_sentence("Fig."));
        assert!(!is_complete_sentence(""));
    }

    #[test]
    fn test_is_complete_sentence_short_period() {
        // Lines ending with '.' but shorter than 10 chars should not match
        assert!(!is_complete_sentence("et al."));
        assert!(!is_complete_sentence("Dr."));
        assert!(!is_complete_sentence("e.g."));
    }

    #[test]
    fn test_is_complete_sentence_exactly_ten() {
        // Exactly 10 chars ending with period should match
        assert!(is_complete_sentence("123456789."));
    }

    #[test]
    fn test_should_preserve_blank_line() {
        assert!(should_preserve_blank_line("Abstract", "Some text"));
        assert!(should_preserve_blank_line("Short line", "Next"));
        assert!(should_preserve_blank_line(
            "This is a complete sentence that ends properly.",
            "Next paragraph starts here."
        ));
    }

    #[test]
    fn test_should_preserve_blank_line_section_headers() {
        // Before or after a section header
        assert!(should_preserve_blank_line("Some text.", "Introduction"));
        assert!(should_preserve_blank_line("Abstract", "First paragraph."));
    }

    #[test]
    fn test_should_preserve_blank_line_short_prev() {
        // After short lines (< 50 chars)
        assert!(should_preserve_blank_line("Title", "Some content follows."));
    }

    #[test]
    fn test_should_preserve_blank_line_complete_sentence_uppercase() {
        // Between a complete sentence and an uppercase-starting line
        assert!(should_preserve_blank_line(
            "End of a paragraph.",
            "Another paragraph begins."
        ));
    }

    #[test]
    fn test_should_preserve_blank_line_negative() {
        // A long incomplete line followed by a lowercase continuation
        // should not preserve the blank line
        let long_incomplete =
            "this is a long line that does not end with punctuation and keeps going on and on";
        assert!(!should_preserve_blank_line(
            long_incomplete,
            "continuation of the text in lowercase"
        ));
    }

    #[test]
    fn test_should_preserve_blank_line_before_figure_caption() {
        assert!(should_preserve_blank_line(
            "this is a long line that does not end with punctuation and keeps going on and on",
            "Figure 1. Caption text here."
        ));
    }

    #[test]
    fn test_should_preserve_blank_line_after_figure_caption() {
        assert!(should_preserve_blank_line(
            "Figure 2. The experimental setup is shown.",
            "continuation of text here"
        ));
    }

    #[test]
    fn test_should_preserve_blank_line_table_caption() {
        assert!(should_preserve_blank_line(
            "this is a long line that does not end with punctuation and keeps going on and on",
            "Table 3: Summary of results."
        ));
    }
}
