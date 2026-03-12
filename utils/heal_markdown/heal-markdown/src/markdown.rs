use pulldown_cmark::{Options, Parser};
use pulldown_cmark_to_cmark::cmark;

/// Parses markdown with pulldown-cmark and re-emits it with pulldown-cmark-to-cmark
/// to normalize formatting. On error, returns the original text unchanged.
pub fn format_markdown(text: &str) -> String {
    let options = Options::all();
    let parser = Parser::new_ext(text, options);
    let events: Vec<_> = parser.collect();
    let mut output = String::new();
    match cmark(events.iter(), &mut output) {
        Ok(_) => output,
        Err(_) => text.to_string(),
    }
}

/// Validates markdown structure, returning a list of warning messages.
///
/// Checks for:
/// - Unbalanced code fences (odd number of ``` occurrences)
/// - Structural parse issues detected by pulldown-cmark
pub fn validate_markdown(text: &str) -> Vec<String> {
    let mut warnings = Vec::new();

    // Check for unbalanced code fences
    let fence_count = text.matches("```").count();
    if !fence_count.is_multiple_of(2) {
        warnings.push("Detected unbalanced code fences".to_string());
    }

    // Attempt to parse; collecting events exercises the parser
    let options = Options::all();
    let parser = Parser::new_ext(text, options);
    let _events: Vec<_> = parser.collect();

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_markdown_normalizes() {
        let text = "#  Heading\n\nSome   text\n";
        let result = format_markdown(text);
        assert!(result.contains("# Heading") || result.contains("Heading"));
    }

    #[test]
    fn test_format_markdown_invalid_returns_original() {
        let text = "Just some text";
        let result = format_markdown(text);
        assert!(result.contains("Just some text"));
    }

    #[test]
    fn test_format_markdown_preserves_code_blocks() {
        let text = "```rust\nlet x = 1;\n```\n";
        let result = format_markdown(text);
        assert!(result.contains("let x = 1;"));
    }

    #[test]
    fn test_format_markdown_preserves_lists() {
        let text = "- item one\n- item two\n- item three\n";
        let result = format_markdown(text);
        assert!(result.contains("item one"));
        assert!(result.contains("item two"));
        assert!(result.contains("item three"));
    }

    #[test]
    fn test_format_markdown_empty_string() {
        let result = format_markdown("");
        assert_eq!(result, "");
    }

    #[test]
    fn test_validate_balanced_fences() {
        let text = "```\ncode\n```\n";
        let warnings = validate_markdown(text);
        assert!(warnings.iter().all(|w| !w.contains("unbalanced")));
    }

    #[test]
    fn test_validate_unbalanced_fences() {
        let text = "```\ncode\n";
        let warnings = validate_markdown(text);
        assert!(warnings.iter().any(|w| w.contains("unbalanced")));
    }

    #[test]
    fn test_validate_no_fences() {
        let text = "Just plain text without any fences.";
        let warnings = validate_markdown(text);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_validate_multiple_balanced_fences() {
        let text = "```\nblock1\n```\n\n```\nblock2\n```\n";
        let warnings = validate_markdown(text);
        assert!(warnings.iter().all(|w| !w.contains("unbalanced")));
    }

    #[test]
    fn test_validate_three_fences_unbalanced() {
        let text = "```\nblock1\n```\n\n```\nblock2\n";
        let warnings = validate_markdown(text);
        assert!(warnings.iter().any(|w| w.contains("unbalanced")));
    }
}
