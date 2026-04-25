pub mod patterns;
pub mod parquet_io;
pub mod pipeline;
pub mod materialize;

use once_cell::sync::Lazy;

pub use patterns::Patterns;

static PARAGRAPH_SPLIT_RE: Lazy<regex::Regex> = Lazy::new(|| {
    regex::Regex::new(r"\n\s*\n").expect("paragraph split regex compiles")
});

/// Split on `\n\s*\n`, trim, drop empties. Matches Python `_split_into_paragraphs`.
pub fn split_paragraphs(text: &str) -> Vec<&str> {
    PARAGRAPH_SPLIT_RE
        .split(text)
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect()
}

/// Apply the two-stage gate to a single paragraph.
/// 1. Prefilter RegexSet must have >=1 hit.
/// 2. If survives: >=1 positive match AND 0 negative matches.
pub fn gate_paragraph(paragraph: &str, patterns: &Patterns) -> bool {
    if !patterns.prefilter.is_match(paragraph) {
        return false;
    }

    let mut has_positive = false;
    for pat in &patterns.positives {
        match pat.find(paragraph) {
            Ok(Some(_)) => {
                has_positive = true;
                break;
            }
            _ => continue,
        }
    }

    if !has_positive {
        return false;
    }

    for pat in &patterns.negatives {
        if let Ok(Some(_)) = pat.find(paragraph) {
            return false;
        }
    }

    true
}

/// Apply gate to all paragraphs in a document.
/// Returns (is_candidate, num_matched_paragraphs).
/// `is_candidate` is true iff `num_matched_paragraphs > 0`.
pub fn gate_document(text: &str, patterns: &Patterns) -> (bool, u32) {
    let mut count: u32 = 0;
    for paragraph in split_paragraphs(text) {
        if gate_paragraph(paragraph, patterns) {
            count += 1;
        }
    }
    (count > 0, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_patterns() -> &'static Patterns {
        static CELL: once_cell::sync::OnceCell<Patterns> = once_cell::sync::OnceCell::new();
        CELL.get_or_init(|| {
            let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            path.push("../../funding_statement_extractor/configs/patterns/funding_patterns.yaml");
            Patterns::load(&path).expect("load real YAML")
        })
    }

    #[test]
    fn split_paragraphs_matches_python_semantics() {
        let input = "a\n\nb\n\n\n c  \n\nd";
        let paragraphs = split_paragraphs(input);
        assert_eq!(paragraphs, vec!["a", "b", "c", "d"]);

        let empty: Vec<&str> = split_paragraphs("");
        assert!(empty.is_empty());

        let single = split_paragraphs("hello");
        assert_eq!(single, vec!["hello"]);
    }

    #[test]
    fn gate_paragraph_clear_positive() {
        let patterns = real_patterns();
        let text = "This work was supported by the NSF under grant number AST-0000001.";
        assert!(gate_paragraph(text, patterns), "expected clear funding statement to pass gate");
    }

    #[test]
    fn gate_paragraph_negative_rescue() {
        let patterns = real_patterns();
        let text = "This claim is supported by figure 3.";
        assert!(
            !gate_paragraph(text, patterns),
            "expected negative pattern to block 'supported by figure'"
        );
    }

    #[test]
    fn gate_paragraph_no_prefilter() {
        let patterns = real_patterns();
        let text = "The quick brown fox jumps over the lazy dog.";
        assert!(
            !gate_paragraph(text, patterns),
            "expected paragraph without prefilter hit to fail gate"
        );
    }

    #[test]
    fn gate_document_counts_matches() {
        let patterns = real_patterns();
        let text = "This work was supported by the NSF under grant number AST-0000001.\n\n\
                    The quick brown fox jumps over the lazy dog.\n\n\
                    This claim is supported by figure 3.";
        let (is_candidate, n) = gate_document(text, patterns);
        assert!(is_candidate);
        assert_eq!(n, 1);
    }

    #[test]
    fn gate_document_no_matches() {
        let patterns = real_patterns();
        let text = "The quick brown fox jumps over the lazy dog.\n\n\
                    Nothing funding-related here at all.\n\n\
                    Just some placeholder text without any sponsor mentions.";
        let (is_candidate, n) = gate_document(text, patterns);
        assert!(!is_candidate);
        assert_eq!(n, 0);
    }
}
