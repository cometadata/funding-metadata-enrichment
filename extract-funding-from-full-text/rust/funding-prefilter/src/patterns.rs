//! Pattern loading and compilation for the two-stage regex gate.
//!
//! This module owns:
//! * [`PREFILTER_ARMS`] — the six top-level alternation arms of the Python
//!   `_FUNDING_PREFILTER_REGEX` (extraction.py lines 21-26), copied verbatim.
//! * [`Patterns`] — a compiled bundle containing the prefilter [`regex::RegexSet`]
//!   and the fine-grained positive / negative [`fancy_regex::Regex`] lists
//!   loaded from the YAML config.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// The six top-level alternation arms of the Python `_FUNDING_PREFILTER_REGEX`.
///
/// Copied verbatim from `funding_statement_extractor/statements/extraction.py`
/// lines 21-26. Each arm begins with a `\b` word boundary at the start of the
/// corresponding Python string literal. When joined with `|` these reproduce
/// the original Python compiled regex byte-for-byte (content-identical).
pub const PREFILTER_ARMS: &[&str] = &[
    r"\b(fund|grant|support|acknowledg|award|sponsor|thank|scholarship|fellowship|financ|grate|gratitude|foundation)\w*\b",
    r"\b(?:NSF|NSFC|NIH|NASA|ESA|CNES|DOE|ERC|EPSRC|DFG|JSPS|MCIN|AEI|FAPESP|CNPq|JPL|CSIC|CONICET|CONACYT|RFBR|HFSP|JST|MEXT|KAKENHI)\b",
    r"\bin\s+(?:the\s+)?(?:framework|scope|context)\s+of\b",
    r"\bis\s+part\s+of\s+the\s+(?:project|research|R\+D\+i)\b",
    r"\bcarried\s+out\s+(?:within|as|in|during)\b",
    r"\b(?:state\s+assignment|госзадания)\b",
];

/// YAML file shape:
/// ```yaml
/// patterns:
///   - '...'
/// negative_patterns:
///   - '...'
/// ```
#[derive(Debug, Deserialize)]
struct PatternsYaml {
    #[serde(default)]
    patterns: Vec<String>,
    #[serde(default)]
    negative_patterns: Vec<String>,
}

/// Compiled pattern bundle used by the gate.
///
/// * `prefilter` — linear-time multi-alternation first pass over paragraphs.
/// * `positives` / `negatives` — the fine-grained YAML layer applied only on
///   prefilter survivors. Both lists use `fancy_regex` to preserve lookaround
///   semantics from the original Python patterns.
pub struct Patterns {
    pub prefilter: regex::RegexSet,
    pub positives: Vec<fancy_regex::Regex>,
    pub negatives: Vec<fancy_regex::Regex>,
}

impl Patterns {
    /// Load the YAML positive / negative lists from `yaml_path` and compile
    /// the hardcoded [`PREFILTER_ARMS`] into a [`regex::RegexSet`].
    ///
    /// All patterns are compiled case-insensitively to match the Python
    /// `re.IGNORECASE` behavior.
    pub fn load(yaml_path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(yaml_path)
            .with_context(|| format!("reading patterns YAML from {}", yaml_path.display()))?;
        let parsed: PatternsYaml = serde_yaml::from_str(&raw)
            .with_context(|| format!("parsing patterns YAML at {}", yaml_path.display()))?;
        Self::from_lists(parsed.patterns, parsed.negative_patterns)
    }

    /// Construct from in-memory lists (used by tests and callers that already
    /// hold the pattern strings).
    pub fn from_lists(positives: Vec<String>, negatives: Vec<String>) -> Result<Self> {
        let prefilter = regex::RegexSetBuilder::new(PREFILTER_ARMS)
            .case_insensitive(true)
            .build()
            .context("compiling prefilter RegexSet from PREFILTER_ARMS")?;

        let positives = compile_list(&positives, "positive")?;
        let negatives = compile_list(&negatives, "negative")?;

        Ok(Self {
            prefilter,
            positives,
            negatives,
        })
    }
}

/// Compile a list of pattern sources with `(?i)` case-insensitive prefix,
/// annotating failures with the list label, pattern index, and source text.
fn compile_list(sources: &[String], label: &str) -> Result<Vec<fancy_regex::Regex>> {
    sources
        .iter()
        .enumerate()
        .map(|(idx, src)| {
            let wrapped = format!("(?i){src}");
            fancy_regex::Regex::new(&wrapped).with_context(|| {
                format!(
                    "compiling {label} pattern #{idx}: {src:?}",
                    label = label,
                    idx = idx,
                    src = src
                )
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn from_lists_compiles_simple_patterns() {
        let positives = vec![r"\bfunded\s+by\b".to_string()];
        let negatives = vec![r"\bsupported\s+by\s+figure\b".to_string()];
        let patterns = Patterns::from_lists(positives, negatives).unwrap();
        assert_eq!(patterns.positives.len(), 1);
        assert_eq!(patterns.negatives.len(), 1);
    }

    #[test]
    fn prefilter_arms_compile() {
        let patterns = Patterns::from_lists(vec![], vec![]).unwrap();
        assert_eq!(patterns.prefilter.len(), 6);

        // "supported by the NSF" must match arms 0 (support\w*) and 1 (NSF).
        let nsf_matches: Vec<usize> = patterns
            .prefilter
            .matches("supported by the NSF")
            .into_iter()
            .collect();
        assert!(nsf_matches.contains(&0), "expected arm 0 to match: {nsf_matches:?}");
        assert!(nsf_matches.contains(&1), "expected arm 1 to match: {nsf_matches:?}");

        // "in the framework of" must match arm 2.
        let framework_matches: Vec<usize> = patterns
            .prefilter
            .matches("in the framework of")
            .into_iter()
            .collect();
        assert!(
            framework_matches.contains(&2),
            "expected arm 2 to match: {framework_matches:?}"
        );

        // Cyrillic "госзадания" must match arm 5.
        let cyrillic_matches: Vec<usize> = patterns
            .prefilter
            .matches("госзадания")
            .into_iter()
            .collect();
        assert!(
            cyrillic_matches.contains(&5),
            "expected arm 5 to match: {cyrillic_matches:?}"
        );
    }

    #[test]
    fn load_real_yaml() {
        let mut yaml_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        yaml_path.push("../../funding_statement_extractor/configs/patterns/funding_patterns.yaml");
        let patterns = Patterns::load(&yaml_path)
            .unwrap_or_else(|e| panic!("failed to load {}: {e:?}", yaml_path.display()));
        assert!(!patterns.positives.is_empty(), "positives should be non-empty");
        assert!(!patterns.negatives.is_empty(), "negatives should be non-empty");
    }
}
