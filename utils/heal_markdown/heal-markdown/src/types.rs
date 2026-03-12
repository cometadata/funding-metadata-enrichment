use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryConfidence {
    None_,
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WordRecoveryResult {
    pub text: String,
    pub confidence: RecoveryConfidence,
    pub changes_made: usize,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FailureCategory {
    Success,
    PartialSuccess,
    NoWordBoundaries,
    MalformedStructure,
    EncodingIssues,
    NeedsReview,
    Unrecoverable,
}

impl fmt::Display for FailureCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FailureCategory::Success => "success",
            FailureCategory::PartialSuccess => "partial",
            FailureCategory::NoWordBoundaries => "no_spaces",
            FailureCategory::MalformedStructure => "malformed",
            FailureCategory::EncodingIssues => "encoding",
            FailureCategory::NeedsReview => "review",
            FailureCategory::Unrecoverable => "unrecoverable",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FailureAnalysis {
    pub category: FailureCategory,
    pub issues: Vec<String>,
    pub recoverable: bool,
    pub recovery_attempted: bool,
    pub recovery_confidence: Option<RecoveryConfidence>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RestorationOptions {
    pub page_length_estimate: usize,
    pub repeat_threshold: f64,
    pub max_header_words: usize,
    pub strip_citations: bool,
}

impl Default for RestorationOptions {
    fn default() -> Self {
        Self {
            page_length_estimate: 50,
            repeat_threshold: 0.8,
            max_header_words: 12,
            strip_citations: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RestorationResult {
    pub source: String,
    pub destination: Option<PathBuf>,
    pub warnings: Vec<String>,
    pub success: bool,
    pub original_size: usize,
    pub output_size: usize,
    pub validation_error: Option<String>,
    pub failure_analysis: Option<FailureAnalysis>,
    pub recovery_attempted: bool,
    pub cleaned_content: Option<String>,
    pub original_content: Option<String>,
}
