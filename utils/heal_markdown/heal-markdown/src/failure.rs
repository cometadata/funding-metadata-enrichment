use crate::types::*;

/// Categorize why a markdown validation failed and whether recovery succeeded.
///
/// Ports the Python `categorize_failure(original_text, validation_error, recovery_result)` logic.
pub fn categorize_failure(
    _original_text: &str,
    validation_error: Option<String>,
    recovery_result: Option<&WordRecoveryResult>,
) -> FailureAnalysis {
    let error_msg = match validation_error {
        None => {
            return FailureAnalysis {
                category: FailureCategory::Success,
                issues: vec![],
                recoverable: true,
                recovery_attempted: false,
                recovery_confidence: None,
            };
        }
        Some(msg) => msg,
    };

    let lower = error_msg.to_lowercase();

    if lower.contains("whitespace ratio") || lower.contains("extremely long lines") {
        // Word boundary issue -- check if recovery helped
        match recovery_result {
            Some(recovery)
                if recovery.changes_made > 0
                    && recovery.confidence != RecoveryConfidence::None_ =>
            {
                FailureAnalysis {
                    category: FailureCategory::PartialSuccess,
                    issues: vec![error_msg],
                    recoverable: true,
                    recovery_attempted: true,
                    recovery_confidence: Some(recovery.confidence),
                }
            }
            Some(_) => FailureAnalysis {
                category: FailureCategory::Unrecoverable,
                issues: vec![error_msg],
                recoverable: false,
                recovery_attempted: true,
                recovery_confidence: None,
            },
            None => FailureAnalysis {
                category: FailureCategory::Unrecoverable,
                issues: vec![error_msg],
                recoverable: false,
                recovery_attempted: false,
                recovery_confidence: None,
            },
        }
    } else if lower.contains("encoding") {
        FailureAnalysis {
            category: FailureCategory::EncodingIssues,
            issues: vec![error_msg],
            recoverable: true,
            recovery_attempted: false,
            recovery_confidence: None,
        }
    } else if lower.contains("structure") || lower.contains("malformed") {
        FailureAnalysis {
            category: FailureCategory::MalformedStructure,
            issues: vec![error_msg],
            recoverable: false,
            recovery_attempted: false,
            recovery_confidence: None,
        }
    } else {
        FailureAnalysis {
            category: FailureCategory::NeedsReview,
            issues: vec![error_msg],
            recoverable: true,
            recovery_attempted: false,
            recovery_confidence: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_error_returns_success() {
        let result = categorize_failure("text", None, None);
        assert_eq!(result.category, FailureCategory::Success);
        assert!(result.recoverable);
    }

    #[test]
    fn test_whitespace_ratio_with_successful_recovery() {
        let recovery = WordRecoveryResult {
            text: "recovered text".into(),
            confidence: RecoveryConfidence::High,
            changes_made: 5,
            issues: vec![],
        };
        let result = categorize_failure(
            "text",
            Some("Very low whitespace ratio (2.0%)".into()),
            Some(&recovery),
        );
        assert_eq!(result.category, FailureCategory::PartialSuccess);
        assert!(result.recoverable);
    }

    #[test]
    fn test_whitespace_ratio_with_failed_recovery() {
        let recovery = WordRecoveryResult {
            text: "text".into(),
            confidence: RecoveryConfidence::None_,
            changes_made: 0,
            issues: vec![],
        };
        let result = categorize_failure(
            "text",
            Some("Very low whitespace ratio (2.0%)".into()),
            Some(&recovery),
        );
        assert_eq!(result.category, FailureCategory::Unrecoverable);
        assert!(!result.recoverable);
    }

    #[test]
    fn test_encoding_error() {
        let result = categorize_failure("text", Some("encoding issues found".into()), None);
        assert_eq!(result.category, FailureCategory::EncodingIssues);
        assert!(result.recoverable);
    }

    #[test]
    fn test_structure_error() {
        let result = categorize_failure("text", Some("malformed structure".into()), None);
        assert_eq!(result.category, FailureCategory::MalformedStructure);
        assert!(!result.recoverable);
    }

    #[test]
    fn test_unknown_error() {
        let result = categorize_failure("text", Some("something weird".into()), None);
        assert_eq!(result.category, FailureCategory::NeedsReview);
        assert!(result.recoverable);
    }
}
