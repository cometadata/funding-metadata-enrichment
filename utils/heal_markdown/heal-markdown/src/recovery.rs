use crate::dictionary::word_dict;
use crate::types::{RecoveryConfidence, WordRecoveryResult};

/// Split a single concatenated line into words using dynamic programming.
///
/// Returns a tuple of (resulting_line, number_of_splits_made).
/// If fewer than 5 words are found or the confidence ratio is below 0.4,
/// the original line is returned unchanged with 0 splits.
pub fn split_line_into_words(line: &str) -> (String, usize) {
    let line_lower = line.to_lowercase();
    let n = line_lower.len();

    if n == 0 {
        return (line.to_string(), 0);
    }

    let dict = word_dict();

    // DP arrays
    let mut dp: Vec<u32> = vec![u32::MAX; n + 1];
    let mut parent: Vec<i32> = vec![-1; n + 1];
    dp[0] = 0;

    for i in 0..n {
        if dp[i] == u32::MAX {
            continue;
        }
        let upper = std::cmp::min(i + 16, n + 1);
        for j in (i + 1)..upper {
            let word = &line_lower[i..j];
            let len = j - i;

            let cost: u32 = if dict.contains(word) {
                1
            } else if len <= 2 {
                100
            } else if len == 3 {
                10
            } else {
                5
            };

            if dp[i] + cost < dp[j] {
                dp[j] = dp[i] + cost;
                parent[j] = i as i32;
            }
        }
    }

    // Backtrack to reconstruct words
    let mut words: Vec<&str> = Vec::new();
    let mut pos = n as i32;
    while pos > 0 {
        let prev = parent[pos as usize];
        if prev < 0 {
            // Could not reach the beginning; return original
            return (line.to_string(), 0);
        }
        words.push(&line_lower[prev as usize..pos as usize]);
        pos = prev;
    }
    words.reverse();

    let total_words = words.len();

    // If fewer than 3 words found, return original (no meaningful split)
    if total_words < 3 {
        return (line.to_string(), 0);
    }

    // Calculate confidence ratio (fraction of words found in dictionary)
    let dict_words = words.iter().filter(|w| dict.contains(*w)).count();
    let confidence_ratio = dict_words as f64 / total_words as f64;

    if confidence_ratio < 0.4 {
        return (line.to_string(), 0);
    }

    let joined = words.join(" ");
    let splits_count = total_words - 1;
    (joined, splits_count)
}

/// Attempt to split concatenated text into words.
///
/// Processes each line individually, skipping lines that are empty,
/// shorter than 20 characters, or already contain spaces.
pub fn split_concatenated_text(
    text: &str,
    _min_confidence: RecoveryConfidence,
) -> WordRecoveryResult {
    // Early exit if text already has enough whitespace-separated words
    if text.split_whitespace().count() > 10 {
        return WordRecoveryResult {
            text: text.to_string(),
            confidence: RecoveryConfidence::None_,
            changes_made: 0,
            issues: Vec::new(),
        };
    }

    let lines: Vec<&str> = text.lines().collect();
    let mut result_lines: Vec<String> = Vec::with_capacity(lines.len());
    let mut changes_made: usize = 0;
    let mut processable_lines: usize = 0;
    let mut issues: Vec<String> = Vec::new();

    for line in &lines {
        // Skip empty lines
        if line.trim().is_empty() {
            result_lines.push(line.to_string());
            continue;
        }

        // Skip lines shorter than 20 chars
        if line.len() < 20 {
            result_lines.push(line.to_string());
            continue;
        }

        // Skip lines that already contain spaces
        if line.contains(' ') {
            result_lines.push(line.to_string());
            continue;
        }

        processable_lines += 1;

        let (split_line, splits) = split_line_into_words(line);
        if splits > 0 {
            changes_made += 1;
            issues.push(format!("Split concatenated line: {} splits", splits));
            result_lines.push(split_line);
        } else {
            result_lines.push(line.to_string());
        }
    }

    // Determine confidence level
    let confidence = if changes_made == 0 || processable_lines == 0 {
        RecoveryConfidence::None_
    } else {
        let ratio = changes_made as f64 / processable_lines as f64;
        if ratio < 0.3 {
            RecoveryConfidence::Low
        } else if ratio < 0.7 {
            RecoveryConfidence::Medium
        } else {
            RecoveryConfidence::High
        }
    };

    WordRecoveryResult {
        text: result_lines.join("\n"),
        confidence,
        changes_made,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RecoveryConfidence;

    #[test]
    fn test_split_line_known_words() {
        let (result, splits) = split_line_into_words("theresearchanalysis");
        assert!(splits > 0);
        assert!(result.contains(" "));
    }

    #[test]
    fn test_split_line_too_short() {
        let (result, splits) = split_line_into_words("abc");
        assert_eq!(splits, 0);
        assert_eq!(result, "abc");
    }

    #[test]
    fn test_split_concatenated_already_has_spaces() {
        let result = split_concatenated_text(
            "This text already has enough word boundaries in it to pass the check",
            RecoveryConfidence::Medium,
        );
        assert_eq!(result.confidence, RecoveryConfidence::None_);
    }

    #[test]
    fn test_split_concatenated_short_line_skipped() {
        let result = split_concatenated_text("short", RecoveryConfidence::Medium);
        assert_eq!(result.changes_made, 0);
    }

    #[test]
    fn test_split_concatenated_line_with_spaces_skipped() {
        let result = split_concatenated_text("has spaces already", RecoveryConfidence::Medium);
        assert_eq!(result.changes_made, 0);
    }
}
