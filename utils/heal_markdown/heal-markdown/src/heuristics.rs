use once_cell::sync::Lazy;
use regex::Regex;

static FOOTER_PATTERN_1: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\s*\d+\s*$").unwrap());
static FOOTER_PATTERN_2: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\s*(?:Page|Pg\.?)\s*\d+(?:\s*of\s*\d+)?\s*$").unwrap());
static FOOTER_PATTERN_3: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^.*?\|\s*\d+\s*$").unwrap());
static BULLET_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^(\s*)([•●◦▪▫·]+)(\s*)").unwrap());
static HYPHENATION_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"([A-Za-z]{2,})-\s*[\r\n]+\s*([a-z][\w-]*)").unwrap());
static CID_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\(cid:\d+\)").unwrap());
static CITATION_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\[\d+\]").unwrap());
static BASE64_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^[A-Za-z0-9+/=]{40,}\s*$").unwrap());
static LEADING_ALPHA_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[A-Za-z]{4}").unwrap());

/// Scan lines, skip blanks, find first line with at least 4 consecutive alpha
/// chars or that starts with `#`. If material was trimmed, return the trimmed
/// text and `true`; otherwise return the original text and `false`.
pub fn trim_leading_noise(text: &str) -> (String, bool) {
    let lines: Vec<&str> = text.lines().collect();
    let mut cut_index: Option<usize> = None;

    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if line.trim_start().starts_with('#') || LEADING_ALPHA_PATTERN.is_match(line) {
            cut_index = Some(i);
            break;
        }
    }

    match cut_index {
        Some(idx) if idx > 0 => {
            let trimmed: String = lines[idx..].join("\n");
            (trimmed, true)
        }
        _ => (text.to_string(), false),
    }
}

/// Remove footer patterns: standalone page numbers, "Page N of M" lines,
/// and pipe-delimited page numbers.
pub fn remove_footer_patterns(text: &str) -> String {
    let result = FOOTER_PATTERN_1.replace_all(text, "");
    let result = FOOTER_PATTERN_2.replace_all(&result, "");
    let result = FOOTER_PATTERN_3.replace_all(&result, "");
    result.into_owned()
}

/// Remove `(cid:N)` patterns, optionally `[N]` citation brackets,
/// and base64-like lines (40+ chars of base64 alphabet).
pub fn remove_citations_and_noise(text: &str, strip_citations: bool) -> String {
    let result = CID_PATTERN.replace_all(text, "");
    let result = if strip_citations {
        CITATION_PATTERN.replace_all(&result, "").into_owned()
    } else {
        result.into_owned()
    };
    let result = BASE64_PATTERN.replace_all(&result, "");
    result.into_owned()
}

/// Identify runs of consecutive lines where each line is empty or <= max_len
/// chars. If a run has >= min_run lines, drop it entirely. Return cleaned text
/// and descriptions of removed spans.
pub fn drop_short_line_runs(
    text: &str,
    max_len: usize,
    min_run: usize,
) -> (String, Vec<String>) {
    let lines: Vec<&str> = text.lines().collect();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut run_start: Option<usize> = None;

    for (i, line) in lines.iter().enumerate() {
        let is_short = line.trim().is_empty() || line.len() <= max_len;
        if is_short {
            if run_start.is_none() {
                run_start = Some(i);
            }
        } else {
            if let Some(start) = run_start {
                let run_len = i - start;
                if run_len >= min_run {
                    runs.push((start, i));
                }
                run_start = None;
            }
        }
    }
    // Handle a run that extends to the end of the text
    if let Some(start) = run_start {
        let run_len = lines.len() - start;
        if run_len >= min_run {
            runs.push((start, lines.len()));
        }
    }

    let mut drop_set = vec![false; lines.len()];
    let mut descriptions = Vec::new();
    for &(start, end) in &runs {
        for i in start..end {
            drop_set[i] = true;
        }
        descriptions.push(format!(
            "Removed short-line run: lines {}-{} ({} lines)",
            start + 1,
            end,
            end - start
        ));
    }

    let kept: Vec<&str> = lines
        .iter()
        .enumerate()
        .filter(|(i, _)| !drop_set[*i])
        .map(|(_, line)| *line)
        .collect();

    (kept.join("\n"), descriptions)
}

/// If a blank line sits between two non-blank lines both <= max_len chars,
/// remove the blank line.
pub fn remove_blank_between_short_lines(text: &str, max_len: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 3 {
        return text.to_string();
    }

    let mut keep = vec![true; lines.len()];

    for i in 1..lines.len() - 1 {
        if lines[i].trim().is_empty() {
            let prev = lines[i - 1];
            let next = lines[i + 1];
            if !prev.trim().is_empty()
                && !next.trim().is_empty()
                && prev.len() <= max_len
                && next.len() <= max_len
            {
                keep[i] = false;
            }
        }
    }

    let kept: Vec<&str> = lines
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, line)| *line)
        .collect();

    kept.join("\n")
}

/// Collect consecutive non-blank lines that are <= max_len chars (skip blank
/// lines between them). If the accumulated token count >= min_tokens, join
/// them with a single space. Return the result and whether any merging occurred.
pub fn merge_short_fragments(
    text: &str,
    max_len: usize,
    min_tokens: usize,
) -> (String, bool) {
    let lines: Vec<&str> = text.lines().collect();
    let mut result_lines: Vec<String> = Vec::new();
    let mut fragment_buf: Vec<&str> = Vec::new();
    let mut changed = false;

    let flush = |buf: &mut Vec<&str>, out: &mut Vec<String>, changed: &mut bool| {
        if buf.len() > 1 {
            let token_count: usize = buf.iter().map(|l| l.split_whitespace().count()).sum();
            if token_count >= min_tokens {
                out.push(buf.join(" "));
                *changed = true;
                buf.clear();
                return;
            }
        }
        for line in buf.drain(..) {
            out.push(line.to_string());
        }
    };

    for line in &lines {
        if line.trim().is_empty() {
            flush(&mut fragment_buf, &mut result_lines, &mut changed);
            result_lines.push(line.to_string());
        } else if line.len() <= max_len {
            fragment_buf.push(line);
        } else {
            flush(&mut fragment_buf, &mut result_lines, &mut changed);
            result_lines.push(line.to_string());
        }
    }
    flush(&mut fragment_buf, &mut result_lines, &mut changed);

    (result_lines.join("\n"), changed)
}

/// Replace Unicode bullet characters at line starts with `- `.
pub fn normalize_bullets(text: &str) -> String {
    BULLET_PATTERN
        .replace_all(text, "${1}- ")
        .into_owned()
}

/// Rejoin words that were hyphenated across a line break.
pub fn fix_hyphenation(text: &str) -> String {
    HYPHENATION_PATTERN
        .replace_all(text, "${1}${2}")
        .into_owned()
}

/// Cap consecutive blank lines at 2, and right-strip each line.
pub fn collapse_blank_lines(text: &str) -> String {
    let mut result_lines: Vec<String> = Vec::new();
    let mut consecutive_blanks = 0;

    for line in text.lines() {
        let trimmed_right = line.trim_end();
        if trimmed_right.is_empty() {
            consecutive_blanks += 1;
            if consecutive_blanks <= 2 {
                result_lines.push(String::new());
            }
        } else {
            consecutive_blanks = 0;
            result_lines.push(trimmed_right.to_string());
        }
    }

    result_lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_leading_noise() {
        let text = "123\n!!!\n---\nAbstract\nSome text";
        let (result, trimmed) = trim_leading_noise(text);
        assert!(trimmed);
        assert!(result.starts_with("Abstract"));
    }

    #[test]
    fn test_trim_leading_noise_no_noise() {
        let text = "Abstract\nSome text";
        let (result, trimmed) = trim_leading_noise(text);
        assert!(!trimmed);
        assert_eq!(result, text);
    }

    #[test]
    fn test_trim_leading_noise_heading() {
        let text = "123\n# Introduction\nSome text";
        let (result, trimmed) = trim_leading_noise(text);
        assert!(trimmed);
        assert!(result.starts_with("# Introduction"));
    }

    #[test]
    fn test_trim_leading_noise_blank_lines_only() {
        let text = "\n\nAbstract\nSome text";
        let (result, trimmed) = trim_leading_noise(text);
        assert!(trimmed);
        assert!(result.starts_with("Abstract"));
    }

    #[test]
    fn test_remove_footer_patterns() {
        let text = "Some text\n  42  \nMore text\nPage 5 of 10\nEnd";
        let result = remove_footer_patterns(text);
        assert!(!result.contains("42"));
        assert!(!result.contains("Page 5"));
        assert!(result.contains("Some text"));
    }

    #[test]
    fn test_remove_footer_patterns_pipe() {
        let text = "Some text\nHeader | 3\nMore text";
        let result = remove_footer_patterns(text);
        assert!(!result.contains("Header | 3"));
        assert!(result.contains("Some text"));
    }

    #[test]
    fn test_remove_footer_patterns_pg() {
        let text = "Content here\nPg. 7\nMore content";
        let result = remove_footer_patterns(text);
        assert!(!result.contains("Pg. 7"));
    }

    #[test]
    fn test_remove_citations_and_noise_cid() {
        let text = "Some text (cid:42) more text";
        let result = remove_citations_and_noise(text, false);
        assert!(!result.contains("cid"));
        assert!(result.contains("Some text"));
    }

    #[test]
    fn test_remove_citations_strip() {
        let text = "Reference [12] found";
        let result = remove_citations_and_noise(text, true);
        assert!(!result.contains("[12]"));
    }

    #[test]
    fn test_remove_citations_keep_brackets() {
        let text = "Reference [12] found";
        let result = remove_citations_and_noise(text, false);
        assert!(result.contains("[12]"));
    }

    #[test]
    fn test_remove_base64_noise() {
        let text = "Normal text\nABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\nMore text";
        let result = remove_citations_and_noise(text, false);
        assert!(!result.contains("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));
        assert!(result.contains("Normal text"));
    }

    #[test]
    fn test_drop_short_line_runs() {
        let text = "Good line\na\nb\nc\n\na\nAnother good line";
        let (result, spans) = drop_short_line_runs(text, 3, 5);
        assert!(!spans.is_empty());
        assert!(result.contains("Good line"));
        assert!(result.contains("Another good line"));
    }

    #[test]
    fn test_drop_short_line_runs_no_drop() {
        let text = "Good line\na\nb\nAnother good line";
        let (result, spans) = drop_short_line_runs(text, 3, 5);
        assert!(spans.is_empty());
        assert_eq!(result, text);
    }

    #[test]
    fn test_merge_short_fragments() {
        let text = "He\nwent\nto\nthe\nstore\nfor\nsome\nmilk";
        let (result, changed) = merge_short_fragments(text, 12, 6);
        assert!(changed);
        assert!(result.contains(" "));
    }

    #[test]
    fn test_merge_short_fragments_no_change() {
        let text = "This is a long line that should not be merged with anything else";
        let (result, changed) = merge_short_fragments(text, 12, 6);
        assert!(!changed);
        assert_eq!(result, text);
    }

    #[test]
    fn test_merge_short_fragments_below_min_tokens() {
        let text = "Hi\nthere";
        let (result, changed) = merge_short_fragments(text, 12, 6);
        assert!(!changed);
        assert_eq!(result, text);
    }

    #[test]
    fn test_normalize_bullets() {
        let text = "\u{2022} Item one\n  \u{25cf} Sub item\n\u{25e6} Another";
        let result = normalize_bullets(text);
        assert!(result.contains("- Item one"));
        assert!(result.contains("  - Sub item"));
        assert!(result.contains("- Another"));
    }

    #[test]
    fn test_normalize_bullets_no_bullets() {
        let text = "- Already a dash\n  - Sub item";
        let result = normalize_bullets(text);
        assert_eq!(result, text);
    }

    #[test]
    fn test_fix_hyphenation() {
        let text = "The re-\nsearch was good";
        let result = fix_hyphenation(text);
        assert!(result.contains("research"));
    }

    #[test]
    fn test_fix_hyphenation_no_hyphen() {
        let text = "The research was good";
        let result = fix_hyphenation(text);
        assert_eq!(result, text);
    }

    #[test]
    fn test_fix_hyphenation_preserves_real_hyphens() {
        let text = "well-known fact\nhere";
        let result = fix_hyphenation(text);
        // "well-known" should not be merged because "known" starts at beginning
        // but the pattern requires a line break after the hyphen
        // "well-known" has no linebreak after hyphen, so it's preserved
        assert!(result.contains("well-known"));
    }

    #[test]
    fn test_collapse_blank_lines() {
        let text = "Line 1\n\n\n\n\nLine 2";
        let result = collapse_blank_lines(text);
        let blanks = result.lines().filter(|l| l.trim().is_empty()).count();
        assert!(blanks <= 2);
    }

    #[test]
    fn test_collapse_blank_lines_rstrip() {
        let text = "Line 1   \n  \nLine 2  ";
        let result = collapse_blank_lines(text);
        for line in result.lines() {
            assert_eq!(line, line.trim_end());
        }
    }

    #[test]
    fn test_remove_blank_between_short_lines() {
        let text = "Short\n\nAlso short";
        let result = remove_blank_between_short_lines(text, 24);
        assert!(!result.contains("\n\n"));
    }

    #[test]
    fn test_remove_blank_between_short_lines_long() {
        let text = "Short\n\nThis is a much longer line that exceeds the maximum length threshold for this test";
        let result = remove_blank_between_short_lines(text, 10);
        // The blank line should be kept because the next line is long
        assert!(result.contains("\n\n"));
    }

    #[test]
    fn test_remove_blank_between_short_lines_no_blank() {
        let text = "Short\nAlso short";
        let result = remove_blank_between_short_lines(text, 24);
        assert_eq!(result, text);
    }
}
