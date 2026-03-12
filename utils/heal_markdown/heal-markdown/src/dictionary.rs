use std::collections::HashSet;
use std::sync::OnceLock;

include!(concat!(env!("OUT_DIR"), "/words_generated.rs"));

const COMMON_WORDS: &[&str] = &[
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "i",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "out",
    "if",
    "about",
    "who",
    "get",
    "which",
    "go",
    "me",
    "when",
    "make",
    "can",
    "like",
    "time",
    "no",
    "just",
    "him",
    "know",
    "take",
    "people",
    "into",
    "year",
    "your",
    "good",
    "some",
    "could",
    "them",
    "see",
    "other",
    "than",
    "then",
    "now",
    "look",
    "only",
    "come",
    "its",
    "over",
    "think",
    "also",
    "back",
    "after",
    "use",
    "two",
    "how",
    "our",
    "work",
    "first",
    "well",
    "way",
    "even",
    "new",
    "want",
    "because",
    "any",
    "these",
    "give",
    "day",
    "most",
    "us",
    "is",
    "was",
    "are",
    "been",
    "has",
    "had",
    "were",
    "said",
    "did",
    "having",
    "may",
    "should",
    "could",
    "would",
    "paper",
    "using",
    "based",
    "show",
    "method",
    "system",
    "data",
    "model",
    "study",
    "results",
    "used",
    "such",
    "which",
    "between",
    "each",
    "where",
    "both",
    "those",
    "through",
    "during",
    "these",
    "research",
    "analysis",
    "approach",
    "problem",
    "network",
    "algorithm",
    "process",
    "performance",
    "learning",
    "training",
];

static WORD_DICT: OnceLock<HashSet<&'static str>> = OnceLock::new();

pub fn word_dict() -> &'static HashSet<&'static str> {
    WORD_DICT.get_or_init(|| {
        let mut set = HashSet::with_capacity(DICTIONARY_WORDS.len() + COMMON_WORDS.len());
        for w in DICTIONARY_WORDS {
            set.insert(*w);
        }
        for w in COMMON_WORDS {
            set.insert(*w);
        }
        set
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_dict_contains_common_words() {
        let dict = word_dict();
        assert!(dict.contains("the"));
        assert!(dict.contains("research"));
        assert!(dict.contains("algorithm"));
    }

    #[test]
    fn test_word_dict_contains_dictionary_words() {
        let dict = word_dict();
        assert!(dict.contains("hello"));
        assert!(dict.contains("world"));
    }

    #[test]
    fn test_word_dict_reasonable_size() {
        let dict = word_dict();
        assert!(dict.len() > 100);
    }
}
