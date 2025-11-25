import re
from typing import List, Tuple

from funding_extractor.core.models import FundingStatement


GRANT_NUMBER_PATTERNS = [
    r'\bgrant\s+(?:agreement\s+)?(?:no\.?|number|#)\s*[A-Z0-9\-]+',
    r'\baward\s+(?:no\.?|number|#)\s*[A-Z0-9\-]+',
    r'\bcontract\s+(?:no\.?|number|#)\s*[A-Z0-9\-]+',
    r'\bproject\s+(?:no\.?|number|#)\s*[A-Z0-9\-]+',
    r'\b[A-Z]{1,4}\d{2}[- ]?[A-Z]{2}\d{5,7}\b',
    r'\b[A-Z]{2,4}[- ]?\d{6,8}\b',
]

THIS_WORK_PATTERNS = [
    r'\bthis\s+(?:work|research|study|project|paper)\s+(?:was|is|has\s+been)\s+(?:supported|funded|financed)',
    r'\bthis\s+material\s+is\s+based\s+(?:upon|on)\s+work\s+supported',
    r'\bresearch\s+reported\s+(?:in\s+this\s+(?:publication|paper)\s+)?was\s+supported',
]

FINANCIAL_VERB_PATTERNS = [
    r'\bfunded\s+by\b',
    r'\bfinanced\s+by\b',
    r'\breceived\s+(?:financial\s+)?(?:funding|support)\s+from',
    r'\bsupported\s+(?:in\s+part\s+)?by\b',
    r'\backnowledg\w*\s+(?:the\s+)?(?:financial\s+)?support',
]

ORGANIZATION_STRUCTURE_PATTERNS = [
    r'\bfoundation\s+(?:for|of)\s+[A-Z]',
    r'\bministry\s+of\s+[A-Z]',
    r'\b(?:national|research)\s+council\b',
    r'\b(?:national|research)\s+(?:science\s+)?foundation\b',
]

UNDER_GRANT_PATTERNS = [
    r'\bunder\s+(?:grant|contract|award|project)\b',
    r'\bunder\s+the\s+(?:auspices|aegis)\s+of\b',
]

PARTIAL_FUNDING_PATTERNS = [
    r'\b(?:partially|partly)\s+(?:supported|funded)\s+by',
    r'\bsupported\s+in\s+part\s+by',
    r'\bin\s+part\s+by\s+(?:a\s+)?grant',
]

AUTHOR_SPECIFIC_PATTERNS = [
    r'\b[A-Z]\.\s*[A-Z][a-z]+\s+(?:is|was|acknowledges)\s+(?:supported|funded)',
    r'\b[A-Z][a-z]+\s+gratefully\s+acknowledges',
    r'\b[A-Z][a-z]+\s+(?:is|was)\s+(?:grateful|thankful)\s+(?:for|to)',
    r'\bthe\s+(?:first|second|third)\s+author\s+(?:is|was)\s+supported',
]

# Negative patterns
CITATION_PATTERNS = [
    r'^\s*\[\d+\]',
    r'^\s*\d+\.\s+[A-Z][a-z]+',
    r'(?:[A-Z]\.\s*)+[A-Za-z]+,\s+[A-Z]\.\s*[A-Za-z]+',
]

TECHNICAL_CONTEXT_PATTERNS = [
    r'supported\s+by\s+(?:evidence|data|results?|findings?|observations?|analysis)',
    r'supported\s+by\s+(?:figure|fig\.|table|section|equation|eq\.|theorem|lemma)',
    r'supported\s+by\s+(?:previous|prior|earlier|related|recent)\s+(?:work|research|study)',
    r'supported\s+by\s+(?:our|these|the|this)\s+(?:results?|findings?|observations?)',
    r'\b(?:decision|technical|emotional)\s+support\b',
    r'\bsupport\s+(?:vector|inference|system)\b',
    r'\bargument\s+is\s+supported\s+by',
    r'\bhypothesis\s+is\s+supported\s+by',
    r'\bclaim\s+is\s+supported\s+by',
]

AUTHOR_BIO_PATTERNS = [
    r'\breceived\s+(?:the|his|her)\s+(?:b\.?s\.?|m\.?s\.?|ph\.?d)',
    r'\bis\s+(?:a|an)\s+(?:professor|researcher|scientist|engineer|student)',
    r'\bjoined\s+(?:the|a)\s+(?:department|faculty|team|group)',
    r'\bwas\s+(?:a\s+)?recipient\s+of\s+(?:the\s+)?(?:best|distinguished)',
    r'\bcurrently\s+(?:works|working)\s+(?:at|for|on)',
]


def _check_patterns(text: str, patterns: List[str]) -> bool:
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def compute_confidence_score(text: str, semantic_score: float = 0.0) -> Tuple[float, List[str]]:
    score = 0.0
    markers = []

    if _check_patterns(text, GRANT_NUMBER_PATTERNS):
        score += 30
        markers.append("grant_number")

    if _check_patterns(text, THIS_WORK_PATTERNS):
        score += 30
        markers.append("this_work_structure")

    if _check_patterns(text, FINANCIAL_VERB_PATTERNS):
        score += 20
        markers.append("financial_verbs")

    if _check_patterns(text, ORGANIZATION_STRUCTURE_PATTERNS):
        score += 20
        markers.append("organization_structure")

    if _check_patterns(text, UNDER_GRANT_PATTERNS):
        score += 15
        markers.append("under_grant")

    if _check_patterns(text, PARTIAL_FUNDING_PATTERNS):
        score += 10
        markers.append("partial_funding")

    if _check_patterns(text, AUTHOR_SPECIFIC_PATTERNS):
        score += 10
        markers.append("author_specific")

    if _check_patterns(text, CITATION_PATTERNS):
        score -= 40
        markers.append("citation_reference")

    if _check_patterns(text, TECHNICAL_CONTEXT_PATTERNS):
        score -= 35
        markers.append("technical_context")

    if _check_patterns(text, AUTHOR_BIO_PATTERNS):
        score -= 30
        markers.append("author_bio")

    return max(0, score), markers


def should_keep_statement(
    statement: FundingStatement,
    high_confidence_threshold: float = 30.0,
    low_confidence_threshold: float = 10.0,
    semantic_score_boost: float = 12.0,
) -> Tuple[bool, str, float]:

    confidence_score, markers = compute_confidence_score(
        statement.statement,
        statement.score
    )

    if confidence_score >= high_confidence_threshold:
        return True, f"high_confidence ({confidence_score:.0f})", confidence_score

    if statement.score >= semantic_score_boost and confidence_score >= 10:
        return True, f"high_semantic ({statement.score:.1f}) + markers", confidence_score

    if confidence_score < low_confidence_threshold:
        return False, f"low_confidence ({confidence_score:.0f})", confidence_score

    if confidence_score >= 20:
        return True, f"medium_confidence ({confidence_score:.0f})", confidence_score

    if statement.score >= 12.0:
        return True, f"medium_confidence + semantic ({statement.score:.1f})", confidence_score

    return False, f"insufficient_confidence ({confidence_score:.0f})", confidence_score


def apply_post_filter(
    statements: List[FundingStatement],
    high_confidence_threshold: float = 30.0,
    low_confidence_threshold: float = 10.0,
    semantic_score_boost: float = 12.0,
) -> List[FundingStatement]:

    filtered = []
    for stmt in statements:
        keep, reason, score = should_keep_statement(
            stmt,
            high_confidence_threshold=high_confidence_threshold,
            low_confidence_threshold=low_confidence_threshold,
            semantic_score_boost=semantic_score_boost,
        )
        if keep:
            filtered.append(stmt)

    return filtered
