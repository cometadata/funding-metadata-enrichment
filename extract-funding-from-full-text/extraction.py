import re
import os
import threading
from typing import Any, Dict, List, Optional, Tuple
from typing import Pattern

import torch
from pylate import models, rank

from models import FundingStatement
from config_loader import load_funding_patterns


def split_into_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


_MODEL_CACHE: Dict[str, models.ColBERT] = {}
_MODEL_CACHE_LOCK = threading.Lock()

_PATTERN_CACHE: Dict[Tuple[Optional[str], Optional[str]], List[Pattern]] = {}
_PATTERN_CACHE_LOCK = threading.Lock()

_QUERY_EMBEDDINGS_CACHE: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Dict[str, Any]] = {}
_QUERY_CACHE_LOCK = threading.Lock()


def _get_model(model_name: str) -> models.ColBERT:
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            model = models.ColBERT(model_name_or_path=model_name)
            _MODEL_CACHE[model_name] = model
        return model


def _get_pattern_cache_key(
    patterns_file: Optional[str],
    custom_config_dir: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    return (patterns_file or "__default__", custom_config_dir or "__default__")


def _get_compiled_patterns(
    patterns_file: Optional[str],
    custom_config_dir: Optional[str]
) -> List[Pattern]:
    key = _get_pattern_cache_key(patterns_file, custom_config_dir)
    with _PATTERN_CACHE_LOCK:
        cached = _PATTERN_CACHE.get(key)
        if cached is not None:
            return cached

    patterns = load_funding_patterns(patterns_file, custom_config_dir)
    compiled = [re.compile(pattern) for pattern in patterns]

    with _PATTERN_CACHE_LOCK:
        _PATTERN_CACHE[key] = compiled

    return compiled


def _get_query_cache_key(model_name: str, queries: Dict[str, str]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    return (model_name, tuple(sorted(queries.items())))


def _get_query_embeddings(
    model_name: str,
    queries: Dict[str, str],
    model: models.ColBERT
) -> Dict[str, Any]:
    if not queries:
        return {}

    key = _get_query_cache_key(model_name, queries)
    with _QUERY_CACHE_LOCK:
        cached = _QUERY_EMBEDDINGS_CACHE.get(key)
        if cached is not None:
            return cached

    embeddings: Dict[str, Any] = {}
    for query_name, query_text in queries.items():
        embeddings[query_name] = model.encode(
            [query_text],
            batch_size=1,
            is_query=True,
            show_progress_bar=False
        )

    with _QUERY_CACHE_LOCK:
        _QUERY_EMBEDDINGS_CACHE[key] = embeddings

    return embeddings


def is_likely_funding_statement(
    paragraph: str, 
    score: float, 
    threshold: float = 28.0,
    patterns_file: Optional[str] = None,
    custom_config_dir: Optional[str] = None,
    compiled_patterns: Optional[List[Pattern]] = None
) -> bool:
    if score < threshold:
        return False
    
    funding_patterns = compiled_patterns
    if funding_patterns is None:
        funding_patterns = _get_compiled_patterns(patterns_file, custom_config_dir)
    
    paragraph_lower = paragraph.lower()
    return any(re.search(pattern, paragraph_lower) for pattern in funding_patterns)


def extract_funding_sentences(paragraph: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
    funding_sentences = []
    
    for i, sentence in enumerate(sentences):
        if re.search(r'\b(?:acknowledg|fund|support|grant|award|project)\w*\b', sentence, re.IGNORECASE):
            sentence = sentence.strip()

            if i + 1 < len(sentences) and sentence.endswith('No'):
                sentence = sentence + ' ' + sentences[i + 1].strip()
            
            funding_sentences.append(sentence)
    
    return funding_sentences


def should_extract_full_paragraph(paragraph: str, score: float) -> bool:
    if score > 25.0:
        return True
    
    if len(paragraph) < 500:
        return True
    
    funding_keywords = [
        'fund', 'grant', 'support', 'acknowledg', 'sponsor', 
        'award', 'financial', 'foundation', 'scholarship', 'fellowship'
    ]
    
    words = paragraph.lower().split()
    if not words:
        return False
        
    keyword_count = sum(1 for word in words 
                       if any(kw in word for kw in funding_keywords))
    density = keyword_count / len(words)
    
    if density > 0.05:
        return True

    text_lower = paragraph.lower()
    keyword_positions = []
    for kw in funding_keywords:
        start = 0
        while True:
            pos = text_lower.find(kw, start)
            if pos == -1:
                break
            keyword_positions.append(pos)
            start = pos + 1
    
    if len(keyword_positions) >= 3:
        keyword_positions.sort()
        for i in range(len(keyword_positions) - 2):
            if keyword_positions[i + 2] - keyword_positions[i] < 300:
                return True
    
    return False


def extract_funding_from_long_paragraph(paragraph: str) -> str:
    funding_keywords = [
        'fund', 'grant', 'support', 'acknowledg', 'sponsor',
        'award', 'financial', 'foundation', 'scholarship', 'fellowship'
    ]
    
    text_lower = paragraph.lower()

    keyword_positions = []
    for kw in funding_keywords:
        start = 0
        while True:
            pos = text_lower.find(kw, start)
            if pos == -1:
                break
            keyword_positions.append(pos)
            start = pos + 1
    
    if not keyword_positions:
        return paragraph
    
    keyword_positions.sort()

    start_pos = max(0, keyword_positions[0] - 100)  # Add some context before
    end_pos = min(len(paragraph), keyword_positions[-1] + 200)  # Add context after
    
    for i in range(start_pos, max(0, start_pos - 50), -1):
        if i == 0 or (i > 0 and paragraph[i-1] in '.!?\n'):
            start_pos = i
            break

    for i in range(end_pos, min(len(paragraph), end_pos + 100)):
        if paragraph[i] in '.!?':
            end_pos = i + 1
            break
    
    return paragraph[start_pos:end_pos].strip()


def extract_funding_statements(
    queries: Dict[str, str],
    file_path: Optional[str] = None,
    content: Optional[str] = None,
    model_name: str = 'lightonai/GTE-ModernColBERT-v1',
    top_k: int = 5,
    threshold: float = 28.0,
    batch_size: int = 32,
    patterns_file: Optional[str] = None,
    custom_config_dir: Optional[str] = None
) -> List[FundingStatement]:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    model = _get_model(model_name)

    if content is None:
        if not file_path:
            raise ValueError("Either file_path or content must be provided")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    
    paragraphs = split_into_paragraphs(content)
    if not paragraphs:
        return []

    compiled_patterns = _get_compiled_patterns(patterns_file, custom_config_dir)

    documents_embeddings = model.encode(
        paragraphs,
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=False
    )

    query_embeddings_map = _get_query_embeddings(model_name, queries, model)

    seen_statements = set()
    funding_statements = []

    for query_name, query_text in queries.items():
        query_embeddings = query_embeddings_map.get(query_name)
        if query_embeddings is None:
            query_embeddings = model.encode(
                [query_text],
                batch_size=1,
                is_query=True,
                show_progress_bar=False
            )

        doc_ids = list(range(len(paragraphs)))
        reranked = rank.rerank(
            documents_ids=[doc_ids],
            queries_embeddings=query_embeddings,
            documents_embeddings=[documents_embeddings]
        )
        
        if reranked and len(reranked) > 0:
            top_results = reranked[0][:top_k]
            
            for result in top_results:
                para_id = result['id']
                score = float(result['score'])
                paragraph = paragraphs[para_id]

                if is_likely_funding_statement(
                    paragraph,
                    score,
                    threshold,
                    compiled_patterns=compiled_patterns
                ):
                    if should_extract_full_paragraph(paragraph, score):
                        statement_text = paragraph.strip()
                    elif len(paragraph) > 1000:
                        statement_text = extract_funding_from_long_paragraph(paragraph)
                    else:
                        funding_sentences = extract_funding_sentences(paragraph)
                        if funding_sentences:
                            statement_text = ' '.join(funding_sentences)
                        else:
                            continue

                    if statement_text not in seen_statements and len(statement_text) > 20:
                        seen_statements.add(statement_text)
                        
                        stmt = FundingStatement(
                            statement=statement_text,
                            score=score,
                            query=query_name,
                            paragraph_idx=para_id
                        )
                        funding_statements.append(stmt)
    
    return funding_statements
