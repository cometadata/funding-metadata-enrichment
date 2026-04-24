import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple
from typing import Pattern

import torch
from pylate import models
from pylate.scores import colbert_scores

from funding_statement_extractor.config.loader import load_funding_patterns
from funding_statement_extractor.statements.models import FundingStatement


def _split_into_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


_FUNDING_PREFILTER_REGEX = re.compile(
    r"\b(fund|grant|support|acknowledg|award|sponsor|thank|scholarship|fellowship|financial)\w*\b",
    re.IGNORECASE,
)


def _prefilter_paragraphs(paragraphs: List[str]) -> Tuple[List[str], List[int]]:
    if not paragraphs:
        return [], []
    keep = [False] * len(paragraphs)
    for i, p in enumerate(paragraphs):
        if _FUNDING_PREFILTER_REGEX.search(p):
            keep[i] = True
            if i > 0:
                keep[i - 1] = True
            if i + 1 < len(paragraphs):
                keep[i + 1] = True
    kept = [paragraphs[i] for i in range(len(paragraphs)) if keep[i]]
    indices = [i for i in range(len(paragraphs)) if keep[i]]
    return kept, indices


class SemanticExtractionService:
    def __init__(self) -> None:
        self._model_cache: Dict[str, models.ColBERT] = {}
        self._model_cache_lock = threading.Lock()

        self._pattern_cache: Dict[Tuple[Optional[str], Optional[str]], Tuple[List[Pattern], List[Pattern]]] = {}
        self._pattern_cache_lock = threading.Lock()

        self._query_embeddings_cache: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Dict[str, Any]] = {}
        self._query_tensor_cache: Dict[Tuple[str, Tuple[Tuple[str, str], ...], str], Tuple[torch.Tensor, List[str]]] = {}
        self._query_cache_lock = threading.Lock()

    def _get_model(self, model_name: str) -> models.ColBERT:
        with self._model_cache_lock:
            model = self._model_cache.get(model_name)
            if model is None:
                model = models.ColBERT(model_name_or_path=model_name)
                self._model_cache[model_name] = model
            return model

    @staticmethod
    def _pattern_cache_key(patterns_file: Optional[str], custom_config_dir: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        return (patterns_file or "__default__", custom_config_dir or "__default__")

    def _get_compiled_patterns(self, patterns_file: Optional[str], custom_config_dir: Optional[str]) -> Tuple[List[Pattern], List[Pattern]]:
        key = self._pattern_cache_key(patterns_file, custom_config_dir)
        with self._pattern_cache_lock:
            cached = self._pattern_cache.get(key)
            if cached is not None:
                return cached

        patterns, negative_patterns = load_funding_patterns(
            patterns_file, custom_config_dir)
        compiled_patterns = [re.compile(pattern, re.IGNORECASE)
                             for pattern in patterns]
        compiled_negative = [re.compile(pattern, re.IGNORECASE)
                             for pattern in negative_patterns]

        result = (compiled_patterns, compiled_negative)
        with self._pattern_cache_lock:
            self._pattern_cache[key] = result
        return result

    @staticmethod
    def _query_cache_key(model_name: str, queries: Dict[str, str]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
        return (model_name, tuple(sorted(queries.items())))

    def _get_query_embeddings(self, model_name: str, queries: Dict[str, str], model: models.ColBERT) -> Dict[str, Any]:
        if not queries:
            return {}

        key = self._query_cache_key(model_name, queries)
        with self._query_cache_lock:
            cached = self._query_embeddings_cache.get(key)
            if cached is not None:
                return cached

        embeddings: Dict[str, Any] = {}
        for query_name, query_text in queries.items():
            embeddings[query_name] = model.encode(
                [query_text], batch_size=1, is_query=True, show_progress_bar=False)

        with self._query_cache_lock:
            self._query_embeddings_cache[key] = embeddings
        return embeddings

    def _prepare_padded_documents(self, documents_embeddings: Any, device: Any) -> torch.Tensor:
        doc_tensors = [torch.as_tensor(e) for e in documents_embeddings]
        padded = torch.nn.utils.rnn.pad_sequence(
            doc_tensors, batch_first=True, padding_value=0
        )
        return padded.to(device)

    def _get_query_embeddings_tensor(
        self,
        model_name: str,
        queries: Dict[str, str],
        model: models.ColBERT,
        device: Any,
    ) -> Tuple[torch.Tensor, List[str]]:
        numpy_map = self._get_query_embeddings(model_name, queries, model)
        cache_key = (model_name, tuple(sorted(queries.items())), str(device))
        with self._query_cache_lock:
            cached = self._query_tensor_cache.get(cache_key)
            if cached is not None:
                return cached

        ordered_names = list(queries.keys())
        q_tensors: List[torch.Tensor] = []
        for name in ordered_names:
            emb = numpy_map[name]
            if isinstance(emb, list) and len(emb) == 1:
                emb = emb[0]
            t = torch.as_tensor(emb)
            if t.ndim == 3 and t.shape[0] == 1:
                t = t.squeeze(0)
            q_tensors.append(t)
        padded = torch.nn.utils.rnn.pad_sequence(
            q_tensors, batch_first=True, padding_value=0
        ).to(device)

        result = (padded, ordered_names)
        with self._query_cache_lock:
            self._query_tensor_cache[cache_key] = result
        return result

    def _score_all_queries(
        self,
        padded_queries: torch.Tensor,
        padded_docs: torch.Tensor,
    ) -> torch.Tensor:
        return colbert_scores(
            queries_embeddings=padded_queries,
            documents_embeddings=padded_docs,
        )

    @staticmethod
    def _is_likely_funding_statement(
        paragraph: str,
        score: float,
        threshold: float,
        compiled_patterns: List[Pattern],
        negative_patterns: List[Pattern],
    ) -> bool:
        if score > 14.0:
            return True

        paragraph_lower = paragraph.lower()

        for pattern in negative_patterns:
            if re.search(pattern, paragraph_lower):
                return False

        has_regex_match = any(re.search(pattern, paragraph_lower)
                              for pattern in compiled_patterns)

        if has_regex_match:
            if score > 3.0:
                return True

        if score < threshold:
            return False

        return has_regex_match

    @staticmethod
    def _extract_funding_sentences(paragraph: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph)
        funding_sentences = []

        for i, sentence in enumerate(sentences):
            if re.search(r"\b(?:acknowledg|fund|support|grant|award|project|hospitality|facilities|scholarship|fellowship|thanks|thank)\w*\b", sentence, re.IGNORECASE):
                sentence = sentence.strip()
                if i + 1 < len(sentences) and sentence.endswith("No"):
                    sentence = sentence + " " + sentences[i + 1].strip()
                funding_sentences.append(sentence)

        return funding_sentences

    @staticmethod
    def _should_extract_full_paragraph(paragraph: str, score: float) -> bool:
        if score > 14.0:
            return True

        if len(paragraph) < 500:
            return True

        funding_keywords = [
            "fund",
            "grant",
            "support",
            "acknowledg",
            "sponsor",
            "award",
            "financial",
            "foundation",
            "scholarship",
            "fellowship",
            "thank",
        ]

        words = paragraph.lower().split()
        if not words:
            return False

        keyword_count = sum(1 for word in words if any(
            kw in word for kw in funding_keywords))
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

    @staticmethod
    def _extract_funding_from_long_paragraph(paragraph: str) -> str:
        funding_keywords = [
            "fund",
            "grant",
            "support",
            "acknowledg",
            "sponsor",
            "award",
            "financial",
            "foundation",
            "scholarship",
            "fellowship",
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
        start_pos = max(0, keyword_positions[0] - 100)
        end_pos = min(len(paragraph), keyword_positions[-1] + 200)

        for i in range(start_pos, max(0, start_pos - 50), -1):
            if i == 0 or (i > 0 and paragraph[i - 1] in ".!?\n"):
                start_pos = i
                break

        for i in range(end_pos, min(len(paragraph), end_pos + 100)):
            if paragraph[i] in ".!?":
                end_pos = i + 1
                break

        return paragraph[start_pos:end_pos].strip()

    def extract_funding_statements(
        self,
        queries: Dict[str, str],
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        model_name: str = "lightonai/GTE-ModernColBERT-v1",
        top_k: int = 5,
        threshold: float = 10.0,
        batch_size: int = 32,
        patterns_file: Optional[str] = None,
        custom_config_dir: Optional[str] = None,
        enable_paragraph_prefilter: bool = False,
    ) -> List[FundingStatement]:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model = self._get_model(model_name)

        if content is None:
            if not file_path:
                raise ValueError(
                    "Either file_path or content must be provided")
            with open(file_path, "r", encoding="utf-8") as fh:
                content = fh.read()

        paragraphs = _split_into_paragraphs(content)
        if not paragraphs:
            return []

        if enable_paragraph_prefilter:
            paragraphs_to_encode, original_indices = _prefilter_paragraphs(paragraphs)
            if not paragraphs_to_encode:
                return []
        else:
            paragraphs_to_encode = paragraphs
            original_indices = list(range(len(paragraphs)))

        compiled_patterns, negative_patterns = self._get_compiled_patterns(
            patterns_file, custom_config_dir)
        documents_embeddings = model.encode(
            paragraphs_to_encode, batch_size=batch_size,
            is_query=False, show_progress_bar=False,
        )

        device = next(model.parameters()).device

        padded_docs = self._prepare_padded_documents(documents_embeddings, device)
        padded_queries, ordered_query_names = self._get_query_embeddings_tensor(
            model_name, queries, model, device,
        )
        all_scores = self._score_all_queries(padded_queries, padded_docs)

        k = min(top_k, all_scores.shape[1])
        topk_scores, topk_idx = torch.topk(all_scores, k=k, dim=1)
        topk_scores_cpu = topk_scores.cpu().tolist()
        topk_idx_cpu = topk_idx.cpu().tolist()

        seen_statements = set()
        funding_statements: List[FundingStatement] = []

        for q_pos, query_name in enumerate(ordered_query_names):
            for rank_pos in range(k):
                encoded_idx = topk_idx_cpu[q_pos][rank_pos]
                score = float(topk_scores_cpu[q_pos][rank_pos])
                para_id = original_indices[encoded_idx]
                paragraph = paragraphs[para_id]

                if self._is_likely_funding_statement(
                    paragraph, score, threshold, compiled_patterns, negative_patterns
                ):
                    if self._should_extract_full_paragraph(paragraph, score):
                        statement_text = paragraph.strip()
                    elif len(paragraph) > 1000:
                        statement_text = self._extract_funding_from_long_paragraph(
                            paragraph)
                    else:
                        funding_sentences = self._extract_funding_sentences(
                            paragraph)
                        if funding_sentences:
                            statement_text = " ".join(funding_sentences)
                        else:
                            continue

                    if statement_text not in seen_statements and len(statement_text) > 20:
                        seen_statements.add(statement_text)
                        stmt = FundingStatement(
                            statement=statement_text,
                            score=score,
                            query=query_name,
                            paragraph_idx=para_id,
                        )
                        funding_statements.append(stmt)

        return funding_statements

    def rescue_by_patterns(
        self,
        content: str,
        existing_statements: List[FundingStatement],
        patterns_file: Optional[str] = None,
        custom_config_dir: Optional[str] = None,
    ) -> List[FundingStatement]:
        compiled_patterns, negative_patterns = self._get_compiled_patterns(
            patterns_file, custom_config_dir
        )

        high_confidence_patterns = [
            re.compile(p, re.IGNORECASE) for p in [
                r'\bthis\s+(?:work|research|study|project|paper)\s+(?:was|is|has\s+been)\s+(?:supported|funded|financed)',
                r'\bgrant\s+(?:no\.?|number|#)\s*[A-Z0-9\-]+',
                r'\baward\s+(?:no\.?|number|#)\s*[A-Z0-9\-]+',
                r'\bfunded\s+by\b',
                r'\bfinanced\s+by\b',
                r'\bsupported\s+in\s+part\s+by\b',
                r'\bunder\s+(?:grant|contract|award)\s+(?:no\.?|number)?\s*[A-Z0-9]',
                r'\breceived\s+(?:financial\s+)?(?:funding|support)\s+from',
                r'\backnowledg\w*\s+(?:the\s+)?(?:financial\s+)?support\s+(?:of|from)',
                r'\bthis\s+material\s+is\s+based\s+(?:upon|on)\s+work\s+supported',
                r'This research used resources of',
            ]
        ]

        paragraphs = _split_into_paragraphs(content)
        if not paragraphs:
            return []

        existing_texts = {stmt.statement.lower().strip()
                          for stmt in existing_statements}
        rescued = []

        for para_id, paragraph in enumerate(paragraphs):
            para_lower = paragraph.lower()

            if para_lower.strip() in existing_texts:
                continue

            if any(re.search(p, para_lower) for p in negative_patterns):
                continue

            pattern_matches = sum(
                1 for p in high_confidence_patterns
                if re.search(p, para_lower)
            )

            regular_pattern_match = any(
                re.search(p, para_lower) for p in compiled_patterns
            )

            should_rescue = (
                pattern_matches >= 2 or
                (pattern_matches >= 1 and regular_pattern_match)
            )

            if should_rescue:
                if len(paragraph) < 500:
                    statement_text = paragraph.strip()
                else:
                    funding_sentences = self._extract_funding_sentences(
                        paragraph)
                    if funding_sentences:
                        statement_text = " ".join(funding_sentences)
                    elif len(paragraph) > 1000:
                        statement_text = self._extract_funding_from_long_paragraph(
                            paragraph)
                    else:
                        statement_text = paragraph.strip()

                if len(statement_text) < 20:
                    continue
                if statement_text.lower().strip() in existing_texts:
                    continue

                existing_texts.add(statement_text.lower().strip())
                rescued.append(FundingStatement(
                    statement=statement_text,
                    score=0.0,
                    query="pattern_rescue",
                    paragraph_idx=para_id,
                ))

        return rescued


_DEFAULT_SEMANTIC_EXTRACTOR = SemanticExtractionService()


def extract_funding_statements(
    queries: Dict[str, str],
    file_path: Optional[str] = None,
    content: Optional[str] = None,
    model_name: str = "lightonai/GTE-ModernColBERT-v1",
    top_k: int = 5,
    threshold: float = 10.0,
    batch_size: int = 32,
    patterns_file: Optional[str] = None,
    custom_config_dir: Optional[str] = None,
    enable_pattern_rescue: bool = False,
    enable_paragraph_prefilter: bool = False,
) -> List[FundingStatement]:
    statements = _DEFAULT_SEMANTIC_EXTRACTOR.extract_funding_statements(
        queries=queries,
        file_path=file_path,
        content=content,
        model_name=model_name,
        top_k=top_k,
        threshold=threshold,
        batch_size=batch_size,
        patterns_file=patterns_file,
        custom_config_dir=custom_config_dir,
        enable_paragraph_prefilter=enable_paragraph_prefilter,
    )

    if enable_pattern_rescue and content is not None:
        rescued = _DEFAULT_SEMANTIC_EXTRACTOR.rescue_by_patterns(
            content=content,
            existing_statements=statements,
            patterns_file=patterns_file,
            custom_config_dir=custom_config_dir,
        )
        statements.extend(rescued)
    elif enable_pattern_rescue and file_path is not None:
        with open(file_path, "r", encoding="utf-8") as fh:
            file_content = fh.read()
        rescued = _DEFAULT_SEMANTIC_EXTRACTOR.rescue_by_patterns(
            content=file_content,
            existing_statements=statements,
            patterns_file=patterns_file,
            custom_config_dir=custom_config_dir,
        )
        statements.extend(rescued)

    return statements
