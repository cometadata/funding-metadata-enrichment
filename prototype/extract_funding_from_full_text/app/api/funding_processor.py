import re
import os
import time
import asyncio
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime
import torch
from pylate import models, rank
from concurrent.futures import ThreadPoolExecutor

from config import settings, FUNDING_PATTERNS


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_num_threads(1)


class FundingExtractor:
    """Core class for extracting funding statements from markdown files."""
    
    def __init__(self, model_name: str = None, load_async: bool = True):
        self.model_name = model_name or settings.model_name
        self.model = None
        self.model_loading = False
        self.model_loaded = False
        self.load_async = load_async
        self._executor = ThreadPoolExecutor(max_workers=1)
        
        if not load_async:
            self._initialize_model()
    
    def _initialize_model(self):
        try:
            self.model_loading = True
            self.model = models.ColBERT(model_name_or_path=self.model_name)
            self.model_loaded = True
            self.model_loading = False
        except Exception as e:
            self.model_loading = False
            raise RuntimeError(f"Failed to initialize model {self.model_name}: {e}")
    
    async def initialize_model_async(self):
        if self.model_loaded or self.model_loading:
            return
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._initialize_model)
    
    async def ensure_model_loaded(self):
        if not self.model_loaded:
            if self.load_async:
                await self.initialize_model_async()
            else:
                self._initialize_model()
                
        while self.model_loading:
            await asyncio.sleep(0.1)
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def is_likely_funding_statement(self, paragraph: str, score: float, threshold: float = None) -> bool:
        if threshold is None:
            threshold = settings.threshold
            
        if score < threshold:
            return False
        
        paragraph_lower = paragraph.lower()
        return any(re.search(pattern, paragraph_lower) for pattern in FUNDING_PATTERNS)
    
    def extract_funding_sentences(self, paragraph: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
        funding_sentences = []
        
        for i, sentence in enumerate(sentences):
            if re.search(r'\b(?:acknowledg|fund|support|grant|award|project)\w*\b', sentence, re.IGNORECASE):
                sentence = sentence.strip()
                
                if i + 1 < len(sentences) and sentence.endswith('No'):
                    sentence = sentence + ' ' + sentences[i + 1].strip()
                
                funding_sentences.append(sentence)
        
        return funding_sentences
    
    async def process_document_async(
        self, 
        content: str, 
        queries: Dict[str, str],
        top_k: int = None,
        threshold: float = None,
        batch_size: int = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict:
        """
        Process a single document asynchronously with progress tracking.
        
        Args:
            content: The markdown content to process
            queries: Dictionary of query names to query text
            top_k: Number of top paragraphs to analyze per query
            threshold: Minimum score threshold for relevance
            batch_size: Batch size for encoding
            progress_callback: Optional callback for progress updates (status, percentage)
            
        Returns:
            Dictionary containing extracted funding statements and metadata
        """
        await self.ensure_model_loaded()
        
        import queue
        progress_queue = queue.Queue()
        
        def sync_callback(message: str, percentage: float):
            progress_queue.put((message, percentage))
        
        async def process_queue():
            processing_done = False
            while not processing_done:
                try:
                    message, percentage = progress_queue.get_nowait()
                    if message == "DONE":
                        processing_done = True
                    elif progress_callback:
                        await progress_callback(message, percentage)
                except queue.Empty:
                    await asyncio.sleep(0.05)
                except Exception as e:
                    print(f"Queue processing error: {e}")
                    break
        
        queue_task = asyncio.create_task(process_queue())
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._process_document_sync,
                content,
                queries,
                top_k,
                threshold,
                batch_size,
                sync_callback
            )
            
            progress_queue.put(("DONE", 1.0))
            
            await queue_task
            
            if progress_callback:
                await progress_callback("Processing complete", 1.0)
                
            return result
        except Exception as e:
            queue_task.cancel()
            try:
                await queue_task
            except asyncio.CancelledError:
                pass
            raise
    
    def process_document(
        self, 
        content: str, 
        queries: Dict[str, str],
        top_k: int = None,
        threshold: float = None,
        batch_size: int = None
    ) -> Dict:
        return self._process_document_sync(
            content, queries, top_k, threshold, batch_size, None
        )
    
    def _process_document_sync(
        self, 
        content: str, 
        queries: Dict[str, str],
        top_k: int = None,
        threshold: float = None,
        batch_size: int = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict:
        """
        Process a single document to extract funding statements.
        
        Args:
            content: The markdown content to process
            queries: Dictionary of query names to query text
            top_k: Number of top paragraphs to analyze per query
            threshold: Minimum score threshold for relevance
            batch_size: Batch size for encoding
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing extracted funding statements and metadata
        """
        start_time = time.time()
        
        top_k = top_k or settings.top_k
        threshold = threshold or settings.threshold
        batch_size = batch_size or settings.batch_size
        
        if progress_callback:
            progress_callback("Splitting content into paragraphs", 0.1)
        
        paragraphs = self.split_into_paragraphs(content)
        if not paragraphs:
            return self._empty_result()
        
        if progress_callback:
            progress_callback(f"Encoding {len(paragraphs)} paragraphs", 0.2)
        
        documents_embeddings = self.model.encode(
            paragraphs,
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=False
        )
        
        doc_results = {
            'num_paragraphs': len(paragraphs),
            'funding_statements': [],
            'by_query': {},
            'processed_at': datetime.now().isoformat(),
            'processing_time': 0
        }
        
        seen_statements = set()
        total_queries = len(queries)
        
        for idx, (query_name, query_text) in enumerate(queries.items()):
            progress_percent = 0.3 + (0.6 * ((idx + 1) / total_queries))
            if progress_callback:
                progress_callback(f"Processing query: {query_name}", progress_percent)
            query_results = []
            
            query_embeddings = self.model.encode(
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
                    
                    if self.is_likely_funding_statement(paragraph, score, threshold):
                        funding_sentences = self.extract_funding_sentences(paragraph)
                        
                        for sentence in funding_sentences:
                            if sentence not in seen_statements and len(sentence) > 20:
                                seen_statements.add(sentence)
                                
                                funding_record = {
                                    'statement': sentence,
                                    'score': score,
                                    'paragraph_idx': para_id,
                                    'query': query_name,
                                    'full_paragraph': paragraph
                                }
                                
                                doc_results['funding_statements'].append(funding_record)
                                query_results.append(funding_record)
            
            if query_results:
                doc_results['by_query'][query_name] = query_results
        
        doc_results['processing_time'] = time.time() - start_time
        
        return doc_results
    
    def _empty_result(self) -> Dict:
        return {
            'num_paragraphs': 0,
            'funding_statements': [],
            'by_query': {},
            'processed_at': datetime.now().isoformat(),
            'processing_time': 0
        }