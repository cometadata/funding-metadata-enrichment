import os
import re
import sys
import json
import time
import psutil
import hashlib
import warnings
import argparse
from pathlib import Path
from datetime import datetime
import multiprocessing

import yaml
import torch
from pylate import models, rank

# For suppressing warnings in the child processes
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract funding acknowledgements iteratively with checkpoint support'
    )
    parser.add_argument(
        '-d', '--directory',
        required=True,
        help='Directory containing markdown files to search'
    )
    parser.add_argument(
        '-q', '--query-file',
        required=True,
        help='YAML file containing search queries'
    )
    parser.add_argument(
        '-o', '--output',
        default='funding_extractions.json',
        help='Output JSON file for results (default: funding_extractions.json)'
    )
    parser.add_argument(
        '--checkpoint-file',
        default=None,
        help='Checkpoint file for saving progress (default: <output>.checkpoint)'
    )
    parser.add_argument(
        '--batch-size-docs',
        type=int,
        default=100,
        help='Number of documents to process per batch (default: 100)'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=50,
        help='Save checkpoint every N documents (default: 50)'
    )
    parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=5,
        help='Number of top paragraphs to analyze per query per document (default: 5)'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=28.0,
        help='Minimum score threshold for relevance (default: 28.0)'
    )
    parser.add_argument(
        '-m', '--model',
        default='lightonai/GTE-ModernColBERT-v1',
        help='Pre-trained ColBERT model to use (default: lightonai/GTE-ModernColBERT-v1)'
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding (default: 32)'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto-detect based on system)'
    )
    parser.add_argument(
        '--maxtasksperchild',
        type=int,
        default=5,
        help='Tasks per worker before replacement; lower for more stability (default: 5)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous checkpoint'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing of all files (ignore checkpoint)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def get_file_hash(file_path):
    return hashlib.md5(file_path.encode()).hexdigest()


def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    return {
        'processed_files': {},
        'last_update': None,
        'total_processed': 0
    }


def save_checkpoint(checkpoint_file, checkpoint_data):
    checkpoint_data['last_update'] = datetime.now().isoformat()
    
    temp_file = checkpoint_file + '.tmp'
    with open(temp_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    os.replace(temp_file, checkpoint_file)


def load_results(output_file):
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    
    return None


def save_results_incremental(output_file, new_results, 
                           search_params, checkpoint_data):
    existing_data = load_results(output_file)
    
    if existing_data:
        existing_results = {doc['file_path']: doc 
                          for doc in existing_data.get('results_by_document', [])}
        
        for doc in new_results:
            if doc:
                existing_results[doc['file_path']] = doc
        
        all_results = list(existing_results.values())
    else:
        all_results = [doc for doc in new_results if doc]
    
    total_statements = sum(len(doc['funding_statements']) for doc in all_results)
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'search_parameters': search_params,
        'processing_info': {
            'total_files_seen': checkpoint_data['total_processed'],
            'files_with_results': len(all_results),
            'last_checkpoint': checkpoint_data['last_update']
        },
        'summary': {
            'total_documents_processed': len(all_results),
            'documents_with_funding': len([d for d in all_results if d['funding_statements']]),
            'total_statements': total_statements
        },
        'results_by_document': all_results
    }
    
    temp_file = output_file + '.tmp'
    with open(temp_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    os.replace(temp_file, output_file)
    
    if len(new_results) > 0:
        print(f"\nUpdated results saved to: {output_file}")


def get_system_limits():
    cpu_count = multiprocessing.cpu_count()
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    max_memory_gb = available_gb * 0.7
    memory_per_worker = 2.0
    max_workers_by_memory = int(max_memory_gb / memory_per_worker)
    
    optimal_workers = min(cpu_count - 1, max_workers_by_memory, 6)
    
    return {
        'cpu_count': cpu_count,
        'available_memory_gb': available_gb,
        'optimal_workers': max(1, optimal_workers)
    }


def load_queries(query_file):
    with open(query_file, 'r') as f:
        queries_data = yaml.safe_load(f)
    
    if not queries_data or 'queries' not in queries_data:
        raise ValueError("Query file must contain a 'queries' key with list of queries")
    
    return queries_data


def find_markdown_files(directory):
    md_files = []
    path = Path(directory)
    
    if not path.exists():
        raise ValueError(f"Directory {directory} does not exist")
    
    for file_path in path.rglob('*.md'):
        md_files.append(str(file_path))
    
    return sorted(md_files)


def split_into_paragraphs(text):
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def is_likely_funding_statement(paragraph, score, threshold=28.0):
    if score < threshold:
        return False
    
    funding_patterns = [
        r'\backnowledg\w*\s+(?:funding|financial|support)',
        r'\bfund\w*\s+(?:by|from|through)',
        r'\bsupport\w*\s+(?:by|from|through)',
        r'\bgrant\w*\s+(?:from|by|number|no\.?|#)',
        r'\baward\w*\s+(?:from|by|number|no\.?|#)',
        r'\bproject\s+(?:number|no\.?|#)',
        r'\bcontract\s+(?:number|no\.?|#)',
        r'\bfinancial\w*\s+support',
        r'\bresearch\w*\s+(?:fund|support)',
        r'\bthis\s+(?:work|research|study)\s+(?:was|is)\s+(?:supported|funded)',
        r'\bgrateful\w*\s+(?:for|to).*(?:fund|support)',
        r'\bthank\w*\s+(?:for|to).*(?:fund|support)',
    ]
    
    paragraph_lower = paragraph.lower()
    return any(re.search(pattern, paragraph_lower) for pattern in funding_patterns)


def extract_funding_sentences(paragraph):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
    funding_sentences = []
    
    for i, sentence in enumerate(sentences):
        if re.search(r'\b(?:acknowledg|fund|support|grant|award|project)\w*\b', sentence, re.IGNORECASE):
            sentence = sentence.strip()
            
            if i + 1 < len(sentences) and sentence.endswith('No'):
                sentence = sentence + ' ' + sentences[i + 1].strip()
            
            funding_sentences.append(sentence)
    
    return funding_sentences


def initialize_worker():
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def process_single_document_worker(args):
    file_path, model_name, queries, top_k, threshold, batch_size = args
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    try:
        model = models.ColBERT(model_name_or_path=model_name)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        paragraphs = split_into_paragraphs(content)
        if not paragraphs:
            return None
        
        documents_embeddings = model.encode(
            paragraphs,
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=False
        )
        
        doc_results = {
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'num_paragraphs': len(paragraphs),
            'funding_statements': [],
            'by_query': {},
            'processed_at': datetime.now().isoformat()
        }
        
        seen_statements = set()
        
        for query_name, query_text in queries.items():
            query_results = []
            
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
                    
                    if is_likely_funding_statement(paragraph, score, threshold):
                        funding_sentences = extract_funding_sentences(paragraph)
                        
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
        
        return doc_results if doc_results['funding_statements'] else None
        
    except Exception as e:
        print(f"\nError processing {file_path}: {e}", file=sys.stderr)
        return None


def main():
    global args
    args = parse_args()
    
    try:
        checkpoint_file = args.checkpoint_file or f"{args.output}.checkpoint"
        
        system_limits = get_system_limits()
        num_workers = args.workers if args.workers else system_limits['optimal_workers']
        
        print(f"System resources:")
        print(f"  CPU cores: {system_limits['cpu_count']}")
        print(f"  Available memory: {system_limits['available_memory_gb']:.1f} GB")
        print(f"  Using {num_workers} parallel workers")
        print(f"  Batch size: {args.batch_size_docs} documents")
        print(f"  Save interval: every {args.save_interval} documents")
        print(f"  Worker Recycling (maxtasksperchild): {args.maxtasksperchild}")

        if args.verbose:
            print(f"\nLoading queries from: {args.query_file}")
        queries_data = load_queries(args.query_file)
        queries = queries_data['queries']
        
        if args.verbose:
            print(f"\nSearching for markdown files in: {args.directory}")
        md_files = find_markdown_files(args.directory)
        print(f"\nFound {len(md_files)} markdown files")
        
        if not md_files:
            print("No markdown files found!")
            return
        
        checkpoint_data = load_checkpoint(checkpoint_file) if args.resume and not args.force else {
            'processed_files': {},
            'last_update': None,
            'total_processed': 0
        }
        
        if args.resume and not args.force:
            files_to_process = []
            for file_path in md_files:
                file_hash = get_file_hash(file_path)
                if file_hash not in checkpoint_data['processed_files']:
                    files_to_process.append(file_path)
            
            print(f"Resuming: {len(checkpoint_data['processed_files'])} already processed")
            print(f"Remaining: {len(files_to_process)} files to process")
        else:
            files_to_process = md_files
            if args.force:
                print("Force mode: reprocessing all files")
        
        if not files_to_process:
            print("All files already processed!")
            return
        
        search_params = {
            'directory': args.directory,
            'query_file': args.query_file,
            'top_k': args.top_k,
            'threshold': args.threshold,
            'model': args.model,
            'batch_size': args.batch_size,
            'batch_size_docs': args.batch_size_docs,
            'num_workers': num_workers,
            'total_files': len(md_files)
        }
        
        total_processed = len(checkpoint_data['processed_files'])
        batch_results = []
        
        for batch_start in range(0, len(files_to_process), args.batch_size_docs):
            batch_end = min(batch_start + args.batch_size_docs, len(files_to_process))
            batch_files = files_to_process[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//args.batch_size_docs + 1}: "
                  f"files {batch_start + 1}-{batch_end} of {len(files_to_process)}")
            
            worker_args = [
                (file_path, args.model, queries, args.top_k, args.threshold, args.batch_size)
                for file_path in batch_files
            ]
            
            with multiprocessing.Pool(
                processes=num_workers,
                initializer=initialize_worker,
                maxtasksperchild=args.maxtasksperchild
            ) as pool:
                async_results = []
                for worker_arg in worker_args:
                    async_result = pool.apply_async(process_single_document_worker, (worker_arg,))
                    async_results.append((async_result, worker_arg[0]))  # Store result and file_path
                
                for async_result, file_path in async_results:
                    try:
                        result = async_result.get()
                        if result:
                            batch_results.append(result)
                        
                        file_hash = get_file_hash(file_path)
                        checkpoint_data['processed_files'][file_hash] = {
                            'path': file_path,
                            'processed_at': datetime.now().isoformat(),
                            'found_funding': result is not None
                        }
                        total_processed += 1
                        checkpoint_data['total_processed'] = total_processed
                        
                        if total_processed % args.save_interval == 0:
                            save_checkpoint(checkpoint_file, checkpoint_data)
                            save_results_incremental(args.output, batch_results, 
                                                   search_params, checkpoint_data)
                            batch_results = []
                            print(f"  Checkpoint saved at {total_processed} documents")
                            
                    except Exception as e:
                        print(f"\nError processing {file_path}: {e}", file=sys.stderr)
            
            if batch_results:
                save_checkpoint(checkpoint_file, checkpoint_data)
                save_results_incremental(args.output, batch_results, 
                                       search_params, checkpoint_data)
                batch_results = []

            print("  Batch complete. Pausing for 2 seconds...")
            time.sleep(2)
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total documents processed: {total_processed}")
        print(f"Results saved to: {args.output}")
        print(f"Checkpoint saved to: {checkpoint_file}")
        
        final_results = load_results(args.output)
        if final_results:
            summary = final_results['summary']
            print(f"\nFinal Summary:")
            print(f"  Documents with funding: {summary['documents_with_funding']}")
            print(f"  Total funding statements: {summary['total_statements']}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")
        if 'checkpoint_data' in locals():
            save_checkpoint(checkpoint_file, checkpoint_data)
            if 'batch_results' in locals() and batch_results:
                save_results_incremental(args.output, batch_results, 
                                       search_params, checkpoint_data)
        print("Checkpoint saved. Use --resume to continue.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()