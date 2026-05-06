import os
import re
import sys
import time
import shutil
import signal
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
import pdfplumber
import pymupdf


MARGIN_WIDTH_POINTS = 70
MIN_SEQUENTIAL_NUMBERS = 4

HEADER_HEIGHT_POINTS = 75
FOOTER_AVOIDANCE_HEIGHT_POINTS = 75
MIN_HEADER_OCCURRENCE = 2
LAYOUT_TOLERANCE = 10
OUTPUT_SUFFIX = "_preprocessed.pdf"

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Batch find and remove/cover specified elements from PDF files.')

    parser.add_argument('-i', '--input', type=str, default='pdfs',
                        help='Input directory containing PDF files (default: pdfs)')
    parser.add_argument('-o', '--output', type=str, default='output',
                        help='Output directory for processed files (default: output)')
    parser.add_argument('-p', '--pattern', type=str, default='**/*.pdf',
                        help='File pattern to match in the input directory (default: **/*.pdf for recursive search)')
    parser.add_argument('-w', '--workers', type=int, default=max(1, os.cpu_count() - 1),
                        help='Number of parallel processes to use (default: CPU cores - 1)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging for debugging purposes')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force reprocessing of all files, ignoring existing output')
    parser.add_argument('-c', '--copy', action='store_true',
                        help='Copy PDFs that do not require redaction to output directory')
    parser.add_argument('-r', '--remove', nargs='+', choices=['line-numbers', 'headers'], required=True,
                        help='Specify what to remove: line-numbers, headers, or both')
    parser.add_argument('--header-text-threshold', type=float, default=0.8,
                        help='Minimum text similarity for header detection (0.0-1.0, default: 0.8)')
    
    return parser.parse_args()


def find_header_regions_by_text(pdf_path, logger, text_threshold=0.8):
    logger.info("Analyzing to find repeating header text...")
    header_texts = {}
    page_count = 0
    redaction_bboxes = {}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            
            for page in pdf.pages:
                page_num = page.page_number - 1
                header_texts[page_num] = []
                
                lines = page.extract_text_lines(layout=True, y_tolerance=5)
                for line in lines:
                    if line['top'] <= HEADER_HEIGHT_POINTS:
                        header_texts[page_num].append({
                            'text': line['text'].strip(),
                            'bbox': pymupdf.Rect(line['x0'], line['top'], line['x1'], line['bottom'])
                        })
                        
            text_groups = defaultdict(list)
            
            for page_num, headers in header_texts.items():
                for header in headers:
                    text = header['text']
                    if not text:
                        continue
                    
                    best_match_key = None
                    best_match_score = 0
                    
                    for group_text in text_groups.keys():
                        similarity = SequenceMatcher(None, text, group_text).ratio()
                        if similarity >= text_threshold and similarity > best_match_score:
                            best_match_key = group_text
                            best_match_score = similarity
                    
                    if best_match_key:
                        text_groups[best_match_key].append((page_num, header['bbox']))
                    else:
                        text_groups[text] = [(page_num, header['bbox'])]
            
            min_occurrences = max(MIN_HEADER_OCCURRENCE, int(page_count * 0.1))
            
            for text, occurrences in text_groups.items():
                unique_pages = set(p[0] for p in occurrences)
                if len(unique_pages) >= min_occurrences:
                    logger.debug(f"Found repeating header text: '{text}' on {len(unique_pages)} pages")
                    for page_num, bbox in occurrences:
                        if page_num not in redaction_bboxes:
                            redaction_bboxes[page_num] = []
                        bbox.y0 -= 2; bbox.y1 += 2
                        redaction_bboxes[page_num].append(bbox)
            
        logger.info(f"Identified {len(redaction_bboxes)} pages with repeating header text.")
    except Exception as e:
        logger.error(f"Could not analyze header regions due to an error: {e}")
        
    return redaction_bboxes


def _is_valid_line_number_column(column, min_len=4, max_avg_step=1.5, min_density=0.7):
    if len(column) < min_len:
        return False
    
    for i in range(1, len(column)):
        if column[i]['numeric_val'] <= column[i-1]['numeric_val']:
            return False

    steps = [column[i]['numeric_val'] - column[i-1]['numeric_val'] for i in range(1, len(column))]
    if not steps:
        return len(column) >= min_len

    avg_step = sum(steps) / len(steps)
    if avg_step > max_avg_step:
        return False
        
    min_val = column[0]['numeric_val']
    max_val = column[-1]['numeric_val']
    expected_range_size = max_val - min_val + 1
    
    if expected_range_size <= 0:
        return False

    density = len(column) / expected_range_size
    return density >= min_density


def find_line_number_regions(pdf_path, logger):
    logger.info("Performing holistic column analysis for line numbers...")
    redaction_bboxes = {}
    X_TOLERANCE = 10 

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_num = page.page_number - 1
                words = page.extract_words(x_tolerance=3, y_tolerance=3)

                numeric_words = []
                for word in words:
                    if word['text'].strip().isdigit():
                        word['numeric_val'] = int(word['text'].strip())
                        numeric_words.append(word)

                if len(numeric_words) < MIN_SEQUENTIAL_NUMBERS:
                    continue

                columns = defaultdict(list)
                for word in numeric_words:
                    column_key = round(word['x0'] / X_TOLERANCE)
                    columns[column_key].append(word)

                page_redactions = []
                
                for column_key, words_in_column in columns.items():
                    words_in_column.sort(key=lambda w: w['top'])

                    if _is_valid_line_number_column(words_in_column, min_len=MIN_SEQUENTIAL_NUMBERS):
                        logger.debug(f"Page {page_num+1}: Valid line number column found at x-pos ~{column_key * X_TOLERANCE}")
                        
                        for w in words_in_column:
                            search_bbox = pymupdf.Rect(
                                w['x0'] - 5,
                                0,
                                w['x1'] + 5,
                                page.height
                            )
                            page_redactions.append({
                                'text': w['text'],
                                'search_bbox': search_bbox,
                                'pdfplumber_bbox': (w['x0'], w['top'], w['x1'], w['bottom'])
                            })
                
                if page_redactions:
                    redaction_bboxes[page_num] = page_redactions
            
        if redaction_bboxes:
            logger.info(f"Found and validated line number columns on {len(redaction_bboxes)} pages.")
        else:
            logger.info("No valid line number columns were found.")
            
    except Exception as e:
        logger.error(f"Could not analyze line number regions due to an error: {e}", exc_info=True)

    return redaction_bboxes


def redact_or_cover_regions(pdf_path, regions_to_process, output_path, logger):
    try:
        doc = pymupdf.open(pdf_path)
        total_redacted = 0
        total_covered = 0

        for page_num, items in regions_to_process.items():
            if page_num >= len(doc):
                continue
            
            page = doc[page_num]
            redactions_on_page = 0

            for item in items:
                if isinstance(item, dict) and 'text' in item:
                    text_to_find = item['text']
                    search_bbox = item['search_bbox']
                    
                    text_instances = page.search_for(text_to_find, clip=search_bbox)
                    
                    if text_instances:
                        bbox = text_instances[0]
                        bbox.x0 -= 2
                        bbox.x1 += 2
                        bbox.y0 -= 1
                        bbox.y1 += 1
                        
                        annot = page.add_redact_annot(bbox, fill=(1, 1, 1))
                        total_redacted += 1
                        redactions_on_page += 1
                        logger.debug(f"Redacting '{text_to_find}' at {bbox}")
                    else:
                        logger.debug(f"Could not find text '{text_to_find}' in search area")
                        
                elif isinstance(item, pymupdf.Rect):
                    bbox = item
                    if page.get_text("text", clip=bbox, sort=True).strip():
                        annot = page.add_redact_annot(bbox, fill=(1, 1, 1))
                        total_redacted += 1
                        redactions_on_page += 1
                    else:
                        page.draw_rect(
                            bbox,
                            color=(1, 1, 1),
                            fill=(1, 1, 1),
                            overlay=True
                        )
                        total_covered += 1
            
            if redactions_on_page > 0:
                page.apply_redactions()

        if total_redacted > 0 or total_covered > 0:
            doc.save(output_path, garbage=4, deflate=True, clean=True)
            logger.info(
                f"Successfully processed file. Redacted {total_redacted} text areas and "
                f"covered {total_covered} vector areas. Saved to '{output_path.name}'"
            )
        else:
            logger.info("No areas were found that required redaction or covering.")
            return "no_action"
            
        doc.close()
        return "success"
    except Exception as e:
        logger.error(f"Failed to redact/cover sections and save file: {e}", exc_info=True)
        return "failure"


def process_single_pdf_worker(pdf_file, args):
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    output_path = Path(args.output)
    output_file = output_path / f"{pdf_file.stem}{OUTPUT_SUFFIX}"

    logger.info(f"Processing: {pdf_file.name}")

    all_regions_to_process = defaultdict(list)
    
    if 'line-numbers' in args.remove:
        line_number_regions = find_line_number_regions(pdf_file, logger)
        for page_num, items in line_number_regions.items():
            all_regions_to_process[page_num].extend(items)
    
    if 'headers' in args.remove:
        header_regions = find_header_regions_by_text(pdf_file, logger, args.header_text_threshold)
        for page_num, bboxes in header_regions.items():
            all_regions_to_process[page_num].extend(bboxes)

    if not all_regions_to_process:
        removal_types = ' and '.join(args.remove)
        logger.info(f"No {removal_types} found to process in {pdf_file.name}.")
        if args.copy:
            copy_output_file = output_path / pdf_file.name
            try:
                shutil.copy2(pdf_file, copy_output_file)
                logger.info(f"Copied {pdf_file.name} to output directory (no action needed)")
                return "copied", pdf_file.name
            except Exception as e:
                logger.error(f"Failed to copy {pdf_file.name}: {e}")
                return "failure", pdf_file.name
        return "skipped", pdf_file.name

    status = redact_or_cover_regions(pdf_file, all_regions_to_process, output_file, logger)
    
    return status, pdf_file.name


def process_batch(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    logger = logging.getLogger(__name__)

    if not input_path.is_dir():
        logger.error(f"Input path is not a directory: {args.input}")
        return 0, 0, 0

    output_path.mkdir(parents=True, exist_ok=True)
    
    pdf_files_all = list(input_path.glob(args.pattern))
    
    if not pdf_files_all:
        logger.warning(f"No PDF files found in '{args.input}' matching pattern: '{args.pattern}'")
        return 0, 0, 0

    if args.force:
        files_to_process = pdf_files_all
    else:
        files_to_process = [
            f for f in pdf_files_all
            if not (output_path / f"{f.stem}{OUTPUT_SUFFIX}").exists()
        ]

    stats = defaultdict(int)
    stats['skipped'] = len(pdf_files_all) - len(files_to_process)
    
    logger.info(f"Found {len(pdf_files_all)} total files. Processing {len(files_to_process)} files with {args.workers} workers.")
    if stats['skipped'] > 0:
        logger.info(f"Skipping {stats['skipped']} files that already have a redacted version in the output directory.")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_pdf_worker, pdf_file, args): pdf_file for pdf_file in files_to_process}

        for i, future in enumerate(as_completed(futures), 1):
            pdf_file = futures[future]
            try:
                status, filename = future.result()
                stats[status] += 1
                logger.info(f"[{i}/{len(files_to_process)}] Finished {filename} with status: {status.upper()}")
            except Exception as e:
                stats['failure'] += 1
                logger.error(f"An unexpected error occurred while processing {pdf_file.name}: {e}", exc_info=True)

    return stats


def main():
    args = parse_arguments()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting batch processing...")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Using {args.workers} parallel workers.")

    start_time = time.time()
    
    stats = process_batch(args)
    success = stats.get('success', 0)
    failure = stats.get('failure', 0)
    skipped = stats.get('skipped', 0) + stats.get('no_action', 0)
    copied = stats.get('copied', 0)
    
    elapsed = time.time() - start_time
    total_files = success + failure + skipped + copied

    logger.info(f"\nBatch processing complete in {elapsed:.2f} seconds.")
    logger.info(f"Total files handled: {total_files}")
    logger.info(f"  - Successful: {success}")
    logger.info(f"  - Failed: {failure}")
    logger.info(f"  - Skipped (or no action needed): {skipped}")
    if copied > 0:
        logger.info(f"  - Copied (no action needed): {copied}")

    return 0 if failure == 0 else 1


if __name__ == "__main__":
    sys.exit(main())