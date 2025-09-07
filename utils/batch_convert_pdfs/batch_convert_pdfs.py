import os 
import json
import time
import yaml
import shutil
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

from PIL import Image
from pdf2image import convert_from_path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Batch convert PDFs to multiple formats with automatic language detection')

    parser.add_argument('-i', '--input', type=str, default='pdfs',
                        help='Input directory containing PDF files (default: pdfs)')
    parser.add_argument('-o', '--output', type=str, default='output',
                        help='Output directory for converted files (default: output)')
    parser.add_argument('-f', '--formats', type=str, nargs='+', choices=['json', 'html', 'markdown', 'md', 'text', 'txt', 'yaml', 'doctags'], default=[
                        'json', 'html', 'markdown', 'text', 'yaml', 'doctags'], help='Output formats to generate (default: all formats)')
    parser.add_argument('-r', '--ocr', action='store_true',
                        help='Enable OCR processing with automatic language detection')
    parser.add_argument('-F', '--force-full-ocr', action='store_true',
                        help='Force full page OCR (implies --ocr)')
    parser.add_argument('-g', '--generate-images', action='store_true',
                        help='Generate page images during conversion')
    parser.add_argument('-d', '--image-dpi', type=int, default=200,
                        help='DPI for page image extraction (default: 200)')
    parser.add_argument('-w', '--image-width', type=int, default=800,
                        help='Maximum width for extracted page images (default: 800)')
    parser.add_argument('-I', '--image-format', type=str, default='PNG', choices=[
                        'PNG', 'JPEG'], help='Format for extracted page images (default: PNG)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('-p', '--pattern', type=str, default='*.pdf',
                        help='File pattern to match (default: *.pdf)')
    parser.add_argument('-c', '--copy-pdf', action='store_true',
                        help='Copy original PDF to output directory')
    parser.add_argument('-R', '--resume', action='store_true',
                        help='Resume processing by skipping already converted PDFs')
    parser.add_argument('-P', '--pages', type=str, default=None,
                        help='Page range to process (e.g., "1-5", "1,3,5", "1-5,8,10-12", "5-", "-10"). Default: all pages')
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1),
                        help='Number of parallel processes to use (default: CPU cores - 1)')

    return parser.parse_args()


def parse_page_range(page_range_str):
    if not page_range_str:
        return None

    pages = set()
    parts = page_range_str.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            if part.startswith('-'):
                end = int(part[1:])
                pages.update(range(1, end + 1))
            elif part.endswith('-'):
                start = int(part[:-1])
                return ('open_start', start)
            else:
                start, end = map(int, part.split('-'))
                pages.update(range(start, end + 1))
        else:
            pages.add(int(part))

    return sorted(list(pages)) if pages else None


def setup_converter(args, page_range=None):
    pipeline_options = PdfPipelineOptions()

    if args.ocr or args.force_full_ocr:
        ocr_options = TesseractCliOcrOptions(lang=["auto"])
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = ocr_options

        if args.force_full_ocr:
            pipeline_options.force_full_page_ocr = True

    if args.generate_images:
        pipeline_options.generate_page_images = True

    if page_range:
        if isinstance(page_range, tuple) and page_range[0] == 'open_start':
            pipeline_options.page_range = page_range
        else:
            pipeline_options.page_range = [p - 1 for p in page_range]

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    return converter


def extract_page_images(pdf_path, output_dir, dpi, max_width, image_format, logger, page_range=None):
    try:
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        if page_range:
            if isinstance(page_range, tuple) and page_range[0] == 'open_start':
                from PyPDF2 import PdfReader
                with open(pdf_path, 'rb') as f:
                    reader = PdfReader(f)
                    total_pages = len(reader.pages)
                pages_to_extract = list(range(page_range[1], total_pages + 1))
            else:
                pages_to_extract = page_range

            first_page = min(pages_to_extract)
            last_page = max(pages_to_extract)
            images = convert_from_path(
                pdf_path, dpi=dpi, first_page=first_page, last_page=last_page)

            filtered_images = []
            for page_num in pages_to_extract:
                idx = page_num - first_page
                if 0 <= idx < len(images):
                    filtered_images.append((page_num, images[idx]))
        else:
            images = convert_from_path(pdf_path, dpi=dpi)
            filtered_images = [(i + 1, img) for i, img in enumerate(images)]

        saved_count = 0
        for page_num, image in filtered_images:
            if image.size[0] > max_width:
                ratio = max_width / image.size[0]
                new_size = (max_width, int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            filename = f"{pdf_path.stem}_page_{page_num:03d}.{image_format.lower()}"
            image_path = images_dir / filename

            image.save(image_path, format=image_format)
            saved_count += 1

        logger.info(f"Extracted {saved_count} page images for {pdf_path.name}")
        return True

    except Exception as e:
        logger.error(f"Error extracting page images from {pdf_path.name}: {str(e)}")
        return False


def export_document(conv_result, base_output_dir, formats, logger):
    if conv_result.status != ConversionStatus.SUCCESS:
        return False

    doc_filename = conv_result.input.file.stem
    doc = conv_result.document
    output_dir = base_output_dir / doc_filename
    output_dir.mkdir(parents=True, exist_ok=True)

    format_map = {'markdown': 'md', 'text': 'txt'}
    try:
        for format_name in formats:
            mapped_format = format_map.get(format_name, format_name)
            if mapped_format == 'json':
                metadata_dir = output_dir / "metadata"
                metadata_dir.mkdir(exist_ok=True)
                doc.save_as_json(metadata_dir / f"{doc_filename}.json", image_mode=ImageRefMode.PLACEHOLDER)
            elif mapped_format == 'html':
                doc.save_as_html(output_dir / f"{doc_filename}.html", image_mode=ImageRefMode.EMBEDDED)
            elif mapped_format == 'md':
                doc.save_as_markdown(output_dir / f"{doc_filename}.md", image_mode=ImageRefMode.PLACEHOLDER)
            elif mapped_format == 'txt':
                doc.save_as_markdown(output_dir / f"{doc_filename}.txt", image_mode=ImageRefMode.PLACEHOLDER, strict_text=True)
            elif mapped_format == 'yaml':
                metadata_dir = output_dir / "metadata"
                metadata_dir.mkdir(exist_ok=True)
                with (metadata_dir / f"{doc_filename}.yaml").open("w") as fp:
                    fp.write(yaml.safe_dump(doc.export_to_dict()))
            elif mapped_format == 'doctags':
                metadata_dir = output_dir / "metadata"
                metadata_dir.mkdir(exist_ok=True)
                doc.save_as_document_tokens(metadata_dir / f"{doc_filename}.doctags.txt")
        return True
    except Exception as e:
        logger.error(f"Error exporting {doc_filename}: {str(e)}")
        return False


def convert_pdf(pdf_path, converter, logger):
    start_time = time.time()
    try:
        result = converter.convert(pdf_path)
        elapsed = time.time() - start_time
        if result.status == ConversionStatus.SUCCESS:
            logger.info(f"Successfully converted {pdf_path.name} in {elapsed:.2f}s")
        elif result.status == ConversionStatus.PARTIAL_SUCCESS:
            logger.warning(f"Partially converted {pdf_path.name} in {elapsed:.2f}s")
        else:
            logger.error(f"Failed to convert {pdf_path.name}")
        return result
    except Exception as e:
        logger.error(f"Exception converting {pdf_path.name}: {str(e)}")
        return None


def is_pdf_processed(pdf_path, output_dir, formats, generate_images=False, copy_pdf=False):
    pdf_stem = pdf_path.stem
    pdf_output_dir = output_dir / pdf_stem
    if not pdf_output_dir.exists():
        return False

    format_map = {'markdown': 'md', 'text': 'txt'}
    for format_name in formats:
        mapped_format = format_map.get(format_name, format_name)
        if mapped_format == 'json':
            expected_file = pdf_output_dir / "metadata" / f"{pdf_stem}.json"
        elif mapped_format == 'html':
            expected_file = pdf_output_dir / f"{pdf_stem}.html"
        elif mapped_format == 'md':
            expected_file = pdf_output_dir / f"{pdf_stem}.md"
        elif mapped_format == 'txt':
            expected_file = pdf_output_dir / f"{pdf_stem}.txt"
        elif mapped_format == 'yaml':
            expected_file = pdf_output_dir / "metadata" / f"{pdf_stem}.yaml"
        elif mapped_format == 'doctags':
            expected_file = pdf_output_dir / "metadata" / f"{pdf_stem}.doctags.txt"
        else:
            continue
        if not expected_file.exists():
            return False

    if generate_images:
        images_dir = pdf_output_dir / "images"
        if not images_dir.exists() or not any(images_dir.iterdir()):
            return False

    if copy_pdf:
        if not (pdf_output_dir / f"original_{pdf_path.name}").exists():
            return False

    return True


def process_single_pdf_worker(pdf_file, args, page_range):
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    converter = setup_converter(args, page_range)
    output_path = Path(args.output)

    logger.info(f"Processing: {pdf_file.name}")
    conv_result = convert_pdf(pdf_file, converter, logger)

    if conv_result is None:
        return "failure", pdf_file.name

    status = "failure"
    if conv_result.status in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS]:
        export_success = export_document(
            conv_result, output_path, args.formats, logger)

        if args.generate_images:
            pdf_output_dir = output_path / pdf_file.stem
            extract_page_images(pdf_file, pdf_output_dir, args.image_dpi,
                                args.image_width, args.image_format, logger, page_range)

        if args.copy_pdf:
            pdf_output_dir = output_path / pdf_file.stem
            pdf_output_dir.mkdir(parents=True, exist_ok=True)
            dest_pdf = pdf_output_dir / f"original_{pdf_file.name}"
            shutil.copy2(pdf_file, dest_pdf)

        if export_success:
            status = "success" if conv_result.status == ConversionStatus.SUCCESS else "partial"
        else:
            status = "export_failure"

    return status, pdf_file.name


def process_batch(args, page_range):
    input_path = Path(args.input)
    output_path = Path(args.output)
    logger = logging.getLogger(__name__)

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        return 0, 0, 0, 0

    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files_all = list(input_path.glob(args.pattern))
    if not pdf_files_all:
        logger.warning(f"No PDF files found matching pattern: {args.pattern}")
        return 0, 0, 0, 0

    stats = defaultdict(int)

    if args.resume:
        pdf_files_to_process = []
        for pdf_file in pdf_files_all:
            if is_pdf_processed(pdf_file, output_path, args.formats, args.generate_images, args.copy_pdf):
                stats['skipped'] += 1
                logger.info(f"Skipping (already processed): {pdf_file.name}")
            else:
                pdf_files_to_process.append(pdf_file)
    else:
        pdf_files_to_process = pdf_files_all

    total_to_process = len(pdf_files_to_process)
    logger.info(f"Found {len(pdf_files_all)} total PDF files. Processing {total_to_process} files with {args.workers} workers.")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_pdf_worker, pdf_file,
                                   args, page_range): pdf_file for pdf_file in pdf_files_to_process}

        for i, future in enumerate(as_completed(futures), 1):
            pdf_file = futures[future]
            try:
                status, filename = future.result()
                stats[status] += 1
                logger.info(f"[{i}/{total_to_process}] Finished processing {filename} with status: {status.upper()}")
            except Exception as e:
                stats['failure'] += 1
                logger.error(f"An error occurred processing {pdf_file.name}: {e}")

    return stats['success'], stats['partial'], stats['failure'] + stats['export_failure'], stats['skipped']


def main():
    args = parse_arguments()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    page_range = None
    if args.pages:
        try:
            page_range = parse_page_range(args.pages)
        except Exception as e:
            logger.error(f"Invalid page range format: {args.pages}")
            return 1

    logger.info("PDF Batch Converter starting...")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Using {args.workers} parallel workers.")

    start_time = time.time()

    success, partial, failure, skipped = process_batch(args, page_range)

    elapsed = time.time() - start_time
    total_found = success + partial + failure + skipped

    logger.info(f"\nConversion complete in {elapsed:.2f} seconds")
    logger.info(f"Total files found: {total_found}")
    logger.info(f"Processed: {success + partial + failure}")
    logger.info(f"Successful: {success}")
    logger.info(f"Partial success: {partial}")
    logger.info(f"Failed: {failure}")
    if skipped > 0:
        logger.info(f"Skipped (already processed): {skipped}")

    return 1 if failure > 0 else 0


if __name__ == "__main__":
    exit(main())
