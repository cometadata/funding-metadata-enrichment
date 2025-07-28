### Funding Metadata Extraction from Full Text Pipeline

This pipeline extracts funding acknowledgements from PDF documents, converting them into structured metadata linking them to their DOIs.

The extraction begins with the [batch_convert_pdfs utility](https://github.com/cometadata/funding-metadata-enrichment/tree/main/extract_funding_from_full_text/batch_convert_pdfs). This tool processes PDF files using [Docling](https://github.com/docling-project/docling), converting them to multiple formats, including markdown.

Once the PDFs are converted to markdown, [extract_funding_w_reranker utility](https://github.com/cometadata/funding-metadata-enrichment/tree/main/extract_funding_from_full_text/extract_funding_w_reranker) is used to extract the funding statements. Here, we employ a two-stage approach to identify and extract:

1. A ColBERT model ([`GTE-ModernColBERT-v1`](https://huggingface.co/lightonai/GTE-ModernColBERT-v1)) embeds and scores each paragraph in the document against a set of funding-related terms/queries, identifying sections of the text with high semantic similarity to funding acknowledgements.

2. High-scoring paragraphs are then confirmed with some regex pattern matching to confirm they contain specific funding language, e.g. "awarded by" "funded by", "grant number", or "supported by". Only texts that pass both stages are classified as a funding statement.


Finally, we use [convert_normalize_ranking_extraction_json utility](https://github.com/cometadata/funding-metadata-enrichment/tree/main/extract_funding_from_full_text/convert_normalize_ranking_extraction_json) to post-processes the raw extraction output. This stage maps funding statements to DOIs by parsing and comparing them against the filenames, as well as optionally normalizes the text of the funding statements. We reconcile the filenames with DOIs using a reference CSV file that contains all those for the input and then separates problematic entries (primarily those containing failed text parsing) into a separate output file.
