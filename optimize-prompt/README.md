# GEPA Funding Extraction Prompt Optimizer

Optimizes prompts for extracting structured funding metadata from unstructured text using [DSPy's GEPA (Genetic-Pareto) optimizer](https://dspy.ai/api/optimizers/GEPA/).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python gepa_funding_extraction_prompt_optimizer.py \
  -d input_data.json \
  -o optimized_program.dspy \
  [--train_fraction 0.5] \
  [--val_fraction 0.25] \
  [--student_model gemini/gemini-2.5-flash] \
  [--teacher_model gemini/gemini-2.5-pro] \
  [--api_key YOUR_API_KEY]
```

## Required Arguments

- `-d, --data_path`: Path to JSON file containing funding statements and expected extractions
- `-o, --output_path`: Path to save the optimized DSPy program

## Optional Arguments

- `--train_fraction`: Training data fraction (default: 0.5)
- `--val_fraction`: Validation data fraction (default: 0.25)  
- `--student_model`: Model for extraction (default: gemini-2.5-flash)
- `--teacher_model`: Model for optimization feedback (default: gemini-2.5-pro)
- `--api_key`: Gemini API key (falls back to GEMINI_API_KEY env var)

## Input Format

JSON array of objects for training and evaluation containing:
- `funding_statement`: Unstructured funding text
- `funders`: Array of funder objects with:
  - `funder_name`: Organization name
  - `funding_scheme`: Program/scheme name
  - `award_ids`: List of grant numbers
  - `award_title`: Award title (optional)

## Output

Optimized DSPy program that improves F1 score for funding metadata extraction.