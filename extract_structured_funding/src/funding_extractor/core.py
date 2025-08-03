import os
import json
import concurrent.futures
from pathlib import Path

from dotenv import load_dotenv

import langextract as lx

from funding_extractor.models import (
    FundingEntity,
    FundingExtractionResult,
    ProcessingStats,
)
from funding_extractor.providers import ModelProvider

load_dotenv()


def create_extraction_prompt() -> str:
    return """You are an AI assistant specialized in extracting and analyzing funding information from research funding statements. Your task is to carefully read through a given funding statement and extract specific information as requested.

Please follow these steps to extract and present the funding information:

1. Read the provided funding statement carefully.

2. Analyze the statement and identify potential funding organizations, grant IDs, and funding schemes or programs. Present your analysis inside <extraction_analysis> tags:

<extraction_analysis>
- List all potential funding organizations found in the statement:
  - Quote the relevant part of the statement
  - Classify as "Include" or "Exclude"
  - Provide arguments for and against inclusion
  - Make a final decision with reasoning
  - Note: Include organizations providing in-kind awards of facilities

- List all potential grant IDs (including their prefixes) found in the statement:
  - Quote the relevant part of the statement
  - Classify as "Include" or "Exclude"
  - Provide arguments for and against inclusion
  - Make a final decision with reasoning
  - Note: Ensure that grant IDs are actual ID values, not named awards (e.g., "Distinguished Researcher Award" should not be included as a grant ID)

- List all potential funding schemes or programs found in the statement:
  - Quote the relevant part of the statement
  - Note which funding organization they are associated with
  - Explain why they should not be included in the final extraction

- Summarize the confirmed items for extraction:
  - List confirmed funding organizations (including those providing in-kind awards of facilities)
  - List confirmed grant IDs (only actual ID values)

This section can be quite detailed to ensure thorough analysis.
</extraction_analysis>

3. After your analysis, extract the final information according to these rules:
   - Include actual funding organizations and those providing in-kind awards of facilities.
   - Do not include research facilities or acknowledgments unless they are providing in-kind awards.
   - Include grant IDs whole, including their prefixes. Extract each grant ID only once (no duplicates).
   - Ensure that grant IDs are actual ID values, not named awards.
   - Use exact text from the source for all extracted information.
   - Do not include funding schemes or programs in the final extraction.


Remember:
- Include organizations providing in-kind awards of facilities in the funding_organizations section.
- Only include actual ID values as grant IDs, not named awards.
- Funding schemes or programs should not be included in the final extraction, even if they are mentioned in the statement. They should only be noted in your analysis to help identify the actual funding organizations.

Please proceed with your analysis and extraction."""


def create_funding_examples() -> list[lx.data.ExampleData]:
    return [
        lx.data.ExampleData(
            text=(
                "This research was funded by the National Institutes of Health "
                "through the Pioneer Award program (grant DP1 OD008486). "
                "Dr. Smith is a recipient of the Howard Hughes Medical Institute's "
                "Investigator Award. We used the services of the "
                "National Center for Macromolecular Imaging."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="funder",
                    extraction_text="National Institutes of Health",
                ),
                lx.data.Extraction(
                    extraction_class="grant_id",
                    extraction_text="DP1 OD008486",
                    attributes={"funder": "National Institutes of Health"},
                ),
                lx.data.Extraction(
                    extraction_class="funder",
                    extraction_text="Howard Hughes Medical Institute",
                ),
            ],
        ),
        lx.data.ExampleData(
            text=(
                "This project has received funding from the European Union’s Horizon "
                "2020 research and innovation programme under the Marie "
                "Skłodowska-Curie grant agreement No 812345. We acknowledge support "
                "from the German Research Foundation (DFG) under Germany's Excellence "
                "Strategy – EXC 2020 – 390657997. Data was deposited at EMBL-EBI "
                "(accession E-MTAB-1234)."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="funder", extraction_text="European Union"
                ),
                lx.data.Extraction(
                    extraction_class="grant_id",
                    extraction_text="812345",
                    attributes={"funder": "European Union"},
                ),
                lx.data.Extraction(
                    extraction_class="funder",
                    extraction_text="German Research Foundation",
                ),
                lx.data.Extraction(
                    extraction_class="grant_id",
                    extraction_text="EXC 2020 – 390657997",
                    attributes={"funder": "German Research Foundation"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text=(
                "We thank the Advanced Light Source for providing beamtime under their "
                "approved user program (ALS-0573). This work was also supported by "
                "the U.S. Department of Energy, Office of Science, under Contract "
                "No. DE-AC02-05CH11231. Analysis was performed on the "
                "Summit supercomputer at the Oak Ridge Leadership Computing Facility."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="funder",
                    extraction_text="Oak Ridge Leadership Computing Facility",
                ),
                lx.data.Extraction(
                    extraction_class="funder", extraction_text="Advanced Light Source"
                ),
                lx.data.Extraction(
                    extraction_class="grant_id",
                    extraction_text="ALS-0573",
                    attributes={"funder": "Advanced Light Source"},
                ),
                lx.data.Extraction(
                    extraction_class="funder",
                    extraction_text="U.S. Department of Energy, Office of Science",
                ),
                lx.data.Extraction(
                    extraction_class="grant_id",
                    extraction_text="DE-AC02-05CH11231",
                    attributes={
                        "funder": "U.S. Department of Energy, Office of Science"
                    },
                ),
            ],
        ),
        lx.data.ExampleData(
            text=(
                "This work was made possible by the NWO-TTW VICI (Netherlands "
                "Organisation for Scientific Research - Applied and Engineering "
                "Sciences Domain Vici Programme) grant 721.555.101. Additional "
                "support was provided by an ERC Advanced Grant (AdG-2021-101010101) "
                "from the European Research Council. We are grateful to the lab of "
                "Dr. Eva Jansen for technical assistance. Research funding was also "
                "provided by the Michael J. Fox Foundation for Parkinson's Research "
                "(MJFF Grant 98765). Authors A.B. and C.D. are employees of "
                "F. Hoffmann-La Roche Ltd, which provided financial support for this study."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="funder",
                    extraction_text="Netherlands Organisation for Scientific Research",
                    attributes={"type": "government_agency"},
                ),
                lx.data.Extraction(
                    extraction_class="grant_id",
                    extraction_text="721.555.101",
                    attributes={
                        "funder": "Netherlands Organisation for Scientific Research"
                    },
                ),
                lx.data.Extraction(
                    extraction_class="funder",
                    extraction_text="European Research Council",
                    attributes={"type": "international_agency"},
                ),
                lx.data.Extraction(
                    extraction_class="grant_id",
                    extraction_text="AdG-2021-101010101",
                    attributes={"funder": "European Research Council"},
                ),
                lx.data.Extraction(
                    extraction_class="funder",
                    extraction_text="Michael J. Fox Foundation for Parkinson's Research",
                    attributes={"type": "foundation"},
                ),
                lx.data.Extraction(
                    extraction_class="grant_id",
                    extraction_text="98765",
                    attributes={
                        "funder": "Michael J. Fox Foundation for Parkinson's Research"
                    },
                ),
                lx.data.Extraction(
                    extraction_class="funder",
                    extraction_text="F. Hoffmann-La Roche Ltd",
                    attributes={"type": "company"},
                ),
            ],
        ),
    ]


def extract_funding_from_statement(
    funding_statement: str,
    provider: ModelProvider | None = None,
    model_id: str | None = None,
    model_url: str | None = None,
    api_key: str | None = None,
    skip_model_validation: bool = False,
    timeout: int = 60,
) -> list[FundingEntity]:
    """Extract funding entities from a single funding statement.

    Args:
        funding_statement: The funding statement text to process
        provider: Model provider to use (defaults to Gemini)
        model_id: LangExtract model to use (defaults to provider's default)
        model_url: Model API URL (for Ollama)
        api_key: API key for the model (uses env var if not provided)
        skip_model_validation: Skip model validation checks
        timeout: Maximum time in seconds to wait for response (default: 60)

    Returns:
        List of extracted funding entities, or empty list if timeout occurs
    """
    from funding_extractor.providers import (
        ModelProvider,
        get_language_model_class,
        get_provider_config,
        validate_provider_requirements,
    )

    # Default to Gemini if no provider specified
    if provider is None:
        provider = ModelProvider.GEMINI

    config = get_provider_config(provider)

    validate_provider_requirements(
        provider, api_key, model_url, model_id, skip_model_validation
    )

    prompt = create_extraction_prompt()
    examples = create_funding_examples()

    extract_params = {
        "text_or_documents": funding_statement,
        "prompt_description": prompt,
        "examples": examples,
        "temperature": 0.1,
        "extraction_passes": 3,
        "max_workers": 1,
        "fence_output": False,
        "use_schema_constraints": True,
    }

    if provider == ModelProvider.GEMINI:
        language_model_class = get_language_model_class(provider)
        extract_params["language_model_type"] = language_model_class
        extract_params["model_id"] = model_id
        extract_params["api_key"] = api_key
        extract_params["language_model_params"] = {
            "timeout": timeout,
        }
    elif provider == ModelProvider.OLLAMA:
        ollama_model_class = get_language_model_class(provider)
        extract_params["language_model_type"] = ollama_model_class
        extract_params["language_model_params"] = {
            "model": model_id,
            "model_url": model_url or "http://localhost:11434",
            "timeout": timeout,
        }
    elif provider in (ModelProvider.OPENAI, ModelProvider.LOCAL_OPENAI):
        openai_model_class = get_language_model_class(provider)
        extract_params["language_model_type"] = openai_model_class
        extract_params["fence_output"] = True
        extract_params["use_schema_constraints"] = False

        if provider == ModelProvider.OPENAI:
            if model_url and not model_url.startswith("https://api.openai.com"):
                os.environ["OPENAI_BASE_URL"] = model_url
                extract_params["language_model_params"] = {
                    "model_id": model_id,
                    "api_key": api_key or "dummy-key",
                    "timeout": timeout,
                }
            else:
                os.environ.pop("OPENAI_BASE_URL", None)
                extract_params["language_model_params"] = {
                    "model_id": model_id,
                    "api_key": api_key,
                    "timeout": timeout,
                }
        else:
            if model_url:
                os.environ["OPENAI_BASE_URL"] = model_url
            extract_params["language_model_params"] = {
                "model_id": model_id,
                "api_key": api_key
                or "dummy-key",
                "timeout": timeout,
            }
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lx.extract, **extract_params)
            result = future.result(timeout=timeout)
            return _convert_extractions_to_entities(result.extractions)
    except concurrent.futures.TimeoutError:
        print(f"Request timed out after {timeout} seconds, returning empty result")
        return []
    except Exception as e:
        raise e


def _convert_extractions_to_entities(
    extractions: list[lx.data.Extraction],
) -> list[FundingEntity]:
    """Convert langextract Extraction objects to FundingEntity objects.

    Groups related extractions (funder, grant_id) into cohesive entities.
    """
    funders_map: dict[str, FundingEntity] = {}

    for extraction in extractions:
        if extraction.extraction_class == "funder":
            funder_name = extraction.extraction_text
            if funder_name not in funders_map:
                funders_map[funder_name] = FundingEntity(
                    funder=funder_name,
                    extraction_texts=[extraction.extraction_text],
                )
            else:
                funders_map[funder_name].extraction_texts.append(
                    extraction.extraction_text
                )

    for extraction in extractions:
        attrs = extraction.attributes or {}

        if extraction.extraction_class == "grant_id":
            funder_name = attrs.get("funder", "")

            if funder_name in funders_map:
                funders_map[funder_name].add_grant(extraction.extraction_text)
                funders_map[funder_name].extraction_texts.append(
                    extraction.extraction_text
                )
            else:
                entity = FundingEntity(
                    funder=funder_name or "Unknown",
                    extraction_texts=[extraction.extraction_text],
                )
                entity.add_grant(extraction.extraction_text)
                funders_map[funder_name or "Unknown"] = entity

    return list(funders_map.values())


def process_funding_file(
    input_file: Path,
    output_file: Path | None = None,
    provider: ModelProvider | None = None,
    model_id: str | None = None,
    model_url: str | None = None,
    api_key: str | None = None,
    batch_size: int = 10,
    skip_model_validation: bool = False,
    timeout: int = 60,
) -> tuple[list[FundingExtractionResult], ProcessingStats]:
    """Process a JSON file containing funding statements.

    Args:
        input_file: Path to input JSON file with funding statements
        output_file: Optional path to save results (written iteratively)
        provider: Model provider to use (defaults to Gemini)
        model_id: LangExtract model to use
        model_url: Model API URL (for Ollama)
        api_key: API key for the model
        batch_size: Number of documents to process before writing to file
        skip_model_validation: Skip model validation checks
        timeout: Maximum time in seconds to wait for each request (default: 60)

    Returns:
        Tuple of (results list, processing statistics)
    """

    with open(input_file) as f:
        data = json.load(f)

    results: list[FundingExtractionResult] = []
    stats = ProcessingStats(
        total_documents=len(data),
        successful=0,
        failed=0,
        total_entities=0,
    )

    # If output file exists, load existing results
    existing_dois = set()
    if output_file and output_file.exists():
        try:
            with open(output_file) as f:
                existing_results = json.load(f)
                for r in existing_results:
                    results.append(FundingExtractionResult(**r))
                    existing_dois.add(r["doi"])
                msg = f"Loaded {len(existing_results)} existing results from "
                print(f"{msg}{output_file}")
        except (json.JSONDecodeError, FileNotFoundError):
            print(
                f"Starting fresh - could not load existing results from {output_file}"
            )

    batch_results = []
    processed_count = len(existing_dois)

    for i, item in enumerate(data):
        doi = item.get("doi", "")

        if doi in existing_dois:
            continue

        funding_statements = item.get("funding_statements", [])

        for statement in funding_statements:
            try:
                entities = extract_funding_from_statement(
                    statement,
                    provider=provider,
                    model_id=model_id,
                    model_url=model_url,
                    api_key=api_key,
                    skip_model_validation=skip_model_validation,
                    timeout=timeout,
                )

                result = FundingExtractionResult(
                    doi=doi,
                    funding_statement=statement,
                    entities=entities,
                )
                batch_results.append(result)
                stats.successful += 1
                stats.total_entities += len(entities)
                processed_count += 1

                msg = f"[{processed_count}/{len(data)}] Processed {doi}: "
                print(f"{msg}found {len(entities)} entities")

            except Exception as e:
                print(f"Error processing {doi}: {e}")
                if "API key" in str(e):
                    print(f"API key issue - api_key provided: {api_key is not None}")
                stats.failed += 1

        if len(batch_results) >= batch_size or i == len(data) - 1:
            if batch_results:
                results.extend(batch_results)
                if output_file:
                    save_results(results, output_file)
                    print(f"Saved {len(results)} total results to {output_file}")
                batch_results = []

    return results, stats


def save_results(results: list[FundingExtractionResult], output_file: Path) -> None:
    """Save extraction results to JSON file."""
    output_data = [result.model_dump() for result in results]

    if output_file.exists():
        backup_file = output_file.with_suffix(".bak")
        output_file.rename(backup_file)

    try:
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        backup_file = output_file.with_suffix(".bak")
        if backup_file.exists():
            backup_file.unlink()
    except Exception as e:
        backup_file = output_file.with_suffix(".bak")
        if backup_file.exists():
            backup_file.rename(output_file)
        raise e
