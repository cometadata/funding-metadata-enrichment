import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GoldAward:
    funding_schemes: List[str] = field(default_factory=list)
    award_ids: List[str] = field(default_factory=list)
    award_titles: List[str] = field(default_factory=list)


@dataclass
class GoldFunder:
    funder_name: Optional[str] = None
    awards: List[GoldAward] = field(default_factory=list)


@dataclass
class GoldDocument:
    doi: str = ""
    funding_statement: str = ""
    funders: List[GoldFunder] = field(default_factory=list)
    markdown: str = ""


def normalize_doi(doi: str) -> str:
    doi = doi.strip()
    doi = re.sub(r"^https?://doi\.org/", "", doi)
    doi = re.sub(r"^doi:", "", doi, flags=re.IGNORECASE)
    return doi.lower()


def _hf_row_to_gold_document(row: dict) -> GoldDocument:
    funders = []
    for funder_data in row.get("funders", []):
        awards = []
        for award_data in funder_data.get("awards", []):
            awards.append(GoldAward(
                funding_schemes=award_data.get("funding_scheme", []),
                award_ids=award_data.get("award_ids", []),
                award_titles=award_data.get("award_title", []),
            ))
        funders.append(GoldFunder(
            funder_name=funder_data.get("funder_name"),
            awards=awards,
        ))
    return GoldDocument(
        doi=row["doi"],
        funding_statement=row["funding_statement"],
        funders=funders,
        markdown=row["markdown"],
    )


def load_hf_dataset(
    dataset_id: str,
    split: str = "test",
    max_samples: Optional[int] = None,
) -> List[GoldDocument]:
    from datasets import load_dataset

    if split == "both":
        splits_to_load = ["train", "test"]
    else:
        splits_to_load = [split]

    documents: List[GoldDocument] = []
    for s in splits_to_load:
        ds = load_dataset(dataset_id, split=s)
        for i, row in enumerate(ds):
            if max_samples is not None and len(documents) >= max_samples:
                break
            documents.append(_hf_row_to_gold_document(row))
        if max_samples is not None and len(documents) >= max_samples:
            break

    return documents


def build_gold_lookup(documents: List[GoldDocument]) -> Dict[str, GoldDocument]:
    return {normalize_doi(doc.doi): doc for doc in documents}
