from typing import List, Optional

from pydantic import BaseModel, Field


class Award(BaseModel):
    funding_scheme: List[str] = Field(default_factory=list, description="Names of funding programs or schemes")
    award_ids: List[str] = Field(default_factory=list, description="List of grant/award identifiers")
    award_title: List[str] = Field(default_factory=list, description="Titles of awards if provided")


class FunderEntity(BaseModel):
    funder_name: Optional[str] = Field(default=None, description="Name of the funding organization")
    awards: List[Award] = Field(default_factory=list, description="List of awards from this funder")


class ExtractionResult(BaseModel):
    statement: str = Field(description="The original funding statement text")
    funders: List[FunderEntity] = Field(default_factory=list, description="List of funders extracted")
