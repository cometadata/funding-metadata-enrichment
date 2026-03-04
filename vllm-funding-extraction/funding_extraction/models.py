from typing import Optional

from pydantic import BaseModel, Field


class Award(BaseModel):
    funding_scheme: list[str] = Field(default_factory=list, description="Names of funding programs or schemes")
    award_ids: list[str] = Field(default_factory=list, description="List of grant/award identifiers")
    award_title: list[str] = Field(default_factory=list, description="Titles of awards if provided")


class FunderEntity(BaseModel):
    funder_name: Optional[str] = Field(default=None, description="Name of the funding organization")
    awards: list[Award] = Field(default_factory=list, description="List of awards from this funder")


class ExtractionResult(BaseModel):
    statement: str = Field(description="The original funding statement text")
    funders: list[FunderEntity] = Field(default_factory=list, description="List of funders extracted")
