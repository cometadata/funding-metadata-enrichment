from typing import Annotated, Optional

from pydantic import BaseModel, BeforeValidator, Field


def _coerce_to_list(v: object) -> object:
    """Wrap a bare string (or None) into a list so Pydantic can validate it."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return v


_StrList = Annotated[list[str], BeforeValidator(_coerce_to_list)]


class Award(BaseModel):
    funding_scheme: _StrList = Field(default_factory=list, description="Names of funding programs or schemes")
    award_ids: _StrList = Field(default_factory=list, description="List of grant/award identifiers")
    award_title: _StrList = Field(default_factory=list, description="Titles of awards if provided")


class FunderEntity(BaseModel):
    funder_name: Optional[str] = Field(default=None, description="Name of the funding organization")
    awards: list[Award] = Field(default_factory=list, description="List of awards from this funder")


class ExtractionResult(BaseModel):
    statement: str = Field(description="The original funding statement text")
    funders: list[FunderEntity] = Field(default_factory=list, description="List of funders extracted")
