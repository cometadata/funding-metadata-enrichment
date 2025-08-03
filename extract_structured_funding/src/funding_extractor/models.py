from typing import NewType

from pydantic import BaseModel, Field

FunderId = NewType("FunderId", str)
GrantId = NewType("GrantId", str)


class Grant(BaseModel):
    """Represents a specific grant or award."""

    grant_id: str = Field(description="Grant or award identifier")


class FundingEntity(BaseModel):
    """Represents a funding entity with potentially multiple grants."""

    funder: str = Field(description="Name of the funding organization")
    grants: list[Grant] = Field(
        default_factory=list, description="List of grants from this funder"
    )
    extraction_texts: list[str] = Field(
        default_factory=list, description="Original texts where entity was found"
    )

    def add_grant(self, grant_id: str) -> None:
        for grant in self.grants:
            if grant.grant_id == grant_id:
                return
        self.grants.append(Grant(grant_id=grant_id))


class FundingExtractionResult(BaseModel):
    """Result of funding extraction from a document."""

    doi: str = Field(description="Document identifier")
    funding_statement: str = Field(description="Original funding statement text")
    entities: list[FundingEntity] = Field(
        default_factory=list, description="Extracted funding entities"
    )


class ProcessingStats(BaseModel):
    """Statistics for processing results."""

    total_documents: int = Field(description="Total documents processed")
    successful: int = Field(description="Successfully processed documents")
    failed: int = Field(description="Failed documents")
    total_entities: int = Field(description="Total entities extracted")
