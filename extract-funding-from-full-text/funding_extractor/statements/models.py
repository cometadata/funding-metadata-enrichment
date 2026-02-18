from typing import Optional

from pydantic import BaseModel, Field


class FundingStatement(BaseModel):
    statement: str = Field(description="The funding statement text (possibly normalized)")
    original: Optional[str] = Field(default=None, description="Original text before normalization")
    score: float = Field(description="Relevance score from semantic search")
    query: str = Field(description="Query that matched this statement")
    paragraph_idx: Optional[int] = Field(default=None, description="Index of source paragraph")
    is_problematic: bool = Field(default=False, description="Whether statement has formatting issues")
