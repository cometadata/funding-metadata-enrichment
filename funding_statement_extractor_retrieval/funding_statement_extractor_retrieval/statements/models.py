from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FundingStatement(BaseModel):
    statement: str
    original: Optional[str] = None
    score: float
    query: str
    paragraph_idx: Optional[int] = None
    is_problematic: bool = False


class DocumentResult(BaseModel):
    filename: str
    funding_statements: List[FundingStatement] = Field(default_factory=list)

    def has_funding(self) -> bool:
        return len(self.funding_statements) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "funding_statements": [
                {
                    "statement": stmt.statement,
                    "original": stmt.original,
                    "score": stmt.score,
                    "query": stmt.query,
                    "is_problematic": stmt.is_problematic,
                }
                for stmt in self.funding_statements
            ],
        }


class ProcessingParameters(BaseModel):
    input_path: str
    input_format: str = "markdown"
    normalize: bool = False
    threshold: float = 28.0
    top_k: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_path": self.input_path,
            "input_format": self.input_format,
            "normalize": self.normalize,
            "threshold": self.threshold,
            "top_k": self.top_k,
        }


class ProcessingResults(BaseModel):
    timestamp: str
    parameters: ProcessingParameters
    results: Dict[str, DocumentResult]
    summary: Dict[str, Any]

    def update_summary(self) -> None:
        total_files = len(self.results)
        files_with_funding = sum(1 for doc in self.results.values() if doc.has_funding())
        total_statements = sum(len(doc.funding_statements) for doc in self.results.values())

        self.summary = {
            "total_files": total_files,
            "files_with_funding": files_with_funding,
            "total_statements": total_statements,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "parameters": self.parameters.to_dict(),
            "results": {filename: doc.to_dict() for filename, doc in self.results.items()},
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingResults":
        parameters = ProcessingParameters(**data["parameters"])
        results: Dict[str, DocumentResult] = {}

        for filename, doc_data in data.get("results", {}).items():
            doc = DocumentResult(filename=filename)
            for stmt_data in doc_data.get("funding_statements", []):
                stmt = FundingStatement(
                    statement=stmt_data["statement"],
                    original=stmt_data.get("original"),
                    score=stmt_data.get("score", 0.0),
                    query=stmt_data.get("query", ""),
                    is_problematic=stmt_data.get("is_problematic", False),
                )
                doc.funding_statements.append(stmt)

            results[filename] = doc

        return cls(
            timestamp=data["timestamp"],
            parameters=parameters,
            results=results,
            summary=data.get("summary", {}),
        )


class ProcessingStats(BaseModel):
    total_documents: int
    successful: int
    failed: int
