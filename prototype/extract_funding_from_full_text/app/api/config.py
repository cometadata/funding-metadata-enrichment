import os
from pathlib import Path
from typing import Dict, Any
import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ColBERT Model Used for re-ranking
    model_name: str = "lightonai/GTE-ModernColBERT-v1"
    batch_size: int = 32
    
    top_k: int = 5
    threshold: float = 28.0
    
    api_title: str = "COMET Funding Statement Extraction API"
    api_version: str = "1.0.0"
    api_description: str = "Extract funding acknowledgements from markdown documents"
    
    query_file: str = "funding_queries.yaml"
    
    normalize_statements: bool = True
    exclude_problematic: bool = False
    
    class Config:
        env_prefix = "FUNDING_API_"
        case_sensitive = False


settings = Settings()


def load_queries(query_file: str = None) -> Dict[str, str]:
    if query_file is None:
        query_file = settings.query_file
    
    query_path = Path(query_file)
    if not query_path.exists():
        query_path = Path(__file__).parent / query_file
    
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_file}")
    
    with open(query_path, 'r') as f:
        queries_data = yaml.safe_load(f)
    
    if not queries_data or 'queries' not in queries_data:
        raise ValueError("Query file must contain a 'queries' key with list of queries")
    
    return queries_data['queries']


# Funding patterns used for validation
FUNDING_PATTERNS = [
    r'\backnowledg\w*\s+(?:funding|financial|support)',
    r'\bfund\w*\s+(?:by|from|through)',
    r'\bsupport\w*\s+(?:by|from|through)',
    r'\bgrant\w*\s+(?:from|by|number|no\.?|#)',
    r'\baward\w*\s+(?:from|by|number|no\.?|#)',
    r'\bproject\s+(?:number|no\.?|#)',
    r'\bcontract\s+(?:number|no\.?|#)',
    r'\bfinancial\w*\s+support',
    r'\bresearch\w*\s+(?:fund|support)',
    r'\bthis\s+(?:work|research|study)\s+(?:was|is)\s+(?:supported|funded)',
    r'\bgrateful\w*\s+(?:for|to).*(?:fund|support)',
    r'\bthank\w*\s+(?:for|to).*(?:fund|support)',
]