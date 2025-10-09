from pydantic import BaseModel
from typing import Dict, Any

class ExtractRequest(BaseModel):
    pdf_path: str
    schema: Dict[str, Any]
    total_schema:Dict[str, Any]