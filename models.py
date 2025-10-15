from pydantic import BaseModel
from typing import Dict, Any, Optional

class ExtractRequest(BaseModel):
    pdf_path: str
    schema: Optional[Dict[str, Any]] = None
    total_schema: Optional[Dict[str, Any]] = None