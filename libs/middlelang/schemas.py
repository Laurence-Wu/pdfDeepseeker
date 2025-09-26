# Placeholder for middle-format definitions (Pydantic models)
from pydantic import BaseModel

class PageBlock(BaseModel):
    type: str
    content: str
