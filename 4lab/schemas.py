from pydantic import BaseModel, Field
from typing import Optional

class AgentResponse(BaseModel):
    success: Optional[bool] = True
    answer: Optional[str] = None
    csv_path: Optional[str] = None
    skill_used: Optional[str] = None
    used_tools: Optional[list[str]] = []