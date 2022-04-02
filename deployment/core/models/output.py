from pydantic import BaseModel, Field


class MessageOutput(BaseModel):
    fraud: int = Field(..., title="hard decision, after threshold")
    probability: float = Field(..., title="soft decision, before threshold")
