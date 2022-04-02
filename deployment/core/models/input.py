from typing import List

from pydantic import BaseModel, Field, validator


class MessageInput(BaseModel):

    V1: float = Field(..., title="V1 feature")
    V2: float = Field(..., title="V2 feature")
    V3: float = Field(..., title="V3 feature")
    V4: float = Field(..., title="V4 feature")
    V5: float = Field(..., title="V5 feature")
    V7: float = Field(..., title="V7 feature")
    V9: float = Field(..., title="V9 feature")
    V10: float = Field(..., title="V10 feature")
    V11: float = Field(..., title="V11 feature")
    V12: float = Field(..., title="V12 feature")
    V13: float = Field(..., title="V13 feature")
    V16: float = Field(..., title="V16 feature")
    V17: float = Field(..., title="V17 feature")
    V18: float = Field(..., title="V18 feature")
    V1_V2: float = Field(..., title="V1_V2 feature")
    V1_V3: float = Field(..., title="V1_V3 feature")
    V1_V4: float = Field(..., title="V1_V4 feature")
    V1_V5: float = Field(..., title="V1_V5 feature")
    V1_V7: float = Field(..., title="V1_V7 feature")
    V1_V9: float = Field(..., title="V1_V9 feature")
    V1_V10: float = Field(..., title="V1_V10 feature")
    V1_V11: float = Field(..., title="V1_V11 feature")
    V1_V12: float = Field(..., title="V1_V12 feature")
    V1_V13: float = Field(..., title="V1_V13 feature")
    V1_V16: float = Field(..., title="V1_V16 feature")
    V1_V17: float = Field(..., title="V1_V17 feature")
    V1_V18: float = Field(..., title="V1_V18 feature")
    V2_V3: float = Field(..., title="V2_V3 feature")
    V2_V4: float = Field(..., title="V2_V4 feature")
    V2_V5: float = Field(..., title="V2_V5 feature")
    V2_V7: float = Field(..., title="V2_V7 feature")
    diff: float = Field(..., title="Time since the last transaction")
