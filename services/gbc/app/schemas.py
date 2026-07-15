from pydantic import BaseModel, Field


class PresignRequest(BaseModel):
    fileName: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0)
    contentType: str = Field(default="text/plain", max_length=100)


class AnalyzeRequest(BaseModel):
    objectKey: str = Field(pattern=r"^uploads/[0-9a-f]{32}\.txt$")
    threshold: float = Field(ge=0.01, le=1.0)
