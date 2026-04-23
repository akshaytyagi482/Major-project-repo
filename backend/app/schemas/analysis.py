from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Verdict = Literal["real", "suspicious", "fake"]


class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=20, description="News/article/body text to evaluate")
    source_url: Optional[str] = None


class UrlAnalysisRequest(BaseModel):
    url: str


class ModalityResult(BaseModel):
    modality: Literal["text", "image", "video"]
    score_fake: float = Field(..., ge=0.0, le=1.0)
    score_real: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    verdict: Verdict
    evidence: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    summary: str
    modality_results: List[ModalityResult]
    recommendations: List[str]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    env: str
