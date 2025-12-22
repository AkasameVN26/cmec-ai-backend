from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union

class SourceSegment(BaseModel):
    content: str
    source_type: str  # ADMIN, LICH_KHAM, CLS, DON_THUOC, GHI_CHU, CHAN_DOAN
    source_id: Optional[Union[str, int]] = None

class PreparedInput(BaseModel):
    id_ho_so: int
    prompt_content: str
    raw_notes_count: int
    source_segments: List[SourceSegment] = []

class EvaluationMetrics(BaseModel):
    duration: float
    input_tokens: int
    output_tokens: int
    ratio: float
    similarity_score: float

class SummaryResponse(BaseModel):
    id_ho_so: int
    source: str
    summary: str
    metrics: Optional[EvaluationMetrics] = None

class MatchDetail(BaseModel):
    summary_idx: int
    source_indices: List[int]
    scores: List[float]

class ExplainResponse(BaseModel):
    notes: List[SourceSegment]
    summary_sentences: List[str]
    matches: List[MatchDetail]
    avg_similarity_score: float = 0.0
    low_similarity_matches: List[MatchDetail] = []

class ExplainRequest(BaseModel):
    summary: str