from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from judicial_ai.schemas.case import CaseInput, DraftPunishment


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case: CaseInput
    draft_punishment: DraftPunishment | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class LegalCheck(BaseModel):
    code: str
    status: Literal["passed", "warning", "error", "not_checked"]
    message: str


class Analogue(BaseModel):
    distance: float | None = Field(default=None, ge=0)
    prison_months: float | None = None
    fine_rub: float | None = None
    fine_multiple: bool | None = None
    ban_on_positions: bool | None = None
    ban_years: float | None = None
    conditional: bool | None = None
    confiscation: bool | None = None


class NumericSummary(BaseModel):
    count: int = 0
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None
    min: float | None = None
    max: float | None = None


class BooleanSummary(BaseModel):
    count: int = 0
    true_count: int = 0
    rate: float | None = None


class PunishmentSummary(BaseModel):
    statistics_scope: Literal["cohort"] = "cohort"
    candidate_count: int = 0
    returned_count: int = 0
    prison_months: NumericSummary = Field(default_factory=NumericSummary)
    fine_rub: NumericSummary = Field(default_factory=NumericSummary)
    ban_years: NumericSummary = Field(default_factory=NumericSummary)
    conditional: BooleanSummary = Field(default_factory=BooleanSummary)
    confiscation: BooleanSummary = Field(default_factory=BooleanSummary)


RecommendationSource = Literal["synthetic", "case_corpus"]


class PunishmentEstimate(BaseModel):
    kind: Literal["imprisonment", "fine"]
    execution: Literal["conditional", "actual", "unspecified"] | None
    suggested_value: float = Field(gt=0)
    range_min: float = Field(gt=0)
    range_max: float = Field(gt=0)
    unit: Literal["months", "rub"]
    selected_outcomes: int = Field(gt=0)


class PunishmentRecommendation(BaseModel):
    status: Literal["available", "unavailable"] = "unavailable"
    data_source: RecommendationSource
    kind: Literal["imprisonment", "fine"] | None = None
    execution: Literal["conditional", "actual", "unspecified"] | None = None
    suggested_value: float | None = Field(default=None, gt=0)
    range_min: float | None = Field(default=None, gt=0)
    range_max: float | None = Field(default=None, gt=0)
    unit: Literal["months", "rub"] | None = None
    matching_cases: int = Field(default=0, ge=0)
    usable_outcomes: int = Field(default=0, ge=0)
    selected_outcomes: int = Field(default=0, ge=0)
    alternatives: list[PunishmentEstimate] = Field(default_factory=list)
    reasons: list[str]
    legal_status: Literal["not_checked"] = "not_checked"


class AppealRisk(BaseModel):
    status: Literal["available", "unavailable"] = "unavailable"
    score: float | None = Field(default=None, ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    model: str | None = None


class ModelInfo(BaseModel):
    engine_version: str
    knn: dict[str, object]
    bert: dict[str, object]
    dense: dict[str, object]


class AnalysisResponse(BaseModel):
    analysis_id: str
    legal_checks: list[LegalCheck]
    analogues: list[Analogue]
    punishment_summary: PunishmentSummary
    recommendation: PunishmentRecommendation
    appeal_risk: AppealRisk
    warnings: list[str]
    model_info: ModelInfo
