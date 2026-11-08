import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

Factor = Annotated[str, Field(min_length=1, max_length=200)]


class CaseInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    article_part: str = Field(min_length=1, max_length=3)
    amount_rub: float | None = Field(default=None, ge=0, le=1_000_000_000_000)
    episodes: int | None = Field(default=None, ge=1, le=1000)
    special_order: bool
    pretrial_agreement: bool
    art_64: bool
    art_69: bool
    mitigating: list[Factor] = Field(default_factory=list, max_length=30)
    aggravating: list[Factor] = Field(default_factory=list, max_length=30)
    narrative: str | None = Field(default=None, max_length=4000)

    @field_validator("article_part")
    @classmethod
    def validate_article_part(cls, value: str) -> str:
        if not re.fullmatch(r"\d(?:\.\d)?", value):
            raise ValueError("article_part must have the form '3' or '5.1'")
        return value


class DraftPunishment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prison_months: float | None = Field(default=None, ge=0, le=1200)
    fine_rub: float | None = Field(default=None, ge=0, le=1_000_000_000_000)
    fine_multiple: float | None = Field(default=None, gt=0, le=1000)
    ban_years: float | None = Field(default=None, ge=0, le=100)
    conditional: bool = False
    confiscation: bool = False
