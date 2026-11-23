from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class CaseRecord:
    article_parts: tuple[str, ...]
    special_order: bool | None
    pretrial_agreement: bool | None
    art_64: bool | None
    art_69: bool | None
    amount_rub: float | None = None
    episodes: int | None = None
    mitigating_count: int | None = None
    aggravating_count: int | None = None
    prison_months: float | None = None
    fine_rub: float | None = None
    fine_multiple: bool | None = None
    ban_on_positions: bool | None = None
    ban_years: float | None = None
    conditional: bool | None = None
    confiscation: bool | None = None
    source_order: int = 0
    principal_punishment: Literal["imprisonment", "fine"] | None = None
    principal_value: float | None = None
    principal_conditional: bool | None = None
    recommendation_facts_trusted: bool = True
    recommendation_group: str | None = None
    recommendation_text_hash: str | None = None
    recommendation_amount_rub: float | None = None
    recommendation_mitigating: frozenset[str] | None = None
    recommendation_aggravating: frozenset[str] | None = None
