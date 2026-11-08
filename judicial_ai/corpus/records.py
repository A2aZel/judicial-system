from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CaseRecord:
    article_parts: tuple[str, ...]
    special_order: bool
    pretrial_agreement: bool
    art_64: bool
    art_69: bool
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
