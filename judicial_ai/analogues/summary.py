from collections.abc import Iterable

from judicial_ai.corpus.records import CaseRecord
from judicial_ai.schemas.analysis import BooleanSummary, NumericSummary, PunishmentSummary


def _quantile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    fraction = position - lower
    if lower + 1 == len(sorted_values):
        return sorted_values[lower]
    return sorted_values[lower] + fraction * (
        sorted_values[lower + 1] - sorted_values[lower]
    )


def numeric_summary(values: Iterable[float | None]) -> NumericSummary:
    present = sorted(value for value in values if value is not None)
    if not present:
        return NumericSummary()
    q1 = _quantile(present, 0.25)
    q3 = _quantile(present, 0.75)
    return NumericSummary(
        count=len(present),
        median=_quantile(present, 0.5),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        min=present[0],
        max=present[-1],
    )


def boolean_summary(values: Iterable[bool | None]) -> BooleanSummary:
    present = [value for value in values if value is not None]
    if not present:
        return BooleanSummary()
    true_count = sum(present)
    return BooleanSummary(
        count=len(present),
        true_count=true_count,
        rate=true_count / len(present),
    )


def summarize_punishment(
    cohort: tuple[CaseRecord, ...], returned_count: int
) -> PunishmentSummary:
    return PunishmentSummary(
        statistics_scope="cohort",
        candidate_count=len(cohort),
        returned_count=returned_count,
        prison_months=numeric_summary(record.prison_months for record in cohort),
        fine_rub=numeric_summary(record.fine_rub for record in cohort),
        ban_years=numeric_summary(record.ban_years for record in cohort),
        conditional=boolean_summary(record.conditional for record in cohort),
        confiscation=boolean_summary(record.confiscation for record in cohort),
    )
