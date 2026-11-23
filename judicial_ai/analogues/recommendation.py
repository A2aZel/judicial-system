from dataclasses import replace
from math import isfinite
from typing import Literal

from judicial_ai.analogues.knn import KNNAnalogueFinder
from judicial_ai.analogues.summary import numeric_summary
from judicial_ai.case_card import CaseCard
from judicial_ai.corpus.records import CaseRecord
from judicial_ai.schemas.analysis import (
    PunishmentEstimate,
    PunishmentRecommendation,
    RecommendationSource,
)

MAX_NEIGHBORS = 20
MIN_SUPPORT = 3
# A threefold window keeps amounts an order of magnitude apart in disjoint cohorts.
AMOUNT_RATIO_LIMIT = 3
FEATURE_NAMES = {
    "amount_rub": "сумма взятки в рублях",
    "episodes": "число эпизодов",
    "mitigating_count": "число смягчающих обстоятельств",
    "aggravating_count": "число отягчающих обстоятельств",
}


def recommend_punishment(
    case: CaseCard, structured_cohort: tuple[CaseRecord, ...], data_source: RecommendationSource
) -> PunishmentRecommendation:
    exact_cohort = _distinct_cases(
        tuple(record for record in structured_cohort if record.article_parts == (case.article_part,))
    )
    matching = len(exact_cohort)
    amount = case.amount_rub
    if not matching:
        reasons = ["Нет дел с той же единственной частью ст. 290 и совпадающими режимами."]
        if amount is not None:
            reasons.append(_amount_matching_reason(amount, 0, 0))
        return PunishmentRecommendation(data_source=data_source, reasons=reasons)
    matching_reason = (
        f"Выбраны дела с единственной частью {case.article_part} ст. 290 и совпадающими "
        "признаками особого порядка, досудебного соглашения, ст. 64 и ст. 69."
    )
    if case.art_69:
        return PunishmentRecommendation(
            data_source=data_source,
            matching_cases=matching,
            reasons=["Ориентир по отдельному составу недоступен при совокупности преступлений по ст. 69."],
        )
    reasons = [matching_reason]
    if amount is not None:
        ranking_cohort = _amount_cohort(exact_cohort, amount)
        reasons.append(_amount_matching_reason(amount, len(ranking_cohort), matching))
        matching = len(ranking_cohort)
        if not matching:
            return PunishmentRecommendation(data_source=data_source, reasons=reasons)
    else:
        reasons.append("Сумма взятки в запросе не указана.")
        ranking_cohort = tuple(record for record in exact_cohort if record.recommendation_facts_trusted)

    if ranking_cohort:
        structured = KNNAnalogueFinder(ranking_cohort).search(case, MAX_NEIGHBORS)
        if not structured.ranking_available:
            return PunishmentRecommendation(
                data_source=data_source,
                matching_cases=matching,
                reasons=[*reasons, "Для ранжирования по числовым фактам дела нет данных."],
            )
        features = []
        for name in structured.active_numeric_features:
            value = getattr(case, name)
            label = FEATURE_NAMES[name]
            features.append(
                f"{label} — {value:.15g}" if value is not None else f"{label} — медиана выборки"
            )
        reasons.append("Признаки сравнения — " + ", ".join(features) + ".")
        selected_cases = tuple(neighbor.record for neighbor in structured.neighbors)
        sample_description = "ближайших дел"
    else:
        reasons.append(
            "Числовые признаки из полных приговоров требуют проверки. "
            "Историческая оценка учитывает всю группу после отбора по части статьи и режимам."
        )
        selected_cases = exact_cohort
        sample_description = "отобранных дел"

    terms: list[float] = []
    term_conditionals: set[bool | None] = set()
    fines: list[float] = []
    for record in selected_cases:
        value = record.principal_value
        if value is None or not isfinite(value) or value <= 0:
            continue
        if record.principal_punishment == "imprisonment":
            terms.append(value)
            term_conditionals.add(record.principal_conditional)
        elif record.principal_punishment == "fine":
            fines.append(value)

    usable = len(terms) + len(fines)
    reasons.append(
        f"Среди {sample_description} ({len(selected_cases)}) удалось учесть исходы {usable} дел."
    )
    imprisonment = len(terms) > len(fines)
    selected = terms if imprisonment else fines
    calculation_reason = (
        "Ориентир — медиана выбранных исходов. Диапазон — от первого до третьего квартиля (Q1–Q3)."
    )
    if len(terms) == len(fines) and len(terms) >= MIN_SUPPORT:
        reasons.extend((
            f"Срок и штраф встречались одинаково часто, по {len(terms)} дел. "
            "Показаны оценки обоих исходов.",
            calculation_reason,
        ))
        return PunishmentRecommendation(
            data_source=data_source,
            matching_cases=matching,
            usable_outcomes=usable,
            alternatives=[_estimate(terms, term_conditionals), _estimate(fines, None)],
            reasons=reasons,
        )
    if len(selected) < MIN_SUPPORT:
        reasons.append(
            "Для оценки нужны как минимум три исхода одного вида основного наказания "
            "и большинство этого вида среди учтённых исходов."
        )
        return PunishmentRecommendation(
            data_source=data_source,
            matching_cases=matching,
            usable_outcomes=usable,
            reasons=reasons,
        )

    estimate = _estimate(selected, term_conditionals if imprisonment else None)
    punishment = "штраф"
    if imprisonment:
        punishment = "лишение свободы"
        if estimate.execution == "conditional":
            punishment = "условное лишение свободы"
        elif estimate.execution == "actual":
            punishment = "лишение свободы с отбыванием"
    reasons.extend((
        f"Основное наказание — {punishment} в {len(selected)} из {usable} учтённых дел.",
        calculation_reason,
    ))
    return PunishmentRecommendation(
        **estimate.model_dump(),
        status="available",
        data_source=data_source,
        matching_cases=matching,
        usable_outcomes=usable,
        reasons=reasons,
    )


def _amount_cohort(records: tuple[CaseRecord, ...], target: float) -> tuple[CaseRecord, ...]:
    if not isfinite(target) or target <= 0:
        return ()
    lower_bound = target / AMOUNT_RATIO_LIMIT
    upper_bound = target * AMOUNT_RATIO_LIMIT
    matched: list[tuple[CaseRecord, float]] = []
    for record in records:
        amount = record.recommendation_amount_rub
        if amount is None and record.recommendation_facts_trusted:
            amount = record.amount_rub
        if amount is None or not isfinite(amount) or amount <= 0:
            continue
        if lower_bound <= amount <= upper_bound:
            matched.append((record, amount))
    if all(record.recommendation_facts_trusted for record, _ in matched):
        return tuple(replace(record, amount_rub=amount) for record, amount in matched)
    # Compare the whole group by amount when other facts are not consistently trusted.
    return tuple(
        replace(record, amount_rub=amount, episodes=None, mitigating_count=None, aggravating_count=None)
        for record, amount in matched
    )


def _amount_matching_reason(amount: float, matched: int, total: int) -> str:
    if not isfinite(amount) or amount <= 0:
        criterion = "Для подбора нужна положительная сумма взятки."
    else:
        criterion = (
            "Подбор требует подтверждённой суммы взятки, которая отличается от указанной "
            f"максимум в {AMOUNT_RATIO_LIMIT} раза."
        )
    return f"Для суммы {amount:.15g} руб. подошло {matched} из {total} дел. {criterion}"


def _estimate(values: list[float], term_conditionals: set[bool | None] | None) -> PunishmentEstimate:
    summary = numeric_summary(values)
    execution: Literal["conditional", "actual", "unspecified"] | None = None
    if term_conditionals is not None:
        if term_conditionals == {True}:
            execution = "conditional"
        elif term_conditionals == {False}:
            execution = "actual"
        else:
            execution = "unspecified"
    return PunishmentEstimate(
        kind="imprisonment" if term_conditionals is not None else "fine",
        execution=execution,
        suggested_value=summary.median,
        range_min=summary.q1,
        range_max=summary.q3,
        unit="months" if term_conditionals is not None else "rub",
        selected_outcomes=summary.count,
    )


def _distinct_cases(records: tuple[CaseRecord, ...]) -> tuple[CaseRecord, ...]:
    parents = list(range(len(records)))
    identities: dict[tuple[str, str], int] = {}

    def representative(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, record in enumerate(records):
        for namespace, value in (
            ("group", record.recommendation_group),
            ("text", record.recommendation_text_hash),
        ):
            if value is not None:
                left = representative(identities.setdefault((namespace, value), index))
                right = representative(index)
                parents[max(left, right)] = min(left, right)

    return tuple(record for index, record in enumerate(records) if representative(index) == index)
