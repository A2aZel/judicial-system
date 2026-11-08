from dataclasses import replace
from pathlib import Path

from judicial_ai.analogues.knn import KNNAnalogueFinder
from judicial_ai.analogues.summary import numeric_summary, summarize_punishment
from judicial_ai.case_card import CaseCard
from judicial_ai.corpus.csv_repository import CsvCorpusRepository
from judicial_ai.schemas.case import CaseInput

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "synthetic_analysis.csv"


def card(**overrides) -> CaseCard:
    values = {
        "article_part": "3",
        "amount_rub": 150000,
        "episodes": 1,
        "special_order": False,
        "pretrial_agreement": False,
        "art_64": False,
        "art_69": False,
        "mitigating": ["синтетический фактор"],
        "aggravating": [],
    }
    values.update(overrides)
    return CaseCard.from_input(CaseInput(**values))


def finder() -> KNNAnalogueFinder:
    return KNNAnalogueFinder(CsvCorpusRepository(FIXTURE_PATH).records)


def test_hard_filters_and_top_k() -> None:
    result = finder().search(card(), top_k=3)
    assert result.candidate_count == 5
    assert result.ranking_available is True
    assert len(result.neighbors) == 3
    assert result.active_numeric_features == (
        "amount_rub",
        "episodes",
        "mitigating_count",
        "aggravating_count",
    )
    assert all(
        "3" in neighbor.record.article_parts
        and not neighbor.record.special_order
        and not neighbor.record.pretrial_agreement
        and not neighbor.record.art_64
        and not neighbor.record.art_69
        for neighbor in result.neighbors
    )
    distances = [neighbor.distance for neighbor in result.neighbors]
    assert all(distance is not None for distance in distances)
    assert distances == sorted(distances)


def test_each_critical_regime_is_a_hard_filter() -> None:
    for field in ("special_order", "pretrial_agreement", "art_64", "art_69"):
        overrides = {field: True}
        if field == "art_69":
            overrides["episodes"] = 2
        result = finder().search(card(**overrides), top_k=10)
        assert result.candidate_count == 1
        assert getattr(result.neighbors[0].record, field) is True


def test_determinism_and_empty_result() -> None:
    first = finder().search(card(), top_k=5)
    second = finder().search(card(), top_k=5)
    assert first == second

    empty = finder().search(card(article_part="6"), top_k=5)
    assert empty.candidate_count == 0
    assert empty.cohort == ()
    assert empty.neighbors == ()
    assert any("не найдены" in warning for warning in empty.warnings)


def test_no_active_features_disables_ranking_and_summary_uses_full_cohort() -> None:
    records = tuple(
        replace(
            record,
            amount_rub=None,
            episodes=None,
            mitigating_count=None,
            aggravating_count=None,
        )
        for record in CsvCorpusRepository(FIXTURE_PATH).records
    )
    no_features = KNNAnalogueFinder(records)
    one = no_features.search(card(amount_rub=None, episodes=None, mitigating=[]), top_k=1)
    three = no_features.search(card(amount_rub=None, episodes=None, mitigating=[]), top_k=3)

    assert one.ranking_available is False
    assert len(one.neighbors) == 1
    assert all(neighbor.distance is None for neighbor in one.neighbors)
    assert any("ранжирование отключено" in warning for warning in one.warnings)
    first_summary = summarize_punishment(one.cohort, len(one.neighbors))
    second_summary = summarize_punishment(three.cohort, len(three.neighbors))
    assert first_summary.model_dump(exclude={"returned_count"}) == second_summary.model_dump(
        exclude={"returned_count"}
    )
    assert first_summary.statistics_scope == "cohort"
    assert first_summary.candidate_count == 5
    assert first_summary.returned_count == 1
    assert first_summary.prison_months.count == 5

    reversed_records = tuple(reversed(records))
    reordered = KNNAnalogueFinder(reversed_records).search(
        card(amount_rub=None, episodes=None, mitigating=[]), top_k=1
    )
    assert summarize_punishment(reordered.cohort, 1) == first_summary


def test_normal_ranking_summary_is_independent_of_top_k() -> None:
    one = finder().search(card(), top_k=1)
    five = finder().search(card(), top_k=5)
    assert summarize_punishment(one.cohort, 1) == summarize_punishment(five.cohort, 1)


def test_summary_uses_only_present_outcomes() -> None:
    summary = numeric_summary([12, None, 24, 36, 60])
    assert summary.count == 4
    assert summary.median == 30
    assert summary.q1 == 21
    assert summary.q3 == 42
    assert summary.iqr == 21
    assert summary.min == 12
    assert summary.max == 60
