import math
from dataclasses import replace

import pytest

from judicial_ai.corpus.processed_repository import ProcessedRepository
from tools.evaluate_knn import (
    Candidate,
    QueryExample,
    eligible_candidates,
    known_median,
    lexical_rank,
    neighbor_median,
    regression_metrics,
)


def synthetic_case(order: int, narrative: str, prison_months=None, fine_rub=None):
    payload = {
        "schema_version": 1,
        "case_id": f"synthetic-{order}",
        "article_parts": ["3"],
        "process": {
            "special_order": False,
            "pretrial_agreement": False,
            "art_64": False,
            "art_69": False,
        },
        "facts": {
            "amount_rub": 100 + order,
            "episodes": 1,
            "mitigating": [],
            "aggravating": [],
            "narrative": narrative,
        },
        "punishment": {
            "prison_months": prison_months,
            "fine_rub": fine_rub,
            "conditional": False,
            "confiscation": False,
        },
    }
    return ProcessedRepository._parse(payload, order)


def candidate(
    order: int,
    case_id: str,
    group: str,
    narrative: str,
    prison_months=None,
    fine_rub=None,
) -> Candidate:
    return Candidate(
        case_id,
        group,
        synthetic_case(order, narrative, prison_months, fine_rub),
    )


def query(case_id: str = "query", group: str = "query-group", narrative: str = "alpha") -> QueryExample:
    processed = synthetic_case(100, narrative, 24, 1000)
    record = processed.record
    from judicial_ai.case_card import CaseCard

    card = CaseCard(
        article_part="3",
        amount_rub=record.amount_rub,
        episodes=record.episodes,
        special_order=record.special_order,
        pretrial_agreement=record.pretrial_agreement,
        art_64=record.art_64,
        art_69=record.art_69,
        mitigating_count=record.mitigating_count,
        aggravating_count=record.aggravating_count,
        narrative=narrative,
    )
    return QueryExample(
        case_id=case_id,
        verdict_group_id=group,
        outcome="changed",
        article_parts=("3",),
        article_part="3",
        card=card,
        narrative=narrative,
        text_sha256=processed.text_sha256,
        prison_months=record.prison_months,
        fine_rub=record.fine_rub,
    )


def test_self_exclusion_removes_case_hash_and_entire_verdict_group() -> None:
    subject = query()
    same_group = candidate(0, "other-id", "query-group", "different text")
    same_hash = candidate(1, "other-id", "other-group", "alpha")
    same_case = candidate(2, "query", "other-group", "different text")
    safe = candidate(3, "safe-id", "safe-group", "safe text")

    eligible = eligible_candidates(subject, (same_group, same_hash, same_case, safe))

    assert eligible == (safe,)


def test_known_median_does_not_impute_missing_targets_as_zero() -> None:
    records = [
        synthetic_case(0, "a", prison_months=12).record,
        synthetic_case(1, "b", prison_months=None).record,
        synthetic_case(2, "c", prison_months=24).record,
    ]

    assert known_median(records, "prison_months") == 18
    assert known_median(records[1:2], "prison_months") is None
    assert known_median(records, "fine_rub") is None


def test_metrics_use_only_known_targets_and_preserve_known_zero() -> None:
    rows = [
        {"actual": {"fine_rub": 0}, "predicted": {"fine_rub": 0}},
        {"actual": {"fine_rub": 99}, "predicted": {"fine_rub": 9}},
        {"actual": {"fine_rub": None}, "predicted": {"fine_rub": 999}},
        {"actual": {"fine_rub": 5}, "predicted": {"fine_rub": None}},
    ]

    result = regression_metrics(rows, "fine_rub")

    assert result["n_queries"] == 4
    assert result["n_known"] == 3
    assert result["n_scored"] == 2
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["mae"] == pytest.approx(math.log(10) / 2)
    assert result["rmse"] == pytest.approx(math.log(10) / math.sqrt(2))
    assert result["median_absolute_error"] == pytest.approx(math.log(10) / 2)


def test_cohort_and_neighbor_medians_are_correct_without_zero_fill() -> None:
    candidates = [
        candidate(0, "a", "ga", "a", prison_months=10, fine_rub=None),
        candidate(1, "b", "gb", "b", prison_months=20, fine_rub=1000),
        candidate(2, "c", "gc", "c", prison_months=100, fine_rub=None),
    ]

    assert known_median((item.case.record for item in candidates), "prison_months") == 20
    neighbors = lexical_rank("a", candidates, top_k=2)
    assert neighbor_median(neighbors, "fine_rub") in {None, 1000}
    assert neighbor_median(neighbors[:1], "prison_months") == 10


def test_bm25_like_baseline_prefers_matching_text_and_has_finite_distances() -> None:
    matching = candidate(0, "match", "gm", "повторяемый факт особый эпизод")
    other_match = candidate(1, "other", "go", "особый эпизод иные обстоятельства")
    unrelated = candidate(2, "unrelated", "gu", "совершенно постороннее описание")

    ranked = lexical_rank("особый эпизод", (unrelated, other_match, matching), top_k=3)

    assert [neighbor.record.source_order for neighbor in ranked[:2]] == [0, 1]
    assert ranked[-1].record.source_order == 2
    assert all(neighbor.distance is not None and math.isfinite(neighbor.distance) for neighbor in ranked)
    assert [neighbor.distance for neighbor in ranked] == sorted(neighbor.distance for neighbor in ranked)


def test_missing_actual_target_is_not_scored_even_when_prediction_is_zero() -> None:
    rows = [
        {"actual": {"prison_months": None}, "predicted": {"prison_months": 0}},
        {"actual": {"prison_months": 0}, "predicted": {"prison_months": 12}},
    ]

    result = regression_metrics(rows, "prison_months")

    assert result["n_known"] == 1
    assert result["n_scored"] == 1
    assert result["mae"] == 12
    assert result["rmse"] == 12
    assert result["median_absolute_error"] == 12


def test_no_known_target_has_null_metrics_instead_of_zero_error() -> None:
    rows = [{"actual": {"fine_rub": None}, "predicted": {"fine_rub": 0}}]

    result = regression_metrics(rows, "fine_rub")

    assert result["n_known"] == 0
    assert result["n_scored"] == 0
    assert result["coverage"] is None
    assert result["mae"] is None
    assert result["rmse"] is None
    assert result["median_absolute_error"] is None


def test_lexical_empty_query_is_deterministic_uninformative_baseline() -> None:
    candidates = [
        candidate(1, "b", "gb", "second"),
        candidate(0, "a", "ga", "first"),
    ]

    ranked = lexical_rank("123 !!!", candidates, top_k=2)

    assert [neighbor.record.source_order for neighbor in ranked] == [1, 0]
    assert all(neighbor.distance == 1 for neighbor in ranked)


def test_median_uses_only_selected_top_k_records() -> None:
    selected = candidate(0, "a", "ga", "a", prison_months=12)
    not_selected = candidate(1, "b", "gb", "b", prison_months=1200)
    neighbors = lexical_rank("a", (selected, not_selected), top_k=1)

    assert len(neighbors) == 1
    assert neighbor_median(neighbors, "prison_months") == 12


def test_candidate_record_replacement_keeps_none_distinct_from_zero() -> None:
    processed = synthetic_case(0, "a", fine_rub=None)
    zero = replace(processed.record, fine_rub=0)

    assert known_median((processed.record,), "fine_rub") is None
    assert known_median((zero,), "fine_rub") == 0
