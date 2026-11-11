import json

import pytest

from tools import recover_vrn_150 as mod


def candidate():
    return {
        "first_court": "центральный районный суд г воронежа",
        "first_case_number": "1-69/2024",
        "first_decision_date": "2024-10-01",
        "appeal_decision_date": "2025-02-10",
    }


def document():
    return {
        "title": "Приговор № 1-69/2024 от 1 октября 2024 г.",
        "court": "Центральный районный суд г. Воронежа",
        "text": (
            "П Р И Г О В О Р\nИМЕНЕМ РОССИЙСКОЙ ФЕДЕРАЦИИ\n"
            "г. Воронеж 1 октября 2024 года\n"
            "Центральный районный суд г. Воронежа по делу № 1-69/2024\n"
            "установил: " + ("обстоятельства дела. " * 300) + "\nприговорил: назначить наказание."
        ),
    }


def test_strict_verdict_accepts_matching_full_document():
    assert mod.verdict_errors(candidate(), document()) == []


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("first_case_number", "1-70/2024", "case_number_not_in_header"),
        ("first_court", "ленинский районный суд г воронежа", "court_not_in_header"),
        ("first_decision_date", "2024-10-02", "decision_date_not_in_header"),
        ("appeal_decision_date", "2024-09-30", "not_earlier_than_appeal"),
    ],
)
def test_strict_verdict_rejects_identity_mismatch(field, value, error):
    row = candidate()
    row[field] = value
    assert error in mod.verdict_errors(row, document())


def test_strict_verdict_rejects_new_appellate_verdict():
    doc = document()
    doc["title"] = "Апелляционный приговор № 1-69/2024"
    assert "not_trial_verdict_title" in mod.verdict_errors(candidate(), doc)


def test_url_policy_allows_only_direct_regular_documents():
    assert mod.allowed_document_url("https://sudact.ru/regular/doc/Ab_1-2/") == "Ab_1-2"
    for url in (
        "https://sudact.ru/regular/doc_ajax/Ab_1-2/",
        "https://sudact.ru/regular/doc/Ab_1-2/?x=1",
        "https://sudact.ru/vsrf/doc/Ab_1-2/",
        "http://sudact.ru/regular/doc/Ab_1-2/",
        "https://example.test/regular/doc/Ab_1-2/",
    ):
        with pytest.raises(ValueError):
            mod.allowed_document_url(url)


def test_select_excludes_unknown_remanded_and_incomplete_identity(tmp_path, monkeypatch):
    status = [
        {"pair_id": "allowed", "verdict_group_id": "group", "source": "voronezh", "pair_status": "missing_sentence", "outcome": "changed", "appeal_artifact": {"document_id": "appeal"}},
        {"pair_id": "wrong-outcome", "verdict_group_id": "group", "source": "voronezh", "pair_status": "missing_sentence", "outcome": "remanded", "appeal_artifact": {"document_id": "appeal"}},
        {"pair_id": "new-verdict", "verdict_group_id": "new-group", "source": "voronezh", "pair_status": "missing_sentence", "outcome": "cancelled", "appeal_artifact": {"document_id": "new-appeal"}},
        {"pair_id": "incomplete", "verdict_group_id": "incomplete-group", "source": "voronezh", "pair_status": "missing_sentence", "outcome": "unchanged", "appeal_artifact": {"document_id": "incomplete-appeal"}},
    ]
    identity = {"first_court": "court", "first_case_number": "1-1/2024", "first_decision_date": "2024-01-01", "document_type": "appeal_against_verdict", "provenance": "test"}
    index = [
        {"underlying_case_id": "group", "identity": identity},
        {"underlying_case_id": "new-group", "identity": dict(identity, document_type="appellate_verdict")},
        {"underlying_case_id": "incomplete-group", "identity": dict(identity, first_court=None)},
    ]
    integrations = {row["appeal_artifact"]["document_id"]: {"document_id": row["appeal_artifact"]["document_id"], "metadata": {"decision_date": "01.02.2024"}} for row in status}
    original = mod.read_jsonl

    def fake_read(path):
        if path == mod.STATUS_PATH:
            return status
        if path == mod.INDEX_PATH:
            return index
        if path == tmp_path / "integration":
            return list(integrations.values())
        return original(path)

    monkeypatch.setattr(mod, "read_jsonl", fake_read)
    monkeypatch.setattr(mod, "INTEGRATION_PATHS", {"voronezh": tmp_path / "integration"})
    result = mod.select_candidates(tmp_path)
    assert result["eligible_status_outcome"] == 3
    assert result["actionable"] == 1
    assert result["excluded_new_sentence"] == 1
    assert result["insufficient_identity"] == 1
    rows = mod.read_jsonl(tmp_path / "candidates.jsonl")
    assert rows[0]["appeal_decision_date"] == "2024-02-01"


def test_finalize_recomputes_manifest_and_report_has_aggregates_only(tmp_path):
    (tmp_path / "candidates.jsonl").write_text("", encoding="utf-8")
    mod.write_json(tmp_path / "progress.json", {"pair": {"status": "rejected_document"}})
    result = mod.finalize(tmp_path)
    assert result == {"confirmed_pairs": 0, "manifest_entries": 4}
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Новых строго подтверждённых полных пар: **0**" in report
    assert "77 + len(pairs)" not in report
    assert "http" not in report
    assert "pair_" not in report
    manifest = mod.read_jsonl(tmp_path / "sha256-manifest.jsonl")
    for row in manifest:
        assert mod.sha256(tmp_path / row["path"]) == row["sha256"]
