import hashlib
import json
import re
from pathlib import Path

from tools import prepare_regional_pairs as mod

FILLER = "Полностью синтетический текст судебного акта без реальных лиц и обстоятельств. " * 40
UID = "24RS0001-01-2024-000001-01"


def sentence_text(date="01.02.2024"):
    return (
        f"П Р И Г О В О Р\n{date}\nТестовый районный суд по делу №1-1/2024 "
        f"с участием сторон установил обстоятельства и приговорил подсудимого. {FILLER}"
    )


def appeal_text(number="22-1/2024", date="01.03.2024", sentence_date="01.02.2024", disposition="оставить без изменения"):
    return (
        f"АПЕЛЛЯЦИОННОЕ ОПРЕДЕЛЕНИЕ\n{date}\nВоронежский областной суд по делу №{number} "
        f"на приговор Тестового районного суда от {sentence_date} по делу №1-1/2024. "
        f"ОПРЕДЕЛИЛ: приговор {disposition}, апелляционную жалобу оставить без удовлетворения. {FILLER}"
    )


def make_case(document_id="case-1", primary_sha=None, uid=None, with_number=True):
    return {
        "underlying_case_id": document_id,
        "case_keys": ["first_case:тестовый районный суд|1-1/2024"] if with_number else ([f"uid:{uid}"] if uid else []),
        "identity": {
            "source": "voronezh",
            "text_sha256": primary_sha,
            "first_court": "Тестовый районный суд" if with_number else None,
            "first_case_number": "1-1/2024" if with_number else None,
            "first_decision_date": "2024-02-01",
            "judicial_uids": [uid] if uid else [],
        },
    }


def make_document(
    root: Path,
    document_id: str,
    document_type: str,
    text: str,
    case_number="22-1/2024",
    decision_date="01.03.2024",
    uid=None,
):
    directory = root / "documents"
    directory.mkdir(exist_ok=True)
    text_path = directory / f"{document_id}.txt"
    raw_path = directory / f"{document_id}.html"
    text_path.write_text(text, encoding="utf-8")
    raw_path.write_bytes(b"synthetic raw artifact")
    return {
        "source": "voronezh",
        "document_id": document_id,
        "document_type": document_type,
        "text_path": str(text_path),
        "raw_path": str(raw_path),
        "text_sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "normalized_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "metadata": {
            "document_id": document_id,
            "source_url": "https://oblsud.vrn.sudrf.ru/modules.php?name=sud_delo&name_op=doc&number=synthetic",
            "case_number": case_number,
            "decision_date": decision_date,
            "court": "Воронежский областной суд",
            "lower_court": "Тестовый районный суд",
            "lower_case_number": "1-1/2024",
            "judicial_uid": uid,
        },
    }


def test_two_different_appeal_events_same_case_are_not_deduplicated(tmp_path: Path) -> None:
    sentence = make_document(tmp_path, "sentence-1", "trial_verdict", sentence_text(), "1-1/2024", "01.02.2024")
    first = make_document(tmp_path, "appeal-1", "appeal_against_verdict", appeal_text())
    second = make_document(
        tmp_path,
        "appeal-2",
        "appeal_against_verdict",
        appeal_text("22-2/2024", "02.03.2024", disposition="изменить"),
        "22-2/2024",
        "02.03.2024",
    )
    case = make_case(primary_sha=sentence["text_sha256"])

    rows, summary = mod.prepare_pairs([case], [sentence, first, second], root=tmp_path)

    confirmed = [row for row in rows if row["pair_status"] == "confirmed_pair"]
    assert len(confirmed) == 2
    assert len({row["pair_id"] for row in confirmed}) == 2
    assert {row["verdict_group_id"] for row in confirmed} == {"case-1"}
    assert {row["outcome"] for row in confirmed} == {"unchanged", "changed"}
    assert summary["confirmed_pairs"] == 2


def test_missing_sides_and_procedural_act_have_distinct_statuses(tmp_path: Path) -> None:
    appeal = make_document(tmp_path, "appeal-only", "appeal_against_verdict", appeal_text())
    sentence = make_document(tmp_path, "sentence-only", "trial_verdict", sentence_text(), "1-1/2024", "01.02.2024")
    procedural = make_document(tmp_path, "procedural", "procedural_appeal", appeal_text())
    appeal_case = make_case("appeal-case", appeal["text_sha256"])
    sentence_case = make_case("sentence-case", sentence["text_sha256"])

    appeal_rows, appeal_summary = mod.prepare_pairs([appeal_case], [appeal, procedural], root=tmp_path)
    sentence_rows, sentence_summary = mod.prepare_pairs([sentence_case], [sentence], root=tmp_path)

    statuses = {(row["verdict_group_id"], row["pair_status"]) for row in appeal_rows + sentence_rows}
    assert ("appeal-case", "missing_sentence") in statuses
    assert ("sentence-case", "missing_appeal") in statuses
    assert any(row["pair_status"] == "excluded_procedural" for row in appeal_rows)
    assert appeal_summary["confirmed_pairs"] == 0
    assert sentence_summary["confirmed_pairs"] == 0


def test_sha_mismatch_prevents_confirmation(tmp_path: Path) -> None:
    sentence = make_document(tmp_path, "sentence", "trial_verdict", sentence_text(), "1-1/2024", "01.02.2024")
    appeal = make_document(tmp_path, "appeal", "appeal_against_verdict", appeal_text())
    case = make_case(primary_sha=sentence["text_sha256"])
    Path(appeal["text_path"]).write_text("changed after hashing", encoding="utf-8")

    rows, _ = mod.prepare_pairs([case], [sentence, appeal], root=tmp_path)

    assert len(rows) == 1
    assert rows[0]["pair_status"] == "artifact_invalid"
    assert "text_sha256_mismatch" in rows[0]["identity_errors"]


def test_single_valid_sentence_is_selected_from_alternatives(tmp_path: Path) -> None:
    invalid = make_document(
        tmp_path,
        "sentence-invalid",
        "trial_verdict",
        sentence_text("02.02.2024"),
        "1-1/2024",
        "02.02.2024",
    )
    valid = make_document(tmp_path, "sentence-valid", "trial_verdict", sentence_text(), "1-1/2024", "01.02.2024")
    appeal = make_document(tmp_path, "appeal", "appeal_against_verdict", appeal_text())
    case = make_case(primary_sha=invalid["text_sha256"])

    rows, _ = mod.prepare_pairs([case], [invalid, valid, appeal], root=tmp_path)

    assert any(row["pair_status"] == "confirmed_pair" for row in rows)
    assert any(row["pair_status"] == "identity_failed" for row in rows)


def test_uid_identity_does_not_invent_missing_first_number(tmp_path: Path) -> None:
    sentence = make_document(
        tmp_path, "sentence", "trial_verdict", sentence_text(), "1-1/2024", "01.02.2024", UID
    )
    appeal = make_document(tmp_path, "appeal", "appeal_against_verdict", appeal_text(), uid=UID)
    sentence["metadata"]["lower_court"] = None
    sentence["metadata"]["lower_case_number"] = None
    appeal["metadata"]["lower_court"] = None
    appeal["metadata"]["lower_case_number"] = None
    case = make_case(primary_sha=sentence["text_sha256"], uid=UID, with_number=False)

    rows, _ = mod.prepare_pairs([case], [sentence, appeal], root=tmp_path)

    assert len(rows) == 1
    assert rows[0]["pair_status"] == "confirmed_pair", rows[0]["identity_errors"]


def test_procedural_and_new_trial_texts_are_rejected(tmp_path: Path) -> None:
    sentence = make_document(tmp_path, "sentence", "trial_verdict", sentence_text(), "1-1/2024", "01.02.2024")
    procedural = make_document(
        tmp_path,
        "procedural-appeal",
        "appeal_against_verdict",
        appeal_text().replace("на приговор", "на постановление о восстановлении срока обжалования приговора"),
    )
    case = make_case(primary_sha=sentence["text_sha256"])
    errors = mod.verify_regional_pair(case, sentence, procedural, sentence_text(), procedural_text(procedural))
    assert "appeal_type_not_confirmed" in errors

    new_trial = appeal_text().replace("АПЕЛЛЯЦИОННОЕ ОПРЕДЕЛЕНИЕ", "ПРИГОВОР НОВОГО РАССМОТРЕНИЯ")
    errors = mod.verify_regional_pair(case, sentence, procedural, sentence_text(), new_trial)
    assert "new_trial_not_accepted" in errors


def procedural_text(document):
    return Path(document["text_path"]).read_text(encoding="utf-8")


def test_direct_url_allows_only_official_document_links(tmp_path: Path) -> None:
    document = make_document(tmp_path, "url", "trial_verdict", sentence_text())
    assert mod.direct_document_url("voronezh", document).endswith("number=synthetic")
    document["metadata"]["source_url"] = "https://oblsud.vrn.sudrf.ru/search?q=synthetic"
    assert mod.direct_document_url("voronezh", document) is None
    document["metadata"]["source_url"] = "https://example.invalid/modules.php?name=sud_delo&name_op=doc&number=1"
    assert mod.direct_document_url("voronezh", document) is None


def test_outcome_comes_from_operative_part() -> None:
    assert mod.extract_outcome(appeal_text()) == "unchanged"
    assert mod.extract_outcome(appeal_text(disposition="изменить")) == "changed"
    assert mod.extract_outcome(appeal_text(disposition="отменить")) == "cancelled"
    procedural = appeal_text().replace("на приговор", "на постановление о восстановлении срока обжалования приговора")
    assert mod.extract_outcome(procedural) == "unchanged"
    without_operative = procedural.replace("ОПРЕДЕЛИЛ: приговор оставить без изменения", "ОПРЕДЕЛИЛ: постановление отменить")
    assert mod.extract_outcome(without_operative) == "unknown"


def test_report_contains_only_aggregates(tmp_path: Path) -> None:
    sentence = make_document(tmp_path, "sentence", "trial_verdict", sentence_text(), "1-1/2024", "01.02.2024")
    appeal = make_document(tmp_path, "appeal", "appeal_against_verdict", appeal_text())
    case = make_case(primary_sha=sentence["text_sha256"])
    rows, summary = mod.prepare_pairs([case], [sentence, appeal], root=tmp_path)

    report = mod.render_report(summary)

    assert "confirmed" not in report or "Подтвержд" in report
    assert "1-1/2024" not in report
    assert "22-1/2024" not in report
    assert "https://" not in report
    assert UID not in report
    assert sentence["text_sha256"] not in report
    assert not re.search(r"[0-9a-f]{64}", report)
    json.dumps(rows, ensure_ascii=False)


def test_ambiguous_document_link_is_not_confirmed(tmp_path: Path) -> None:
    sentence = make_document(tmp_path, "sentence", "trial_verdict", sentence_text(), "1-1/2024", "01.02.2024")
    appeal = make_document(tmp_path, "appeal", "appeal_against_verdict", appeal_text())
    first = make_case("case-1", sentence["text_sha256"])
    second = make_case("case-2", appeal["text_sha256"])

    rows, _ = mod.prepare_pairs([first, second], [sentence, appeal], root=tmp_path)

    assert not any(row["pair_status"] == "confirmed_pair" for row in rows)
    assert any("ambiguous_case_link" in row["identity_errors"] for row in rows)


def test_related_collection_relative_artifact_paths_resume_correctly(tmp_path: Path) -> None:
    document = make_document(tmp_path, "relative", "trial_verdict", sentence_text())
    document["text_path"] = Path(document["text_path"]).name
    document["raw_path"] = Path(document["raw_path"]).name
    document["_artifact_base"] = str(tmp_path / "documents")

    (artifact, text), errors = mod.artifact_for_document(document, root=tmp_path)

    assert errors == []
    assert text == sentence_text()
    assert artifact["path"].endswith("relative.txt")
    assert artifact["raw_path"].endswith("relative.html")
