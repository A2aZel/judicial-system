import hashlib
import json

import pytest

from tools import prepare_rostov_7kas_pairs as mod


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sentence_text(number="1-1/2024"):
    return (
        "П Р И Г О В О Р\nименем Российской Федерации\n"
        "г. Тестовск 1 февраля 2024 года\n"
        f"Тестовый районный суд, рассмотрев уголовное дело № {number}, установил.\n"
        + "Доказательства исследованы в судебном заседании. " * 30
        + "\nП Р И Г О В О Р И Л: признать виновным и назначить наказание."
    )


def appeal_text(date="1 марта 2024 года", disposition="оставить без изменения", same_uid_marker=""):
    return (
        "АПЕЛЛЯЦИОННОЕ ОПРЕДЕЛЕНИЕ\n"
        f"г. Тестовск {date}\nАпелляционный суд {same_uid_marker}\n"
        "рассмотрел апелляционную жалобу на приговор Тестового районного суда от 1 февраля 2024 года.\n"
        + "Судебная коллегия проверила материалы дела и доводы сторон. " * 30
        + f"\nО П Р Е Д Е Л И Л: приговор Тестового районного суда от 1 февраля 2024 года {disposition}."
    )


def make_document(
    root,
    record_id,
    document_type,
    text,
    *,
    source="rostov",
    number="1-1/2024",
    first_keys=None,
    uids=None,
    decision_date=None,
    case_number=None,
    court=None,
):
    text_data = text.encode("utf-8")
    raw_data = ("official raw " + record_id).encode()
    evidence_data = ("official identity evidence " + record_id).encode()
    text_path = root / "texts" / f"{record_id}.txt"
    raw_path = root / "raw" / f"{record_id}.html"
    evidence_path = root / "evidence" / f"{record_id}.html"
    for path, data in ((text_path, text_data), (raw_path, raw_data), (evidence_path, evidence_data)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    text_sha = digest(text_data)
    raw_sha = digest(raw_data)
    evidence_sha = digest(evidence_data)
    first_key = f"first_case:тестовый районный суд|{number}"
    if document_type == "trial_verdict":
        decision_date = decision_date or "01.02.2024"
        case_number = case_number or number
        court = court or "Тестовый районный суд"
    else:
        decision_date = decision_date or "01.03.2024"
        case_number = case_number or "22-1/2024"
        court = court or "Апелляционный суд"
    row = {
        "underlying_case_id": "underlying-" + record_id,
        "identity": {
            "record_id": record_id,
            "source": source,
            "document_type": document_type,
            "text_path": str(text_path.relative_to(root)),
            "text_sha256": text_sha,
            "normalized_text_sha256": digest(mod.compact_text(text).encode()),
            "raw_path": str(raw_path.relative_to(root)),
            "raw_sha256": raw_sha,
            "identity_evidence_path": str(evidence_path.relative_to(root)),
            "identity_link_evidence_sha256": evidence_sha,
            "identity_status": "proven",
            "identity_link_verified": True,
            "identity_errors": [],
            "quality_errors": [],
            "full_relevant_merits_text": True,
            "case_keys": first_keys or [first_key],
            "judicial_uids": uids or [],
            "first_court": "Тестовый районный суд",
            "first_case_number": number,
            "first_decision_date": "2024-02-01",
            "document_urls": ["https://official.invalid/document"],
        },
    }
    metadata = {
        "text_sha256": text_sha,
        "decision_date": decision_date,
        "case_number": case_number,
        "court": court,
    }
    return row, metadata


def run_pipeline(tmp_path, documents, *, resume=False):
    rows = [document[0] for document in documents]
    metadata = [document[1] for document in documents]
    index = tmp_path / "index.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"
    output = tmp_path / "output"
    mod.write_jsonl(index, rows)
    mod.write_jsonl(metadata_path, metadata)
    summary = mod.prepare_pairs(tmp_path, index, metadata_path, output, resume=resume)
    statuses = mod.read_jsonl(output / "pair-status.jsonl")
    return summary, statuses, output


def test_confirmed_pair_requires_and_preserves_both_artifacts(tmp_path):
    documents = [
        make_document(tmp_path, "sentence-record", "trial_verdict", sentence_text()),
        make_document(tmp_path, "appeal-record", "appeal_against_verdict", appeal_text()),
    ]
    summary, statuses, output = run_pipeline(tmp_path, documents)
    assert summary["confirmed_pairs"] == 1
    assert len(statuses) == 1
    status = statuses[0]
    assert status["pair_status"] == "confirmed_pair"
    assert status["identity_errors"] == []
    assert status["outcome"] == "unchanged"
    assert status["sentence_artifact"]["text_sha256"] == documents[0][0]["identity"]["text_sha256"]
    assert status["appeal_artifact"]["raw_sha256"] == documents[1][0]["identity"]["raw_sha256"]
    assert status["verdict_group_id"]
    report = (output / "report.md").read_text(encoding="utf-8")
    for forbidden in ("sentence-record", "appeal-record", "1-1/2024", "official.invalid", documents[0][0]["identity"]["text_sha256"]):
        assert forbidden not in report
    assert "confirmed pairs: 1" in report


def test_missing_acts_and_new_trial_verdict_are_not_pairs(tmp_path):
    documents = [
        make_document(tmp_path, "later-only", "cassation_against_verdict", appeal_text().replace("АПЕЛЛЯЦИОННОЕ", "КАССАЦИОННОЕ").replace("апелляционную", "кассационную"), number="1-2/2024"),
        make_document(tmp_path, "sentence-only", "trial_verdict", sentence_text()),
        make_document(tmp_path, "new-trial", "appellate_verdict", sentence_text(), number="1-3/2024"),
    ]
    summary, statuses, _ = run_pipeline(tmp_path, documents)
    counts = {status["source_record_id"]: status["pair_status"] for status in statuses}
    assert counts == {
        "later-only": "missing_sentence",
        "new-trial": "rejected_new_trial_verdict",
        "sentence-only": "missing_appeal",
    }
    assert summary["confirmed_pairs"] == 0
    assert summary["missing_sentence"] == 1
    assert summary["missing_appeal"] == 1
    assert summary["rejected_new_trial_verdict"] == 1


def test_sha_and_disjoint_uid_fail_identity(tmp_path):
    sentence = make_document(tmp_path, "sentence-record", "trial_verdict", sentence_text(), uids=["SYNTHETIC-UID-1"])
    appeal = make_document(tmp_path, "appeal-record", "appeal_against_verdict", appeal_text(), uids=["SYNTHETIC-UID-2"])
    summary, statuses, _ = run_pipeline(tmp_path, [sentence, appeal])
    assert summary["confirmed_pairs"] == 0
    assert statuses[0]["pair_status"] == "identity_failed"
    assert "judicial_uid_mismatch" in statuses[0]["identity_errors"]
    text_path = tmp_path / sentence[0]["identity"]["text_path"]
    text_path.write_text(text_path.read_text(encoding="utf-8") + "changed", encoding="utf-8")
    summary, statuses, _ = run_pipeline(tmp_path, [sentence, appeal])
    assert "text_sha256_mismatch" in statuses[0]["identity_errors"]


def test_only_official_case_key_aliases_link_documents(tmp_path):
    sentence = make_document(tmp_path, "sentence-record", "trial_verdict", sentence_text())
    appeal = make_document(
        tmp_path,
        "appeal-record",
        "appeal_against_verdict",
        appeal_text(),
        number="1-2/2024",
        first_keys=["first_case:тестовый районный суд|1-2/2024"],
    )
    sentence[0]["identity"]["case_keys"].append("first_case:тестовый районный суд|1-2/2024")
    summary, statuses, _ = run_pipeline(tmp_path, [sentence, appeal])
    assert summary["confirmed_pairs"] == 1
    appeal[0]["identity"]["case_keys"] = []
    appeal[0]["identity"]["weak_case_keys"] = ["first_number:тестовый районный суд|1-2/2024"]
    summary, _, _ = run_pipeline(tmp_path, [sentence, appeal])
    assert summary["confirmed_pairs"] == 0


def test_same_uid_different_appeal_events_are_not_deduplicated(tmp_path):
    documents = [
        make_document(tmp_path, "sentence-record", "trial_verdict", sentence_text(), uids=["SHARED-UID"]),
        make_document(tmp_path, "appeal-one", "appeal_against_verdict", appeal_text(), uids=["SHARED-UID"]),
        make_document(
            tmp_path,
            "appeal-two",
            "appeal_against_verdict",
            appeal_text(date="15 марта 2024 года", disposition="изменить", same_uid_marker="второе событие"),
            uids=["SHARED-UID"],
            decision_date="15.03.2024",
        ),
    ]
    summary, statuses, _ = run_pipeline(tmp_path, documents)
    assert summary["confirmed_pairs"] == 2
    assert summary["duplicates"] == 0
    assert {status["outcome"] for status in statuses} == {"unchanged", "changed"}
    assert len({status["pair_id"] for status in statuses}) == 2


def test_exact_duplicate_event_is_reported_once(tmp_path):
    original = make_document(tmp_path, "appeal-one", "appeal_against_verdict", appeal_text())
    duplicate = make_document(tmp_path, "appeal-two", "appeal_against_verdict", appeal_text())
    summary, statuses, _ = run_pipeline(tmp_path, [original, duplicate])
    assert summary["duplicates"] == 1
    assert summary["missing_sentence"] == 1
    assert [status["pair_status"] for status in statuses].count("duplicate") == 1


def test_procedural_act_is_rejected(tmp_path):
    procedural = make_document(tmp_path, "procedural", "procedural_appeal", appeal_text())
    summary, statuses, _ = run_pipeline(tmp_path, [procedural])
    assert summary["confirmed_pairs"] == 0
    assert statuses[0]["pair_status"] == "identity_failed"
    assert statuses[0]["identity_errors"] == ["unsupported_document_type"]


def test_cassation_of_new_appellate_verdict_is_rejected(tmp_path):
    text = (
        "КАССАЦИОННОЕ ОПРЕДЕЛЕНИЕ\n1 марта 2024 года\n"
        "рассмотрела кассационную жалобу на апелляционный приговор областного суда, "
        "которым приговор Тестового районного суда от 1 февраля 2024 года отменён с вынесением нового приговора.\n"
        + "Обстоятельства нового рассмотрения проверены. " * 40
        + "ОПРЕДЕЛИЛА: апелляционный приговор оставить без изменения."
    )
    document = make_document(tmp_path, "new-appellate-verdict", "cassation_against_verdict", text, source="7kas")
    summary, statuses, _ = run_pipeline(tmp_path, [document])
    assert statuses[0]["pair_status"] == "rejected_new_trial_verdict"
    assert statuses[0]["identity_errors"] == ["new_trial_verdict_not_accepted"]
    assert summary["by_source"]["7kas"]["outcomes_available_later_acts"] == {}


def test_court_match_does_not_erase_district_city_conflict():
    assert mod.court_matches("на приговор Сальского районного суда", "Сальский районный суд городской")
    assert not mod.court_matches("на приговор Сальского городского суда", "Сальский районный суд городской")


def test_resume_reuses_only_matching_confirmed_artifacts(tmp_path):
    documents = [
        make_document(tmp_path, "sentence-record", "trial_verdict", sentence_text()),
        make_document(tmp_path, "appeal-record", "appeal_against_verdict", appeal_text()),
    ]
    _, first, _ = run_pipeline(tmp_path, documents)
    _, second, _ = run_pipeline(tmp_path, documents, resume=True)
    assert first == second
    changed = dict(second[0])
    changed["identity_errors"] = ["text_sha256_mismatch"]
    assert not mod.previous_confirmed({second[0]["pair_id"]: second[0]}, changed)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "КАССАЦИОННОЕ ОПРЕДЕЛЕНИЕ. Просят отменить приговор. "
            + "Доводы стороны. " * 20
            + "ОПРЕДЕЛИЛА: приговор Тестового районного суда "
            + "с длинными реквизитами участников " * 40
            + "оставить без изменений.",
            "unchanged",
        ),
        (
            "КАССАЦИОННОЕ ОПРЕДЕЛЕНИЕ. ОПРЕДЕЛИЛА: кассационную жалобу на приговор "
            + "с полными реквизитами " * 20 + "оставить без удовлетворения.",
            "unchanged",
        ),
        (
            "КАССАЦИОННОЕ ОПРЕДЕЛЕНИЕ. ОПРЕДЕЛИЛА: апелляционное определение областного суда отменить, "
            "уголовное дело передать на новое судебное рассмотрение в тот же суд апелляционной инстанции.",
            "remanded",
        ),
        (
            "АПЕЛЛЯЦИОННОЕ ОПРЕДЕЛЕНИЕ. Просят отменить приговор. ОПРЕДЕЛИЛ: приговор отменить, дело направить на новое рассмотрение.",
            "cancelled",
        ),
        ("АПЕЛЛЯЦИОННОЕ ОПРЕДЕЛЕНИЕ. В жалобе просят приговор отменить.", "unknown"),
    ],
)
def test_outcome_uses_only_operative_part(text, expected):
    assert mod.extract_outcome(text) == expected


def test_summary_counts_available_outcomes_without_confirmed_pairs(tmp_path):
    later = make_document(
        tmp_path,
        "cassation-only",
        "cassation_on_merits_appeal",
        "КАССАЦИОННОЕ ОПРЕДЕЛЕНИЕ\n1 марта 2024 года\nприговор Тестового районного суда от 1 февраля 2024 года и апелляционное определение\n"
        + "Проверены доводы кассационной жалобы. " * 40
        + "ОПРЕДЕЛИЛА: апелляционное определение отменить, дело передать на новое апелляционное рассмотрение.",
        source="7kas",
    )
    summary, statuses, _ = run_pipeline(tmp_path, [later])
    assert statuses[0]["outcome"] == "remanded"
    assert "appeal_type_not_confirmed" not in statuses[0]["identity_errors"]
    assert summary["by_source"]["7kas"]["outcomes_available_later_acts"] == {"remanded": 1}
    assert summary["outcomes_confirmed_pairs"] == {}
    assert summary["network_document_requests"] == 0
    assert summary["processed_dataset_or_index_changed"] is False


def test_sudact_full_sentence_is_linked_with_text_only_provenance(tmp_path):
    appeal = make_document(tmp_path, "later-only", "cassation_against_verdict", appeal_text().replace("АПЕЛЛЯЦИОННОЕ", "КАССАЦИОННОЕ").replace("апелляционную", "кассационную"), source="7kas")
    text = sentence_text()
    docs_dir = tmp_path / "corpus/sudact/docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "synthetic.txt").write_text(text, encoding="utf-8")
    metadata = [{
        "id": "synthetic", "url": "https://sudact.ru/regular/doc/synthetic/",
        "title": "Приговор № 1-1/2024 от 1 февраля 2024 года",
        "court": "Тестовый районный суд - Уголовное", "period": "2024-H1",
        "chars": len(text), "fetched_at": "2024-03-01T00:00:00Z",
    }]
    mod.write_jsonl(tmp_path / "corpus/sudact/docs.jsonl", metadata)
    assert mod.fingerprint_sudact(tmp_path) == 1
    summary, statuses, _ = run_pipeline(tmp_path, [appeal])
    assert summary["confirmed_pairs"] == 1
    assert summary["text_only_provenance_pairs"] == 1
    assert statuses[0]["pair_status"] == "confirmed_pair"
    assert statuses[0]["sentence_artifact"]["source"] == "sudact"
    assert statuses[0]["sentence_artifact"]["raw_path"] is None
    assert statuses[0]["identity_errors"] == []

    changed = text.replace("Доказательства", "Д0казательства", 1)
    assert len(changed) == len(text)
    (docs_dir / "synthetic.txt").write_text(changed, encoding="utf-8")
    summary, _, _ = run_pipeline(tmp_path, [appeal])
    assert summary["confirmed_pairs"] == 0

    wrong_header = text.replace("1 февраля 2024 года", "2 февраля 2024 года", 1) + " Историческая справка: 1 февраля 2024 года."
    (docs_dir / "synthetic.txt").write_text(wrong_header, encoding="utf-8")
    metadata[0]["chars"] = len(wrong_header)
    mod.write_jsonl(tmp_path / "corpus/sudact/docs.jsonl", metadata)
    mod.fingerprint_sudact(tmp_path)
    summary, _, _ = run_pipeline(tmp_path, [appeal])
    assert summary["confirmed_pairs"] == 0


def test_regional_resume_rejects_stale_outcome_and_contract():
    artifacts = {"sentence_artifact": {"text_sha256": "s", "raw_sha256": None}, "appeal_artifact": {"text_sha256": "a", "raw_sha256": "r"}}
    old = {"pair_id": "p", "pair_status": "confirmed_pair", "processing_contract": mod.PAIR_CONTRACT, "outcome": "unchanged", "identity_errors": [], **artifacts}
    current = {**old, "outcome": "changed"}
    assert not mod.previous_confirmed({"p": old}, current)
    assert mod.previous_confirmed({"p": {**old, "outcome": "changed"}}, current)
    assert not mod.previous_confirmed({"p": {**old, "outcome": "changed", "processing_contract": mod.PAIR_CONTRACT - 1}}, current)
