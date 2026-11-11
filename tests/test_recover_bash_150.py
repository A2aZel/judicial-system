import json

import pytest

from tools import recover_bash_150 as mod


def candidate():
    return {
        "appeal_event_id": "ae_test",
        "source": "bashkortostan",
        "outcome": "changed",
        "first_court": "октябрьский районный суд г уфы",
        "first_case_number": "1-12/2022",
        "first_decision_date": "2022-03-21",
    }


def verdict():
    return """ПРИГОВОР
Именем Российской Федерации
г. Уфа 21 марта 2022 года
Октябрьский районный суд г. Уфы Республики Башкортостан, рассмотрев уголовное дело № 1-12/2022,
УСТАНОВИЛ
""" + "Доказательства рассмотрены судом. " * 80 + "\nПРИГОВОРИЛ\nПризнать виновным."


def test_sentence_header_and_full_body_are_required():
    assert mod.validate_sentence(candidate(), verdict(), "Приговор") == []


def test_date_must_precede_court_and_number_must_match():
    text = verdict().replace("г. Уфа 21 марта 2022 года\n", "г. Уфа\n")
    assert "decision_date_mismatch" in mod.validate_sentence(candidate(), text, "Приговор")
    assert "case_number_mismatch" in mod.validate_sentence(candidate(), verdict().replace("1-12/2022", "1-13/2022"), "Приговор")


def test_new_trial_and_quote_are_rejected():
    new_trial = verdict().replace("ПРИГОВОР", "АПЕЛЛЯЦИОННЫЙ ПРИГОВОР", 1)
    assert "new_trial_verdict" in mod.validate_sentence(candidate(), new_trial, "Апелляционный приговор")
    quote = verdict().replace("ПРИГОВОРИЛ\nПризнать виновным.", "Суд указал на необходимость проверки.")
    assert "operative_verdict_missing" in mod.validate_sentence(candidate(), quote, "Приговор")


def test_only_direct_public_document_page_is_allowed():
    assert mod.check_document_url("https://sudact.ru/regular/doc/Abcd1234/") == "sudact.ru"
    for url in (
        "https://sudact.ru/regular/doc_ajax/Abcd1234/",
        "https://sudact.ru/regular/doc/Abcd1234/?page=2",
        "https://vsrf.ru/api/Abcd1234/",
        "https://vs.bkr.sudrf.ru/modules.php?name=sud_delo",
    ):
        with pytest.raises(mod.AccessStopped):
            mod.check_document_url(url)


def test_record_search_is_resumable(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    mod.write_jsonl(tmp_path / "candidates.jsonl", [candidate()])
    mod.save_progress(tmp_path, {"ae_test": {"status": "pending", "attempts": []}})
    mod.record_search("ae_test", "no_result", "exact query", [], [], tmp_path)
    progress = mod.load_progress(tmp_path)
    evidence = mod.ROOT / progress["ae_test"]["search_evidence"]
    assert progress["ae_test"]["status"] == "no_result"
    assert progress["ae_test"]["search_evidence_sha256"] == mod.sha256(evidence)
    assert json.loads(evidence.read_text(encoding="utf-8"))["query"] == "exact query"


def test_selection_excludes_other_statuses_outcomes_and_incomplete_identity(tmp_path, monkeypatch):
    statuses = []
    indexed = []
    specs = [
        ("changed", "missing_sentence", "2022-03-21"),
        ("remanded", "missing_sentence", "2022-03-21"),
        ("cancelled", "missing_sentence", None),
        ("unknown", "missing_sentence", "2022-03-21"),
        ("changed", "rejected_new_trial", "2022-03-21"),
        ("changed", "missing_appeal", "2022-03-21"),
    ]
    for position, (outcome, pair_status, date) in enumerate(specs):
        event = f"ae_{position}"
        case = f"case_{position}"
        statuses.append({
            "appeal_event_id": event,
            "underlying_case_id": case,
            "source": "bashkortostan",
            "outcome": outcome,
            "pair_status": pair_status,
        })
        indexed.append({"underlying_case_id": case, "identity": {
            "first_court": "октябрьский районный суд г уфы",
            "first_case_number": "1-12/2022",
            "first_decision_date": date,
        }})
    status_path = tmp_path / "status.jsonl"
    index_path = tmp_path / "index.jsonl"
    mod.write_jsonl(status_path, statuses)
    mod.write_jsonl(index_path, indexed)
    monkeypatch.setattr(mod, "STATUS", status_path)
    monkeypatch.setattr(mod, "INDEX", index_path)
    selected = mod.select_candidates(tmp_path)
    assert [row["appeal_event_id"] for row in selected] == ["ae_0"]


def test_public_report_contains_only_aggregates(tmp_path):
    mod.write_jsonl(tmp_path / "candidates.jsonl", [candidate()])
    mod.save_progress(tmp_path, {"ae_test": {"status": "no_result"}})
    summary = mod.write_report(tmp_path)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert summary["confirmed_pairs"] == 0
    assert "1-12/2022" not in report
    assert "http" not in report
    assert "ae_test" not in report


def test_new_trial_appeal_is_saved_separately_and_never_searched(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    text = "ОПРЕДЕЛИЛ: приговор отменить, дело направить на новое судебное разбирательство."
    text_path = tmp_path / "appeal.txt"
    raw_path = tmp_path / "appeal.html"
    text_path.write_text(text, encoding="utf-8")
    raw_path.write_text(text, encoding="utf-8")
    row = {
        "appeal_event_id": "ae_test",
        "appeal": {
            "path": "appeal.txt",
            "sha256": mod.sha256(text_path),
            "raw_path": "appeal.html",
            "raw_sha256": mod.sha256(raw_path),
        },
    }
    status_path = tmp_path / "status.jsonl"
    mod.write_jsonl(status_path, [row])
    monkeypatch.setattr(mod, "STATUS", status_path)
    mod.write_jsonl(tmp_path / "candidates.jsonl", [candidate()])
    mod.save_progress(tmp_path, {"ae_test": {"status": "pending", "attempts": []}})
    assert "new_trial" in mod.validate_source_appeal(row)
    result = mod.check_appeal_batch(tmp_path, size=1, batch=2)
    assert result["rejected_new_trial"] == 1
    rejected = mod.read_jsonl(tmp_path / "rejected.jsonl")
    assert rejected[0]["status"] == "rejected_new_trial"
    assert rejected[0]["validation_errors"] == ["new_trial"]


def test_downloaded_duplicate_reuses_raw_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    mod.write_jsonl(tmp_path / "candidates.jsonl", [candidate()])
    mod.save_progress(tmp_path, {"ae_test": {"status": "candidate_found", "attempts": []}})
    mod.write_json(tmp_path / "access.json", {"sudact.ru": {"robots_allowed": True}})
    mod.write_jsonl(tmp_path / "status.jsonl", [{"appeal_event_id": "ae_test"}])
    monkeypatch.setattr(mod, "STATUS", tmp_path / "status.jsonl")
    monkeypatch.setattr(mod, "validate_source_appeal", lambda row: [])
    monkeypatch.setattr(mod, "parse_document", lambda page, host: {"title": "Приговор", "court": "", "text": "text"})
    monkeypatch.setattr(mod, "validate_sentence", lambda item, text, title: [])
    duplicate = tmp_path / "raw" / "existing.html"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(b"same document")
    monkeypatch.setattr(mod, "throttled_get", lambda url, output, pause: (b"same document", "text/html"))
    result = mod.fetch_sentence("ae_test", "https://sudact.ru/regular/doc/Abcd1234/", tmp_path)
    assert result["deduplicated"] is True
    progress = mod.load_progress(tmp_path)
    assert progress["ae_test"]["raw_path"] == "raw/existing.html"
    assert len(list((tmp_path / "raw").glob("*.html"))) == 1


def test_finalize_never_confirms_matching_sentence_with_new_trial_appeal(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    appeal_text = "ОПРЕДЕЛИЛ: приговор отменить, дело направить на новое судебное рассмотрение."
    appeal_text_path = tmp_path / "appeal.txt"
    appeal_raw_path = tmp_path / "appeal.html"
    appeal_text_path.write_text(appeal_text, encoding="utf-8")
    appeal_raw_path.write_text(appeal_text, encoding="utf-8")
    row = {
        "appeal_event_id": "ae_test",
        "pair_status": "missing_sentence",
        "outcome": "cancelled",
        "appeal": {
            "path": "appeal.txt",
            "sha256": mod.sha256(appeal_text_path),
            "raw_path": "appeal.html",
            "raw_sha256": mod.sha256(appeal_raw_path),
        },
    }
    status_path = tmp_path / "status.jsonl"
    mod.write_jsonl(status_path, [row])
    monkeypatch.setattr(mod, "STATUS", status_path)
    sentence = verdict()
    raw_path = tmp_path / "sentence.html"
    text_path = tmp_path / "sentence.txt"
    raw_path.write_text('<h1>Приговор</h1><hr class="hr-h1">' + sentence + '<h3>Суд:</h3>', encoding="utf-8")
    text_path.write_text(sentence, encoding="utf-8")
    mod.write_jsonl(tmp_path / "candidates.jsonl", [candidate()])
    mod.save_progress(tmp_path, {"ae_test": {
        "status": "validated_sentence",
        "document_url": "https://sudact.ru/regular/doc/Abcd1234/",
        "raw_path": "sentence.html",
        "raw_sha256": mod.sha256(raw_path),
        "text_path": "sentence.txt",
        "text_sha256": mod.sha256(text_path),
        "document_evidence": "evidence.json",
        "document_evidence_sha256": "proof",
    }})
    result = mod.finalize(tmp_path)
    assert result["confirmed_pairs"] == 0
    assert mod.load_progress(tmp_path)["ae_test"]["status"] == "rejected_new_trial"
    assert mod.read_jsonl(tmp_path / "recovered-pairs.jsonl") == []
    assert mod.read_jsonl(tmp_path / "rejected.jsonl")[0]["status"] == "rejected_new_trial"


def test_windows_paths_are_normalized_and_legacy_values_resolve(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    assert mod.project_path(tmp_path / "raw" / "document.html") == "raw/document.html"
    assert mod.resolve_project_path("raw\\document.html") == tmp_path / "raw" / "document.html"


def test_repeated_batch_and_finalize_do_not_duplicate_records(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    appeal_text = "ПОСТАНОВИЛ: приговор оставить без изменения."
    appeal_text_path = tmp_path / "appeal.txt"
    appeal_raw_path = tmp_path / "appeal.html"
    appeal_text_path.write_text(appeal_text, encoding="utf-8")
    appeal_raw_path.write_text(appeal_text, encoding="utf-8")
    row = {
        "appeal_event_id": "ae_test",
        "pair_status": "missing_sentence",
        "outcome": "changed",
        "appeal": {
            "path": "appeal.txt",
            "sha256": mod.sha256(appeal_text_path),
            "raw_path": "appeal.html",
            "raw_sha256": mod.sha256(appeal_raw_path),
        },
    }
    status_path = tmp_path / "status.jsonl"
    mod.write_jsonl(status_path, [row])
    monkeypatch.setattr(mod, "STATUS", status_path)
    sentence = verdict()
    raw_path = tmp_path / "sentence.html"
    text_path = tmp_path / "sentence.txt"
    raw_path.write_text('<h1>Приговор</h1><hr class="hr-h1">' + sentence + '<h3>Суд:</h3>', encoding="utf-8")
    text_path.write_text(sentence, encoding="utf-8")
    mod.write_jsonl(tmp_path / "candidates.jsonl", [candidate()])
    mod.save_progress(tmp_path, {"ae_test": {
        "status": "validated_sentence",
        "document_url": "https://sudact.ru/regular/doc/Abcd1234/",
        "raw_path": "sentence.html",
        "raw_sha256": mod.sha256(raw_path),
        "text_path": "sentence.txt",
        "text_sha256": mod.sha256(text_path),
        "document_evidence": "evidence.json",
        "document_evidence_sha256": "proof",
    }})
    assert mod.finalize(tmp_path)["confirmed_pairs"] == 1
    assert mod.finalize(tmp_path)["confirmed_pairs"] == 1
    assert len(mod.read_jsonl(tmp_path / "recovered-pairs.jsonl")) == 1
    progress = mod.load_progress(tmp_path)
    mod.sync_rejected(tmp_path, progress)
    mod.sync_rejected(tmp_path, progress)
    assert mod.read_jsonl(tmp_path / "rejected.jsonl") == []


def test_spaced_established_marker_is_accepted():
    text = verdict().replace("УСТАНОВИЛ", "У С Т А Н О В И Л")
    assert mod.validate_sentence(candidate(), text, "Приговор") == []
