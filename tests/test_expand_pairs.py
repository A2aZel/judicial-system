import json

from tools import expand_pairs as mod


def test_report_contains_only_aggregates(tmp_path):
    from tests.test_training_dataset import moscow_candidate, moscow_document, moscow_valid_texts
    sentence, appeal = moscow_valid_texts()
    mod.write_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", [moscow_candidate()])
    mod.atomic_json(tmp_path / "work" / "selection.json", {"outside_both_links": 1, "by_year": {"2026": 1}})
    documents = {"sentence": moscow_document(tmp_path, "sentence", sentence), "appeal": moscow_document(tmp_path, "appeal", appeal)}
    mod.atomic_json(tmp_path / "pair-progress.json", {"synthetic-moscow": {"status": "confirmed_pair", "outcome": "changed", "documents": documents, "errors": []}})
    mod.write_report(tmp_path, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Новых подтверждённых полных пар: 1" in text
    assert "synthetic" not in text
    assert "case_id" not in text
    assert "http" not in text


def test_error_checkpoint_does_not_count_as_pair(tmp_path):
    mod.atomic_json(tmp_path / "work" / "selection.json", {"outside_both_links": 0, "by_year": {}})
    mod.write_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", [])
    mod.atomic_json(tmp_path / "pair-progress.json", {"synthetic": {"status": "download_error", "errors": ["HTTP 404"]}})
    mod.write_report(tmp_path, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Новых подтверждённых полных пар: 0" in text
    assert "download_error | 0" in text


def candidate_with_two_sentences():
    return {
        "appeal_id": "synthetic-moscow",
        "identity": {
            "uid": "SYNTHETIC-UID", "number": "10-1/2024", "year": 2024,
            "courtName": "Московский городской суд", "act_date": "01.03.2024", "documentType": "Уголовное дело",
        },
        "aliases": [],
        "lower": [{"number": "1-1/2024", "courtName": "Тестовый районный суд", "courtCode": "FIRST"}],
        "participants": [],
        "first": {
            "_id": "synthetic-first", "uid": "SYNTHETIC-UID", "number": "1-1/2024", "year": 2024,
            "courtName": "Тестовый районный суд", "dateFinal": "01.02.2024", "lastHearingDate": "01.02.2024",
            "participants": [],
        },
        "years": ["2024"],
        "first_attachments": [{"link": "wrong", "date": "01.02.2024"}, {"link": "right", "date": "01.02.2024"}],
        "appeal_attachments": [{"link": "appeal", "date": "01.03.2024"}],
    }


def test_processing_continues_to_next_attachment(tmp_path, monkeypatch):
    from tests.test_training_dataset import moscow_valid_texts
    sentence, appeal = moscow_valid_texts()
    wrong = sentence.replace("1-1/2024", "1-11/2024")
    candidate = candidate_with_two_sentences()
    mod.write_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", [candidate])
    progress_path = tmp_path / "pair-progress.json"

    def fake_ensure(raw_dir, appeal_id, kind, attachment, current, progress_path, progress, pause):
        text = wrong if attachment["link"] == "wrong" else (sentence if kind == "sentence" else appeal)
        current["documents"][kind] = {"url": attachment["link"], "path": f"{kind}.bin", "sha256": "x", "text_path": f"{kind}.txt", "text_source_sha256": "x", "text_sha256": "x"}
        return current["documents"][kind], text

    monkeypatch.setattr(mod, "ensure_document", fake_ensure)
    result = mod.download_expansion(tmp_path, pause=0, limit=1)
    assert result["statuses"]["confirmed_pair"] == 1
    progress = mod.read_json(progress_path, {})
    assert len(progress["synthetic-moscow"]["attempts"]) == 2


def test_report_uses_only_current_candidates_and_remaining_never_negative(tmp_path):
    mod.atomic_json(tmp_path / "work" / "selection.json", {"outside_both_links": 0, "by_year": {}})
    mod.write_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", [])
    mod.atomic_json(tmp_path / "pair-progress.json", {"stale": {"status": "download_error", "errors": ["HTTP 404"]}})
    mod.write_report(tmp_path, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "remaining | 0" in text
    assert "remaining | -" not in text


def test_confirmed_aliases_are_preserved(tmp_path, monkeypatch):
    first = {"_id": "first", "uid": "U", "number": "1-1/2024", "year": 2024, "courtName": "Суд", "dateFinal": "01.02.2024", "participants": [], "attachments": [{"displayName": "Приговор", "link": "sentence"}]}
    appeal = {
        "_id": "appeal", "uid": "U", "number": "10-1/2024", "year": 2024, "courtName": "Аппеляция",
        "dateReg": "2024-02-01", "dateFinal": "2024-03-01", "documentType": "Уголовное дело",
        "lower": [{"number": "1-1/2024", "courtName": "Суд", "courtCode": "FIRST"}],
        "participants": [],
        "attachments": [{"displayName": "Постановление суда апелляционной инстанции", "link": "appeal"}],
        "upper": [{"number": "10-99/2024", "year": 2024, "dateFinal": "2024-03-01", "documentType": "Уголовное дело"}],
    }
    monkeypatch.setattr(mod, "iter_zip_records", lambda path: [appeal] if path.name == mod.APPEALS.filename else [first])
    monkeypatch.setattr(mod, "has_article_290", lambda record: True)
    monkeypatch.setattr(mod, "sentence_number", lambda record: True)
    monkeypatch.setattr(mod, "in_target_period", lambda record: False)
    monkeypatch.setattr(mod, "link_first_instance", lambda record, index: ("found", first))
    mod.outside_candidates(tmp_path)
    rows = mod.read_jsonl(tmp_path / "work" / "appeal-candidates.jsonl")
    assert rows[0]["aliases"][0]["number"] == "10-99/2024"
