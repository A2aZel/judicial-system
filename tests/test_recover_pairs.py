import json

from tools import recover_pairs as mod
from tests.test_training_dataset import moscow_candidate, moscow_document, moscow_valid_texts


def candidate_for_recovery():
    candidate = moscow_candidate()
    candidate["first"]["attachments"] = [{"displayName": "Приговор", "link": "sentence", "date": "01.02.2024"}]
    candidate["first_attachments"] = [{"link": "sentence", "date": "01.02.2024"}]
    candidate["appeal_attachments"] = [{"link": "appeal", "date": "01.03.2024"}]
    return candidate


def test_backup_requires_unique_compatible_first_instance():
    candidate = candidate_for_recovery()
    first = candidate["first"]
    earlier = dict(first, _id="earlier", dateFinal="01.01.2024")
    index = {("SYNTHETIC-UID", "1-1/2024"): [earlier]}
    assert mod.backup_first(candidate, index) == earlier
    index[("SYNTHETIC-UID", "1-1/2024")].append(dict(first, _id="other", dateFinal="01.01.2024"))
    assert mod.backup_first(candidate, index) is None


def test_recovery_continues_after_terminal_source_status(tmp_path, monkeypatch):
    sentence, appeal = moscow_valid_texts()
    candidate = candidate_for_recovery()
    mod.write_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", [candidate])
    mod.atomic_json(tmp_path / "pair-progress.json", {"synthetic-moscow": {"status": "verification_failed", "errors": ["sentence_number_mismatch"]}})
    monkeypatch.setattr(mod, "select_candidates", lambda limit: [{"candidate": candidate, "state": {"status": "verification_failed"}, "raw_dir": tmp_path, "priority": "wrong_attachment", "source": "synthetic"}])
    monkeypatch.setattr(mod, "load_backup_index", lambda: {})
    def fake_ensure(raw_dir, appeal_id, kind, attachment, current, progress_path, progress, pause):
        document = moscow_document(raw_dir, kind, sentence if kind == "sentence" else appeal)
        current["documents"][kind] = document
        return document, sentence if kind == "sentence" else appeal
    monkeypatch.setattr(mod, "ensure_document", fake_ensure)
    result = mod.recover(tmp_path, pause=0, limit=1)
    assert result["statuses"] == {"confirmed_pair": 1}


def test_incompatible_combination_is_not_recovered(tmp_path, monkeypatch):
    candidate = candidate_for_recovery()
    candidate["first_attachments"][0]["date"] = "02.03.2024"
    mod.write_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", [candidate])
    mod.atomic_json(tmp_path / "pair-progress.json", {"synthetic-moscow": {"status": "no_compatible_attachment", "errors": []}})
    monkeypatch.setattr(mod, "select_candidates", lambda limit: [{"candidate": candidate, "state": {"status": "no_compatible_attachment"}, "raw_dir": tmp_path, "priority": "format_mismatch", "source": "synthetic"}])
    monkeypatch.setattr(mod, "load_backup_index", lambda: {})
    result = mod.recover(tmp_path, pause=0, limit=1)
    assert result["statuses"] == {"no_compatible_attachment": 1}


def test_report_has_only_aggregates(tmp_path):
    mod.write_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", [])
    mod.atomic_json(tmp_path / "pair-progress.json", {})
    mod.write_report(tmp_path, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Recovered full pairs: 0" in text
    assert "http" not in text
    assert "case_id" not in text
