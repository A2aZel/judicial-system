import json
import urllib.error

import pytest

from tools import pair_preparation as mod


def valid_texts():
    sentence = "ПРИГОВОР Тестового районного суда от 01.02.2024 по делу 1-1/2024. Суд установил обстоятельства. Суд ПРИГОВОРИЛ. " + "мотивировка " * 150
    appeal = (
        "АПЕЛЛЯЦИОННОЕ ОПРЕДЕЛЕНИЕ Московского городского суда 10-1/2024 от 01.03.2024. "
        "Рассмотрена жалоба на приговор Тестового районного суда от 01.02.2024 по делу 1-1/2024. "
        "ОПРЕДЕЛИЛ: Приговор оставить без изменения, " + "доводы " * 150
    )
    return sentence, appeal


def candidate(number="10-1/2024", date="01.03.2024", aliases=None):
    return {
        "appeal_id": "synthetic-" + number,
        "identity": {
            "uid": "SYNTHETIC-UID", "number": number, "year": 2024,
            "courtName": "Московский городской суд", "act_date": date,
            "documentType": "Уголовное дело (в апелляции)",
        },
        "aliases": aliases or [],
        "lower": [{"number": "1-1/2024", "courtName": "Тестовый районный суд"}],
        "first": {
            "uid": "SYNTHETIC-UID", "number": "1-1/2024", "year": 2024,
            "courtName": "Тестовый районный суд", "dateFinal": "01.02.2024",
            "lastHearingDate": "01.02.2024",
        },
        "appeal_attachments": [],
        "first_attachments": [],
    }


def entry(document_type="trial_verdict"):
    return {
        "underlying_case_id": "synthetic-underlying",
        "identity": {
            "source": "mosgorsud", "document_type": document_type,
            "judicial_uids": ["SYNTHETIC-UID"],
            "first_court": "Тестовый районный суд", "first_case_number": "1-1/2024",
            "first_decision_date": "2024-02-01",
        },
    }


def document(root, name, text):
    raw = root / "corpus/mosgorsud/raw/texts" / (name + ".bin")
    text_path = raw.with_suffix(".txt")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(text.encode("utf-8"))
    text_path.write_text(text, encoding="utf-8")
    raw_sha = mod.sha256_file(raw)
    return {
        "path": raw.name, "sha256": raw_sha, "text_path": text_path.name,
        "text_sha256": mod.sha256_file(text_path), "text_source_sha256": raw_sha,
    }


def fixture(root, candidates=None, sentence_text=None, appeal_text=None, with_appeal=True):
    sentence, appeal = valid_texts()
    sentence_text = sentence if sentence_text is None else sentence_text
    appeal_text = appeal if appeal_text is None else appeal_text
    sentence_doc = document(root, "sentence", sentence_text)
    appeal_doc = document(root, "appeal", appeal_text)
    item = entry()
    item["identity"].update({
        "text_path": "corpus/mosgorsud/raw/texts/sentence.txt",
        "text_sha256": sentence_doc["text_sha256"],
        "raw_path": "corpus/mosgorsud/raw/texts/sentence.bin",
        "raw_sha256": sentence_doc["sha256"],
    })
    (root / "index.jsonl").write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    raw = root / "corpus/mosgorsud/raw"
    candidates = [candidate()] if candidates is None else candidates
    mod.atomic_jsonl(raw / "work/appeal-candidates.jsonl", candidates)
    progress = {}
    for item_candidate in candidates:
        documents = {"sentence": sentence_doc}
        if with_appeal:
            documents["appeal"] = appeal_doc
        progress[item_candidate["appeal_id"]] = {"status": "confirmed_pair", "documents": documents, "errors": []}
    mod.atomic_json(raw / "pair-progress.json", progress)
    return root / "index.jsonl", root / "pairs"


def test_full_pair_rechecks_sha_and_ignores_legacy_status(tmp_path):
    index, output = fixture(tmp_path)
    progress_path = tmp_path / "corpus/mosgorsud/raw/pair-progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    for state in progress.values():
        state["status"] = "verification_failed"
        state["errors"] = ["stale_error"]
    mod.atomic_json(progress_path, progress)
    rows = mod.prepare(index, output, tmp_path)
    assert len(rows) == 1
    assert rows[0]["pair_status"] == "confirmed_pair"
    assert rows[0]["outcome"] == "unchanged"
    assert rows[0]["identity_errors"] == []
    assert rows[0]["artifacts"]["sentence"]["text_sha256"]
    assert rows[0]["artifacts"]["appeal"]["raw_sha256"]


def test_integrity_mismatch_invalidates_pair_and_resume(tmp_path):
    index, output = fixture(tmp_path)
    assert mod.prepare(index, output, tmp_path)[0]["pair_status"] == "confirmed_pair"
    appeal = tmp_path / "corpus/mosgorsud/raw/texts/appeal.txt"
    appeal.write_text("повреждено", encoding="utf-8")
    rows = mod.prepare(index, output, tmp_path)
    assert rows[0]["pair_status"] == "integrity_mismatch"
    assert rows[0]["identity_errors"] == ["integrity_mismatch"]


def test_whitespace_and_ocr_identity_are_canonical(tmp_path):
    sentence = "ПРИГОВОР Тестового районного суда от 01 . 02 . 2024 по делу 01 - 0001 / 2024. Суд ПРИГОВОРИЛ. " + "мотивировка " * 150
    first = mod.canonical_verdict_group_id(sentence, mod.index_identity(entry()))
    spaced = sentence.replace("ПРИГОВОР", "П Р И Г О В О Р", 1)
    second = mod.canonical_verdict_group_id(spaced, mod.index_identity(entry()))
    assert first == second
    normalized = mod.dataset.normalized_verdict_text(sentence)
    assert normalized == mod.dataset.normalized_verdict_text(sentence.replace(" ", "  "))
    identity = mod.index_identity(entry())
    identity["number"] = mod.canonical_number("01 - 0001 / 2024", 2024)
    identity["date"] = mod.canonical_date("01 . 02 . 2024")
    assert mod.canonical_verdict_group_id(sentence, identity) == first
    assert mod.official.document_number_match(sentence, "1-1/2024", 2024)


def test_two_appeal_events_remain_separate_but_retrieval_dedupes(tmp_path):
    second_appeal = valid_texts()[1].replace("10-1/2024", "10-2/2024").replace("01.03.2024", "02.03.2024")
    second_doc = document(tmp_path, "appeal-2", second_appeal)
    candidates = [candidate(), candidate("10-2/2024", "02.03.2024")]
    index, output = fixture(tmp_path, candidates=candidates)
    progress_path = tmp_path / "corpus/mosgorsud/raw/pair-progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress[candidates[1]["appeal_id"]]["documents"]["appeal"] = second_doc
    mod.atomic_json(progress_path, progress)
    rows = mod.prepare(index, output, tmp_path)
    assert len(rows) == 2
    assert {row["pair_status"] for row in rows} == {"confirmed_pair"}
    assert len({row["case_id"] for row in rows}) == 2
    assert len({row["verdict_group_id"] for row in rows}) == 1
    assert len(mod.deduplicate_retrieval(rows)) == 1


def test_missing_sentence_and_missing_appeal(tmp_path):
    index, output = fixture(tmp_path, candidates=[])
    assert mod.prepare(index, output, tmp_path)[0]["pair_status"] == "missing_appeal"
    item = entry("appeal_against_verdict")
    item["identity"]["text_path"] = "missing.txt"
    item["identity"]["text_sha256"] = "0" * 64
    (tmp_path / "appeal-only.jsonl").write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = mod.prepare(tmp_path / "appeal-only.jsonl", tmp_path / "other", tmp_path)
    assert rows[0]["pair_status"] == "missing_sentence"


def test_missing_operative_marker_is_not_confirmed(tmp_path):
    appeal = valid_texts()[1].replace("ОПРЕДЕЛИЛ: Приговор оставить без изменения,", "Мотивы указаны ниже.")
    index, output = fixture(tmp_path, appeal_text=appeal)
    row = mod.prepare(index, output, tmp_path)[0]
    assert row["pair_status"] == "identity_failed"
    assert row["outcome"] == "unknown"
    assert "appeal_type_not_confirmed" in row["identity_errors"]


def test_regional_pair_requires_motivational_part():
    sentence, appeal = valid_texts()
    abridged = sentence.replace("Суд установил обстоятельства. ", "")
    assert "sentence_full_text_not_confirmed" in mod.verify_regional_pair(candidate(), abridged, appeal)


def test_procedural_act_is_not_merits_pair(tmp_path):
    appeal = valid_texts()[1].replace("Приговор оставить без изменения,", "Срок обжалования восстановить.")
    index, output = fixture(tmp_path, appeal_text=appeal)
    row = mod.prepare(index, output, tmp_path)[0]
    assert row["pair_status"] == "procedural_act"
    assert row["outcome"] == "unknown"


def test_only_matching_official_aliases_are_accepted(tmp_path):
    aliases = [{"number": "10-99/2024", "year": 2024, "date": "01.03.2024", "documentType": "Уголовное дело (в апелляции)"}]
    appeal = valid_texts()[1].replace("10-1/2024", "10-99/2024")
    index, output = fixture(tmp_path, candidates=[candidate(aliases=aliases)], appeal_text=appeal)
    assert mod.prepare(index, output, tmp_path)[0]["pair_status"] == "confirmed_pair"
    unofficial = candidate(aliases=aliases)
    assert mod.confirmed_candidate_aliases(unofficial) == []
    aliases[0]["date"] = "02.03.2024"
    index, output = fixture(tmp_path, candidates=[candidate(aliases=aliases)], appeal_text=appeal)
    assert mod.prepare(index, output, tmp_path)[0]["pair_status"] == "identity_failed"


@pytest.mark.parametrize("disposition", [
    "Приговор отменить, дело направить на новое рассмотрение,",
    "Приговор отменить, дело передать на новое судебное рассмотрение,",
    "Приговор изменить, дело в части ареста передать на новое судебное разбирательство,",
])
def test_new_trial_is_not_confirmed(tmp_path, disposition):
    appeal = valid_texts()[1].replace("Приговор оставить без изменения,", disposition)
    index, output = fixture(tmp_path, appeal_text=appeal)
    row = mod.prepare(index, output, tmp_path)[0]
    assert row["pair_status"] == "new_trial"
    assert row["outcome"] in {"cancelled", "changed"}
    assert "new_trial" in row["identity_errors"]


def test_returned_presentation_is_procedural_not_confirmed(tmp_path):
    appeal = valid_texts()[1].replace(
        "Приговор оставить без изменения,",
        "Определение о возвращении без рассмотрения апелляционного представления оставить без изменения. Приговор оставить без изменения,",
    )
    index, output = fixture(tmp_path, appeal_text=appeal)
    row = mod.prepare(index, output, tmp_path)[0]
    assert row["pair_status"] == "procedural_act"
    assert row["outcome"] == "unknown"


def test_resume_skips_reverification_and_rewrites_outputs(tmp_path, monkeypatch):
    index, output = fixture(tmp_path)
    calls = []
    original = mod.process_candidate

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(mod, "process_candidate", counted)
    mod.prepare(index, output, tmp_path)
    first = (output / "raw/pair-status.jsonl").read_bytes()
    (output / "raw/pair-status.jsonl").write_text("", encoding="utf-8")
    mod.prepare(index, output, tmp_path)
    assert len(calls) == 1
    assert (output / "raw/pair-status.jsonl").read_bytes() == first


def test_duplicate_event_and_public_report_aggregates(tmp_path):
    index, output = fixture(tmp_path)
    row = mod.prepare(index, output, tmp_path)[0]
    rows, duplicates = mod.mark_duplicates([row, dict(row)])
    assert duplicates == 1
    assert collections_status(rows) == {"confirmed_pair": 1, "duplicate": 1}
    report = mod.render_report(rows, 1, duplicates)
    for forbidden in ("SYNTHETIC-UID", "1-1/2024", "https://", row["case_id"], row["artifacts"]["appeal"]["text_sha256"]):
        assert forbidden not in report
    assert "Confirmed pairs: **1**" in report


def test_recovers_official_pair_missing_from_unified_index_without_duplicates(tmp_path):
    index, output = fixture(tmp_path)
    index.write_text("", encoding="utf-8")
    recovered = mod.prepare(index, output, tmp_path)
    assert len(recovered) == 1
    assert recovered[0]["pair_status"] == "confirmed_pair"
    assert recovered[0]["discovery"] == "official_pair_recovery"

    index, _ = fixture(tmp_path)
    indexed = mod.prepare(index, output, tmp_path)
    assert len(indexed) == 1
    assert indexed[0]["pair_status"] == "confirmed_pair"
    assert indexed[0].get("discovery") is None


def test_recovery_does_not_reject_or_reintroduce_unconfirmed_candidate(tmp_path):
    appeal = valid_texts()[1].replace("Приговор оставить без изменения,", "Срок обжалования восстановить.")
    index, output = fixture(tmp_path, appeal_text=appeal)
    index.write_text("", encoding="utf-8")
    assert mod.prepare(index, output, tmp_path) == []


def collections_status(rows):
    result = {}
    for row in rows:
        result[row["pair_status"]] = result.get(row["pair_status"], 0) + 1
    return result


def test_download_policy_delay_and_access_denial(tmp_path):
    assert mod.allowed_direct_url("https://mos-gorsud.ru/mgs/cases/docs/content/document")
    assert not mod.allowed_direct_url("https://mos-gorsud.ru/search?q=test")
    assert not mod.allowed_direct_url("https://example.invalid/mgs/cases/docs/content/document")
    delays = []

    def denied(*args, **kwargs):
        raise urllib.error.HTTPError("https://mos-gorsud.ru/mgs/cases/docs/content/document", 403, "denied", {}, None)

    with pytest.raises(mod.AccessDenied):
        mod.download_artifact(tmp_path, tmp_path / "pairs", "https://mos-gorsud.ru/mgs/cases/docs/content/document", delays.append, denied)
    assert delays == [4.0]


def test_resume_prefers_verified_download_over_stale_missing_state(tmp_path):
    index, output = fixture(tmp_path)
    row = mod.prepare(index, output, tmp_path)[0]
    progress_path = output / "raw/pair-progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    saved = next(iter(progress["entries"].values()))
    stale = json.loads(json.dumps(saved))
    stale["row"]["case_id"] = "case_stale"
    stale["row"]["pair_status"] = "missing_appeal"
    stale["row"]["outcome"] = "unknown"
    stale["row"]["artifacts"]["appeal"] = None
    progress["entries"]["case_stale"] = stale
    mod.atomic_json(progress_path, progress)
    resumed = mod.prepare(index, output, tmp_path)
    assert resumed[0]["case_id"] == row["case_id"]
    assert resumed[0]["pair_status"] == "confirmed_pair"
