import hashlib
import io
import json
from pathlib import Path

import pytest

from tools import prepare_bash_tat_nov_pairs as mod
from tools.regional_courts import AccessBlocked, CaptchaError, DownloadError


SENTENCE = (
    "ПРИГОВОР Тестового районного суда от 01.02.2024 по делу 1-1/2024. "
    "Суд ПРИГОВОРИЛ. Синтетический текст. " + "мотивировка " * 150
)
APPEAL = (
    "АПЕЛЛЯЦИОННОЕ ОПРЕДЕЛЕНИЕ Верховного суда 22-1/2024 от 01.03.2024. "
    "Рассмотрена жалоба на приговор Тестового районного суда от 01.02.2024 по делу 1-1/2024. "
    "ОПРЕДЕЛИЛ: Приговор оставить без изменения. " + "доводы " * 150
)


def metadata():
    return {
        "lower_number": "1-1/2024",
        "lower_court": "Тестовый районный суд",
        "sentence_date": "01.02.2024",
        "decision_date": "01.03.2024",
        "identity_link_verified": True,
    }


def test_verify_pair_requires_full_texts_types_number_court_dates_and_outcome():
    assert mod.verify_pair(metadata(), SENTENCE, APPEAL) == []
    assert mod.verify_pair(metadata(), "ПРИГОВОР", APPEAL) == ["text_too_short"]
    assert "sentence_number_mismatch" in mod.verify_pair(metadata(), SENTENCE.replace("1-1/2024", "1-11/2024"), APPEAL)
    assert "sentence_number_conflict" in mod.verify_pair(metadata(), SENTENCE, APPEAL.replace("1-1/2024", "1-2/2024"))
    without_lower_number = APPEAL.replace(" по делу 1-1/2024", "")
    assert mod.verify_pair(metadata(), SENTENCE, without_lower_number) == []
    unbound = dict(metadata(), identity_link_verified=False)
    assert "appeal_number_mismatch" in mod.verify_pair(unbound, SENTENCE, without_lower_number)
    assert "lower_court_mismatch" in mod.verify_pair(metadata(), SENTENCE.replace("Тестового", "Иного"), APPEAL)
    assert "sentence_date_mismatch" in mod.verify_pair(metadata(), SENTENCE.replace("01.02.2024", "02.02.2024"), APPEAL)
    assert "appeal_date_mismatch" in mod.verify_pair(metadata(), SENTENCE, APPEAL.replace("01.03.2024", "02.03.2024"))
    missing_date = dict(metadata(), sentence_date=None)
    assert "sentence_date_missing" in mod.verify_pair(missing_date, SENTENCE, APPEAL)
    procedural = APPEAL.replace("на приговор", "на постановление о восстановлении срока обжалования приговора")
    assert "appeal_type_not_confirmed" in mod.verify_pair(metadata(), SENTENCE, procedural)
    new_trial = APPEAL.replace("АПЕЛЛЯЦИОННОЕ ОПРЕДЕЛЕНИЕ", "АПЕЛЛЯЦИОННЫЙ ПРИГОВОР")
    assert "new_trial_verdict_not_review_act" in mod.verify_pair(metadata(), SENTENCE, new_trial)
    assert mod.extract_outcome("Защитник просит приговор отменить. ОПРЕДЕЛИЛ: жалобу оставить без удовлетворения.") == "unknown"


def make_record(root: Path, record_id: str, source: str, document_type: str, text: str, number: str = "1-1/2024"):
    safe_id = record_id.replace("_", "-")
    text_path = root / "texts" / f"{safe_id}.txt"
    raw_path = root / "raw" / f"{safe_id}.html"
    evidence_path = root / "evidence" / f"{safe_id}.html"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    raw_path.write_bytes(b"synthetic raw")
    evidence_path.write_bytes(b"synthetic official evidence")
    text_sha = hashlib.sha256(text_path.read_bytes()).hexdigest()
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    evidence_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    identity = {
        "record_id": record_id,
        "source": source,
        "text_path": str(text_path.relative_to(root)),
        "text_sha256": text_sha,
        "raw_path": str(raw_path.relative_to(root)),
        "raw_sha256": raw_sha,
        "text_characters": len(text),
        "document_urls": [],
        "judicial_uids": [],
        "first_court": "Тестовый районный суд",
        "first_case_number": number,
        "first_decision_date": "2024-02-01",
        "document_type": document_type,
        "identity_status": "proven",
        "identity_link_verified": True,
        "identity_link_method": "official_listing_same_row",
        "identity_evidence_path": str(evidence_path.relative_to(root)),
        "identity_link_evidence_sha256": evidence_sha,
        "full_relevant_merits_text": True,
    }
    record = {
        "underlying_case_id": "case-" + record_id,
        "case_keys": [f"first_case:Тестовый районный суд|{number}"],
        "identity": identity,
    }
    meta = {
        "source": source,
        "text_sha256": text_sha,
        "court": "Верховный суд",
        "case_number": "22-1/2024",
        "decision_date": "01.03.2024",
        "document_id": "doc-" + record_id,
        "judicial_act_kind": document_type,
    }
    return record, meta


def write_inputs(root: Path, records, metadata_rows):
    index = root / "index.jsonl"
    metadata = root / "metadata.jsonl"
    index.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    metadata.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata_rows), encoding="utf-8")
    return index, metadata


def blocked_access():
    return {source: {"status": "robots_blocked", "rules": ["/*name=sud_delo"]} for source in mod.SOURCES}


def test_prepare_pairs_end_to_end_and_aggregate_report(tmp_path):
    appeal, appeal_meta = make_record(tmp_path, "a1", "bashkortostan", "appeal_against_verdict", APPEAL)
    sentence, sentence_meta = make_record(tmp_path, "s1", "other", "trial_verdict", SENTENCE)
    missing, missing_meta = make_record(tmp_path, "a2", "tatarstan", "appeal_against_verdict", APPEAL.replace("оставить без изменения", "изменить"), "1-2/2024")
    lonely, lonely_meta = make_record(tmp_path, "s2", "novosibirsk", "trial_verdict", SENTENCE, "1-3/2024")
    new_trial, new_meta = make_record(tmp_path, "n1", "bashkortostan", "appellate_verdict", APPEAL, "1-4/2024")
    procedural, procedural_meta = make_record(tmp_path, "p1", "tatarstan", "criminal_appeal_other", APPEAL, "1-5/2024")
    index, metadata_path = write_inputs(
        tmp_path,
        [appeal, sentence, missing, lonely, new_trial, procedural],
        [appeal_meta, sentence_meta, missing_meta, lonely_meta, new_meta, procedural_meta],
    )
    output = tmp_path / "output"
    summary = mod.prepare_pairs(index, metadata_path, output, tmp_path, access=blocked_access())
    assert summary["candidates"] == 5
    assert summary["confirmed_pairs"] == 1
    assert summary["missing_sentence"] == 1
    assert summary["missing_appeal"] == 1
    assert summary["rejected_new_trial"] == 1
    assert summary["rejected_procedural"] == 1
    assert summary["confirmed_outcomes"]["unchanged"] == 1
    rows = [json.loads(line) for line in (output / "pair-status.jsonl").read_text(encoding="utf-8").splitlines()]
    confirmed = next(row for row in rows if row["pair_status"] == "confirmed_pair")
    assert confirmed["sentence"]["sha256"] == sentence["identity"]["text_sha256"]
    assert confirmed["appeal"]["sha256"] == appeal["identity"]["text_sha256"]
    assert confirmed["verdict_group_id"].startswith("vg_")
    assert confirmed["identity_errors"] == []
    report = (output / "report.md").read_text(encoding="utf-8")
    for forbidden in ("1-1/2024", "22-1/2024", "https://", text_sha if (text_sha := appeal["identity"]["text_sha256"]) else "", "a1"):
        assert forbidden not in report
    assert summary["processed_dataset_included"] is False


def test_duplicate_event_and_distinct_events_with_same_uid(tmp_path):
    first, first_meta = make_record(tmp_path, "a1", "bashkortostan", "appeal_against_verdict", APPEAL)
    second = json.loads(json.dumps(first))
    second["identity"]["record_id"] = "a2"
    second["underlying_case_id"] = "case-a2"
    third, third_meta = make_record(tmp_path, "a3", "bashkortostan", "appeal_against_verdict", APPEAL.replace("01.03.2024", "02.03.2024"), "1-2/2024")
    third_meta["case_number"] = "22-2/2024"
    third_meta["decision_date"] = "02.03.2024"
    for record in (first, second, third):
        record["identity"]["judicial_uids"] = ["02RS0001-01-2024-000001-01"]
    index, metadata_path = write_inputs(tmp_path, [first, second, third], [first_meta, third_meta])
    summary = mod.prepare_pairs(index, metadata_path, tmp_path / "output", tmp_path, access=blocked_access())
    assert summary["missing_sentence"] == 2
    assert summary["duplicates"] == 1
    assert summary["identity_failures"] == 0
    rows = [json.loads(line) for line in (tmp_path / "output" / "pair-status.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({row["appeal_event_id"] for row in rows if row["pair_status"] != "duplicate_event"}) == 2


def test_sha_mismatch_never_confirms(tmp_path):
    appeal, appeal_meta = make_record(tmp_path, "a1", "bashkortostan", "appeal_against_verdict", APPEAL)
    sentence, sentence_meta = make_record(tmp_path, "s1", "other", "trial_verdict", SENTENCE)
    Path(tmp_path / sentence["identity"]["text_path"]).write_text(SENTENCE + "повреждение", encoding="utf-8")
    index, metadata_path = write_inputs(tmp_path, [appeal, sentence], [appeal_meta, sentence_meta])
    summary = mod.prepare_pairs(index, metadata_path, tmp_path / "output", tmp_path, access=blocked_access())
    assert summary["confirmed_pairs"] == 0
    assert summary["artifact_invalid"] == 1
    row = json.loads((tmp_path / "output" / "pair-status.jsonl").read_text(encoding="utf-8").strip())
    assert "text_sha256_mismatch" in row["identity_errors"]


class FakeResponse:
    def __init__(self, payload):
        self.payload = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size=-1):
        return self.payload.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeOpener:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def open(self, request, timeout):
        self.calls += 1
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return FakeResponse(payload)


def allowed_access():
    return {source: {"status": "allowed", "rules": []} for source in mod.SOURCES}


def direct_sentence_url():
    return "https://vs.bkr.sudrf.ru/modules.php?name=sud_delo&name_op=doc&number=1"


def test_direct_document_resume_uses_cache_and_no_second_request(tmp_path):
    appeal, appeal_meta = make_record(tmp_path, "a1", "bashkortostan", "appeal_against_verdict", APPEAL)
    appeal_meta["sentence_url"] = direct_sentence_url()
    index, metadata_path = write_inputs(tmp_path, [appeal], [appeal_meta])
    opener = FakeOpener([("<html><body>" + SENTENCE + "</body></html>").encode("utf-8")])
    first = mod.prepare_pairs(index, metadata_path, tmp_path / "output", tmp_path, pause=0, access=allowed_access(), opener=opener)
    second = mod.prepare_pairs(index, metadata_path, tmp_path / "output", tmp_path, pause=0, access=allowed_access(), opener=opener)
    assert first["confirmed_pairs"] == second["confirmed_pairs"] == 1
    assert opener.calls == 1


def test_search_and_robots_blocked_urls_are_rejected_before_transport():
    with pytest.raises(DownloadError):
        mod.check_url("https://vs.bkr.sudrf.ru/search?q=1", "bashkortostan", [])
    with pytest.raises(AccessBlocked):
        mod.check_url(direct_sentence_url(), "bashkortostan", ["/*name=sud_delo"])


def test_captcha_is_not_cached_or_bypassed(tmp_path):
    metadata_row = {"sentence_url": direct_sentence_url()}
    with pytest.raises(CaptchaError):
        mod.explicit_artifact(
            metadata_row, "sentence", "bashkortostan", "event", tmp_path / "output",
            allowed_access(), pause=0, opener=FakeOpener([b"SmartCaptcha"]), root=tmp_path,
        )
    assert not list((tmp_path / "output").glob("artifacts/*"))
