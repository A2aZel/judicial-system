import hashlib
import io
import json
import zipfile
from urllib.error import HTTPError

import pytest

from tools import mosgorsud_open_data as mod


def record(**overrides):
    value = {
        "_id": "synthetic-appeal-1",
        "uid": "SYNTHETIC-UID",
        "number": "10-1/2024",
        "year": 2024,
        "dateReg": "10.02.2024",
        "dateFinal": "01.03.2024",
        "courtCode": "APPEAL",
        "courtName": "Синтетический апелляционный суд",
        "documentType": "Уголовное дело",
        "lower": [{"number": "1-1/2024", "courtCode": "FIRST", "courtName": "Тестовый районный суд"}],
        "participants": [{"displayName": "Синтетический Участник", "codexArticles": [{"name": "Ст. 290, Ч. 3"}]}],
        "attachments": [],
    }
    value.update(overrides)
    return value


def first_record(**overrides):
    value = record(
        _id="synthetic-first-1",
        number="1-1/2024",
        courtCode="FIRST",
        courtName="Тестовый районный суд",
        dateFinal="01.02.2024",
        lastHearingDate="01.02.2024",
        lower=[],
    )
    value.update(overrides)
    return value


def test_streaming_array_handles_utf8_boundaries_and_close():
    payload = [{"text": "ёж", "nested": [{"value": 1}]}, {"text": "тест"}, {}]
    stream = io.BytesIO((json.dumps(payload, ensure_ascii=False) + " \n").encode("utf-8"))
    assert list(mod.iter_json_array(stream, chunk_size=3)) == payload


def test_streaming_array_rejects_trailing_data():
    stream = io.BytesIO(b'[{"ok": true}] {"unexpected": 1}')
    with pytest.raises(ValueError, match="после"):
        list(mod.iter_json_array(stream, chunk_size=5))


@pytest.mark.parametrize("value,expected", [
    ("Ст. 290, Ч. 6", True),
    ("ст290 ч.1", True),
    ("Ст. 1290", False),
    ("Ст. 290.1", False),
    ("290", False),
])
def test_article_290_is_structured_and_exact(value, expected):
    assert mod.article_290(value) is expected
    synthetic = record(participants=[{"codexArticles": [{"name": value}]}])
    assert mod.has_article_290(synthetic) is expected


def test_period_dimensions_are_not_conflated():
    synthetic = record(year=2021, number="10-1/2021", dateReg="02.02.2022", dateFinal="03.03.2023")
    assert mod.in_target_period(synthetic)
    assert mod.record_years(synthetic) == {"number": 2021, "registration": 2022, "act": [2023]}


def test_modern_sentence_number_is_candidate_only():
    assert mod.sentence_number(record(lower=[{"number": "01-0001/2025"}]))
    assert not mod.sentence_number(record(lower=[{"number": "3/1-0001/2025"}]))


def test_link_uses_uid_court_number_without_deduplicating_appeal_events():
    first = first_record()
    index = mod.first_index([first])
    appeal_1 = record()
    appeal_2 = record(_id="synthetic-appeal-2", number="10-2/2024")
    assert mod.link_first_instance(appeal_1, index) == ("found", first)
    assert mod.link_first_instance(appeal_2, index) == ("found", first)
    wrong_court = record(lower=[{"number": "1-1/2024", "courtCode": "OTHER", "courtName": "Иной суд"}])
    assert mod.link_first_instance(wrong_court, index) == ("not_found", None)


def test_link_rejects_ambiguous_identity():
    first = first_record()
    second = first_record(_id="synthetic-first-2", dateFinal="02.02.2024")
    index = mod.first_index([first, second])
    assert mod.link_first_instance(record(), index) == ("ambiguous", None)


def test_attachment_types_and_confirmed_aliases():
    synthetic = record(
        attachments=[
            {"displayName": "Постановление суда апелляционной инстанции", "link": "https://data.mos-gorsud.ru/mgs/cases/docs/content/test"},
            {"displayName": "Апелляционная жалоба", "link": "https://data.mos-gorsud.ru/mgs/cases/docs/content/other"},
        ],
        upper=[{"number": "10-99/2024", "year": 2024, "dateFinal": "01.03.2024", "documentType": "Уголовное дело"}],
    )
    assert len(mod.appellate_attachments(synthetic)) == 1
    assert mod.confirmed_aliases(synthetic)[0]["number"] == "10-99/2024"
    first = first_record(attachments=[{"displayName": "Приговор", "link": "https://data.mos-gorsud.ru/mgs/cases/docs/content/sentence"}])
    assert len(mod.sentence_attachments(first)) == 1


class FakeResponse:
    def __init__(self, url, payload, content_type="application/zip", status=200, headers=None):
        self.url = url
        self.payload = io.BytesIO(payload)
        self.status = status
        self.headers = headers or {"Content-Type": content_type}

    def read(self, size=-1):
        return self.payload.read(size)

    def __enter__(self):
        return self

    def close(self):
        return None

    def __exit__(self, *args):
        return False


class FakeOpener:
    def __init__(self, payload, content_type="application/zip", status=200, headers=None):
        self.payload = payload
        self.content_type = content_type
        self.status = status
        self.headers = headers
        self.calls = 0
        self.last_request = None

    def open(self, request, timeout):
        self.calls += 1
        self.last_request = request
        return FakeResponse(request.full_url, self.payload, self.content_type, self.status, self.headers)


def zip_payload():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.json", json.dumps([record()], ensure_ascii=False))
    return buffer.getvalue()


def test_downloader_is_atomic_and_skips_valid_zip(tmp_path):
    spec = mod.DatasetSpec("synthetic", "synthetic-dataset", 1, "2026-01-01", "synthetic.zip")
    opener = FakeOpener(zip_payload())
    first = mod.download_dataset(spec, tmp_path, pause=0, opener=opener)
    second = mod.download_dataset(spec, tmp_path, pause=0, opener=opener)
    assert opener.calls == 1
    assert first["sha256"] == second["sha256"]
    assert mod.valid_zip(tmp_path / spec.filename)
    assert not (tmp_path / (spec.filename + ".part")).exists()


def test_downloader_resumes_partial_zip_with_range(tmp_path):
    spec = mod.DatasetSpec("synthetic", "synthetic-dataset", 1, "2026-01-01", "synthetic.zip")
    payload = zip_payload()
    part = tmp_path / (spec.filename + ".part")
    part.write_bytes(payload[:100])
    mod.atomic_json(tmp_path / (spec.filename + ".part.meta.json"), {"version": {"etag": '"synthetic-v1"'}})
    opener = FakeOpener(
        payload[100:], status=206,
        headers={"Content-Range": f"bytes 100-{len(payload) - 1}/{len(payload)}", "Content-Length": str(len(payload) - 100), "ETag": '"synthetic-v1"'},
    )
    mod.download_dataset(spec, tmp_path, pause=0, opener=opener)
    assert opener.last_request.get_header("Range") == "bytes=100-"
    assert (tmp_path / spec.filename).read_bytes() == payload


def test_downloader_stops_on_captcha(tmp_path):
    spec = mod.DatasetSpec("synthetic", "synthetic-dataset", 1, "2026-01-01", "synthetic.zip")
    opener = FakeOpener(b"<html>SmartCaptcha</html>", "text/html")
    with pytest.raises(mod.CaptchaError):
        mod.download_dataset(spec, tmp_path, pause=0, opener=opener)
    assert not (tmp_path / spec.filename).exists()


def synthetic_candidate():
    return {
        "identity": {"number": "10-1/2024", "year": 2024, "act_date": "01.03.2024", "documentType": "Уголовное дело"},
        "lower": [{"courtName": "Тестовый районный суд"}],
        "first": first_record(),
    }


def valid_pair_texts():
    sentence = "ПРИГОВОР Тестового районного суда от 01.02.2024 по делу 1-1/2024. Суд установил обстоятельства. Суд ПРИГОВОРИЛ. " + "мотивировка " * 150
    appeal = (
        "АПЕЛЛЯЦИОННОЕ ОПРЕДЕЛЕНИЕ Московского городского суда 10-1/2024 от 01.03.2024. "
        "Рассмотрена жалоба на приговор Тестового районного суда от 01.02.2024 по делу 1-1/2024. "
        "ОПРЕДЕЛИЛ: Приговор оставить без изменения, " + "доводы " * 150
    )
    return sentence, appeal


def test_pair_verification_requires_both_full_texts_and_identity():
    sentence, appeal = valid_pair_texts()
    assert mod.verify_pair(synthetic_candidate(), sentence, appeal) == []
    assert "appeal_date_mismatch" in mod.verify_pair(synthetic_candidate(), sentence, appeal.replace("01.03.2024", "02.03.2024"))
    assert mod.verify_pair(synthetic_candidate(), "ПРИГОВОР", appeal) == ["text_too_short"]
    abridged = sentence.replace("Суд установил обстоятельства. ", "")
    assert "sentence_full_text_not_confirmed" in mod.verify_pair(synthetic_candidate(), abridged, appeal)
    procedural = appeal.replace("на приговор", "на постановление о восстановлении срока обжалования приговора")
    assert "appeal_type_not_confirmed" in mod.verify_pair(synthetic_candidate(), sentence, procedural)


def test_number_normalization_accepts_zero_padding_and_short_year():
    variants = mod.normalize_number("01-0022/2023")
    assert "1-22/2023" in variants
    assert "1-22/23" in variants


def test_text_resume_does_not_repeat_saved_error(tmp_path):
    candidate = synthetic_candidate()
    candidate.update({"appeal_id": "synthetic-error", "appeal_attachments": [{"link": "https://data.mos-gorsud.ru/mgs/cases/docs/content/a"}], "first_attachments": [{"link": "https://data.mos-gorsud.ru/mgs/cases/docs/content/s"}]})
    mod.atomic_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", [candidate])
    mod.atomic_json(tmp_path / "pair-progress.json", {"synthetic-error": {"status": "download_error", "errors": ["HTTP 404"]}})
    result = mod.download_texts(tmp_path, tmp_path / "report.md", pause=0, limit=1)
    assert result["attempted"] == 0
    assert result["statuses"] == {"download_error": 1}


def test_html_extraction_ignores_scripts(tmp_path):
    path = tmp_path / "act.bin"
    path.write_bytes("<html><style>secret</style><body>Приговор<script>hidden</script>текст</body></html>".encode())
    text = mod.extract_text(path)
    assert "Приговор" in text
    assert "hidden" not in text
    assert "secret" not in text


def test_full_case_number_boundaries_reject_substring_identity():
    sentence, appeal = valid_pair_texts()
    wrong_sentence = sentence.replace("1-1/2024", "1-11/2024")
    assert not mod.document_number_match(wrong_sentence, "1-1/2024", 2024)
    assert "sentence_number_mismatch" in mod.verify_pair(synthetic_candidate(), wrong_sentence, appeal)
    assert mod.document_number_match(sentence, "1-1/2024", 2024)


def test_identity_requires_court_and_date_in_both_documents():
    sentence, appeal = valid_pair_texts()
    without_court = sentence.replace("Тестового районного суда", "Иного суда")
    assert "lower_court_mismatch" in mod.verify_pair(synthetic_candidate(), without_court, appeal)
    without_date = sentence.replace("01.02.2024", "02.02.2024")
    assert "sentence_date_mismatch" in mod.verify_pair(synthetic_candidate(), without_date, appeal)
    without_appeal_court = appeal.replace("Московского городского суда", "Иного суда")
    assert "appeal_court_mismatch" in mod.verify_pair(synthetic_candidate(), sentence, without_appeal_court)


def test_procedural_annulment_is_not_sentence_outcome():
    sentence, appeal = valid_pair_texts()
    procedural = (
        "АПЕЛЛЯЦИОННОЕ ПОСТАНОВЛЕНИЕ Московского городского суда 10-1/2024 от 01.03.2024. "
        "Жалоба на приговор Тестового районного суда от 01.02.2024. ПОСТАНОВИЛ: "
        "постановление об отказе восстановить срок обжалования приговора отменить, " + "доводы " * 150
    )
    assert mod.extract_outcome(procedural) == "unknown"
    assert "appeal_type_not_confirmed" in mod.verify_pair(synthetic_candidate(), sentence, procedural)
    assert mod.extract_outcome(appeal) == "unchanged"
    assert mod.extract_outcome(appeal.replace("оставить без изменения", "изменить")) == "changed"
    assert mod.extract_outcome(appeal.replace("оставить без изменения", "отменить")) == "cancelled"


def test_aliases_require_real_date_and_type_and_participate_in_identity():
    missing_date = record(upper=[{"number": "10-99/2024", "year": 2024, "documentType": "Уголовное дело"}])
    assert mod.confirmed_aliases(missing_date) == []
    sentence, appeal = valid_pair_texts()
    candidate = synthetic_candidate()
    candidate["aliases"] = [{"number": "10-99/2024", "year": 2024, "date": "01.03.2024", "documentType": "Уголовное дело"}]
    aliased_appeal = appeal.replace("10-1/2024", "10-99/2024")
    assert mod.verify_pair(candidate, sentence, aliased_appeal) == []


def test_transport_rejects_short_content_length(tmp_path):
    destination = tmp_path / "short.bin"
    payload = zip_payload()
    opener = FakeOpener(payload, headers={"Content-Length": str(len(payload) + 10)})
    with pytest.raises(mod.DownloadError, match="Неполный документ"):
        mod.download_file(mod.APPEALS.api_url, destination, pause=0, opener=opener)
    assert destination.with_name(destination.name + ".part").exists()


def test_transport_read_error_becomes_download_error(tmp_path):
    class BrokenResponse(FakeResponse):
        def __init__(self, url):
            super().__init__(url, b"prefix")
            self.calls = 0

        def read(self, size=-1):
            self.calls += 1
            if self.calls > 1:
                raise OSError("synthetic timeout")
            return b"prefix"

    class BrokenOpener:
        def open(self, request, timeout):
            return BrokenResponse(request.full_url)

    with pytest.raises(mod.DownloadError, match="Ошибка чтения"):
        mod.download_file(mod.APPEALS.api_url, tmp_path / "broken.bin", pause=0, opener=BrokenOpener())


def stored_document(raw_dir, kind, text):
    binary = raw_dir / "texts" / f"{kind}.bin"
    text_path = raw_dir / "texts" / f"{kind}.txt"
    mod.atomic_write(binary, text.encode("utf-8"))
    mod.atomic_write(text_path, text.encode("utf-8"))
    digest = mod.sha256_file(binary)
    return {
        "url": f"https://data.mos-gorsud.ru/mgs/cases/docs/content/{kind}",
        "path": binary.name,
        "bytes": binary.stat().st_size,
        "sha256": digest,
        "text_path": text_path.name,
        "text_source_sha256": digest,
        "text_sha256": mod.sha256_file(text_path),
    }


def test_revalidation_invalidates_and_restores_missing_artifacts(tmp_path):
    sentence, appeal = valid_pair_texts()
    candidate = synthetic_candidate()
    candidate["appeal_id"] = "synthetic-artifacts"
    mod.atomic_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", [candidate])
    documents = {"sentence": stored_document(tmp_path, "sentence", sentence), "appeal": stored_document(tmp_path, "appeal", appeal)}
    state = {"status": "confirmed_pair", "documents": documents, "errors": [], "outcome": "unchanged"}
    mod.atomic_json(tmp_path / "pair-progress.json", {"synthetic-artifacts": state})
    progress = mod.revalidate_progress(tmp_path)
    assert progress["synthetic-artifacts"]["status"] == "confirmed_pair"
    (tmp_path / "texts" / "sentence.txt").unlink()
    progress = mod.revalidate_progress(tmp_path)
    invalid = progress["synthetic-artifacts"]
    assert invalid["status"] == "artifact_invalid"
    assert invalid["both_files_downloaded"] is True
    assert invalid["both_texts_extracted"] is False
    progress_path = tmp_path / "pair-progress.json"
    mod.ensure_document(tmp_path, "synthetic-artifacts", "sentence", invalid["documents"]["sentence"], invalid, progress_path, progress, 0)
    progress = mod.revalidate_progress(tmp_path)
    assert progress["synthetic-artifacts"]["status"] == "confirmed_pair"
    (tmp_path / "texts" / "appeal.bin").write_bytes(b"corrupted")
    progress = mod.revalidate_progress(tmp_path)
    assert progress["synthetic-artifacts"]["status"] == "artifact_invalid"
    assert progress["synthetic-artifacts"]["both_files_downloaded"] is False


def test_new_binary_forces_text_reextraction(tmp_path, monkeypatch):
    old_text, _ = valid_pair_texts()
    new_text = old_text.replace("мотивировка", "новыйтекст")
    candidate = synthetic_candidate()
    candidate["appeal_id"] = "synthetic-new-binary"
    current = {"documents": {}}
    progress = {"synthetic-new-binary": current}
    progress_path = tmp_path / "pair-progress.json"
    attachment = {"link": "https://data.mos-gorsud.ru/mgs/cases/docs/content/sentence"}
    payloads = [old_text.encode("utf-8"), new_text.encode("utf-8")]

    def fake_download(url, destination, pause):
        payload = payloads.pop(0)
        part = destination.with_name(destination.name + ".part")
        part.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest(), len(payload)

    monkeypatch.setattr(mod, "download_file", fake_download)
    first, _ = mod.ensure_document(tmp_path, "synthetic-new-binary", "sentence", attachment, current, progress_path, progress, 0)
    first["sha256"] = "stale-checkpoint-hash"
    current["documents"]["sentence"] = first
    second, extracted = mod.ensure_document(tmp_path, "synthetic-new-binary", "sentence", attachment, current, progress_path, progress, 0)
    assert "новыйтекст" in extracted
    assert second["text_source_sha256"] == second["sha256"]
    assert second["text_sha256"] == mod.sha256_file(tmp_path / "texts" / second["text_path"])


def test_attachment_combinations_find_compatible_second_document(tmp_path, monkeypatch):
    sentence, appeal = valid_pair_texts()
    wrong_sentence = sentence.replace("1-1/2024", "1-11/2024")
    candidate = synthetic_candidate()
    candidate.update({
        "appeal_id": "synthetic-combinations",
        "first_attachments": [
            {"link": "https://data.mos-gorsud.ru/mgs/cases/docs/content/wrong", "date": "01.02.2024"},
            {"link": "https://data.mos-gorsud.ru/mgs/cases/docs/content/right", "date": "01.02.2024"},
        ],
        "appeal_attachments": [{"link": "https://data.mos-gorsud.ru/mgs/cases/docs/content/appeal", "date": "01.03.2024"}],
    })
    mod.atomic_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", [candidate])
    payloads = {"wrong": wrong_sentence.encode("utf-8"), "right": sentence.encode("utf-8"), "appeal": appeal.encode("utf-8")}

    def fake_download(url, destination, pause):
        key = url.rsplit("/", 1)[-1]
        payload = payloads[key]
        part = destination.with_name(destination.name + ".part")
        part.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest(), len(payload)

    monkeypatch.setattr(mod, "download_file", fake_download)
    result = mod.download_texts(tmp_path, tmp_path / "report.md", pause=0, limit=1)
    assert result["statuses"] == {"confirmed_pair": 1}
    progress = mod.read_json(tmp_path / "pair-progress.json", {})
    state = progress["synthetic-combinations"]
    assert len(state["attempts"]) == 2
    assert state["both_files_downloaded"] is True
    assert state["both_texts_extracted"] is True
    assert state["outcome"] == "unchanged"


def test_incompatible_attachment_dates_are_terminal_not_artifact_invalid(tmp_path):
    candidate = synthetic_candidate()
    candidate.update({
        "appeal_id": "synthetic-no-combination",
        "first_attachments": [{"link": "https://data.mos-gorsud.ru/mgs/cases/docs/content/sentence", "date": "02.03.2024"}],
        "appeal_attachments": [{"link": "https://data.mos-gorsud.ru/mgs/cases/docs/content/appeal", "date": "01.03.2024"}],
    })
    mod.atomic_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", [candidate])
    result = mod.download_texts(tmp_path, tmp_path / "report.md", pause=0, limit=1)
    assert result["statuses"] == {"no_compatible_attachment": 1}
    progress = mod.revalidate_progress(tmp_path)
    assert progress["synthetic-no-combination"]["status"] == "no_compatible_attachment"


def test_outcome_requires_resolution_marker_not_party_request():
    request_only = "Защитник просит приговор отменить. В удовлетворении отказать. " + "доводы " * 100
    assert mod.operative_text(request_only) is None
    assert mod.extract_outcome(request_only) == "unknown"
    assert not mod.has_sentence_operative(request_only)
    with_resolution = request_only + " ПОСТАНОВИЛ: апелляционную жалобу оставить без удовлетворения."
    assert mod.extract_outcome(with_resolution) == "unknown"


def test_missing_metadata_dates_prevent_confirmation():
    sentence, appeal = valid_pair_texts()
    candidate = synthetic_candidate()
    candidate["first"]["dateFinal"] = None
    candidate["first"]["lastHearingDate"] = None
    assert "sentence_date_missing" in mod.verify_pair(candidate, sentence, appeal)
    candidate = synthetic_candidate()
    candidate["identity"]["act_date"] = None
    assert "appeal_date_missing" in mod.verify_pair(candidate, sentence, appeal)
    candidate["identity"]["act_date"] = "not-a-date"
    assert "appeal_date_missing" in mod.verify_pair(candidate, sentence, appeal)


def test_attachment_combinations_do_not_stop_at_sixteen():
    candidate = synthetic_candidate()
    candidate["first_attachments"] = [{"link": "s1"}, {"link": "s2"}]
    candidate["appeal_attachments"] = [{"link": f"a{i}"} for i in range(16)]
    combinations = mod.attachment_combinations(candidate)
    assert len(combinations) == 32
    assert combinations[-1][0]["link"] == "s2"
    assert combinations[-1][1]["link"] == "a15"


def test_unsupported_format_survives_revalidate_and_resume(tmp_path):
    binary = tmp_path / "texts" / "unsupported.bin"
    mod.atomic_write(binary, b"not-a-supported-format")
    document = {
        "url": "https://data.mos-gorsud.ru/mgs/cases/docs/content/unsupported",
        "path": binary.name,
        "bytes": binary.stat().st_size,
        "sha256": mod.sha256_file(binary),
    }
    candidate = synthetic_candidate()
    candidate.update({
        "appeal_id": "synthetic-unsupported",
        "first_attachments": [{"link": document["url"]}],
        "appeal_attachments": [{"link": document["url"]}],
    })
    mod.atomic_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", [candidate])
    state = {"status": "unsupported_format", "documents": {"sentence": document, "appeal": document}, "errors": ["unsupported"]}
    mod.atomic_json(tmp_path / "pair-progress.json", {"synthetic-unsupported": state})
    progress = mod.revalidate_progress(tmp_path)
    assert progress["synthetic-unsupported"]["status"] == "unsupported_format"
    assert progress["synthetic-unsupported"]["both_files_downloaded"] is True
    assert progress["synthetic-unsupported"]["both_texts_extracted"] is False
    result = mod.download_texts(tmp_path, tmp_path / "report.md", pause=0, limit=1)
    assert result["attempted"] == 0


class QueueOpener:
    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append(request)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_range_restart_on_changed_resource_does_not_mix_versions(tmp_path):
    spec = mod.DatasetSpec("synthetic", "synthetic-dataset", 1, "2026-01-01", "synthetic.zip")
    old_payload = zip_payload()
    new_payload = zip_payload() + b"new-version"
    part = tmp_path / (spec.filename + ".part")
    part.write_bytes(old_payload[:100])
    mod.atomic_json(tmp_path / (spec.filename + ".part.meta.json"), {"version": {"etag": '"old"'}})
    partial = FakeResponse(
        spec.api_url, new_payload[100:], status=206,
        headers={"Content-Range": f"bytes 100-{len(new_payload) - 1}/{len(new_payload)}", "ETag": '"new"'},
    )
    complete = FakeResponse(spec.api_url, new_payload, headers={"ETag": '"new"'})
    opener = QueueOpener([partial, complete])
    mod.download_dataset(spec, tmp_path, pause=0, opener=opener)
    assert len(opener.calls) == 2
    assert (tmp_path / spec.filename).read_bytes() == new_payload


def test_http_416_accepts_complete_matching_part(tmp_path):
    spec = mod.DatasetSpec("synthetic", "synthetic-dataset", 1, "2026-01-01", "synthetic.zip")
    payload = zip_payload()
    part = tmp_path / (spec.filename + ".part")
    part.write_bytes(payload)
    mod.atomic_json(tmp_path / (spec.filename + ".part.meta.json"), {"version": {"etag": '"v1"'}})
    error = HTTPError(spec.api_url, 416, "range", {"Content-Range": f"bytes */{len(payload)}", "ETag": '"v1"'}, io.BytesIO())
    opener = QueueOpener([error])
    mod.download_dataset(spec, tmp_path, pause=0, opener=opener)
    assert (tmp_path / spec.filename).read_bytes() == payload
    assert len(opener.calls) == 1


def test_http_416_without_validator_safely_restarts(tmp_path):
    spec = mod.DatasetSpec("synthetic", "synthetic-dataset", 1, "2026-01-01", "synthetic.zip")
    payload = zip_payload()
    part = tmp_path / (spec.filename + ".part")
    part.write_bytes(payload)
    error = HTTPError(spec.api_url, 416, "range", {"Content-Range": f"bytes */{len(payload)}"}, io.BytesIO())
    complete = FakeResponse(spec.api_url, payload)
    opener = QueueOpener([error, complete])
    mod.download_dataset(spec, tmp_path, pause=0, opener=opener)
    assert len(opener.calls) == 2
    assert (tmp_path / spec.filename).read_bytes() == payload


def minimal_analysis():
    counts = mod.empty_status()
    counts.update({"first_dataset_records": 0, "first_article_290": 0, "link_ambiguous": 0})
    return {
        "generated_at": "synthetic",
        "counts": counts,
        "appeals_dataset_records": 0,
        "sources": [],
        "by_year": {dimension: {} for dimension in ("number", "registration", "act")},
        "resume": {"not_started": 0, "remaining": 0},
        "pair_statuses": {},
        "diagnostics": {"sentence_bad": 0, "sentence_number_conflict": 0, "manual_review": 0},
        "outcomes_confirmed": {"unchanged": 1},
        "outcomes_other": {"unknown": 2},
    }


def test_first_run_has_no_invented_history_and_outcome_is_split(tmp_path):
    report = tmp_path / "report.md"
    mod.write_report(minimal_analysis(), report)
    text = report.read_text(encoding="utf-8")
    assert "прежние 11" not in text
    assert "Изменение относительно сохранённого отчёта" not in text
    assert "Исход подтверждённых пар" in text
    assert "Исход других проверенных/неподтверждённых записей" in text


def candidate_for_download(identifier):
    candidate = synthetic_candidate()
    candidate.update({
        "appeal_id": identifier,
        "first_attachments": [{"link": f"https://data.mos-gorsud.ru/mgs/cases/docs/content/{identifier}-sentence", "date": "01.02.2024"}],
        "appeal_attachments": [{"link": f"https://data.mos-gorsud.ru/mgs/cases/docs/content/{identifier}-appeal", "date": "01.03.2024"}],
    })
    return candidate


def test_http_404_continues_with_next_candidate(tmp_path, monkeypatch):
    candidates = [candidate_for_download("missing"), candidate_for_download("available")]
    mod.atomic_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", candidates)
    sentence, appeal = valid_pair_texts()

    def fake_download(url, destination, pause):
        if "missing-sentence" in url:
            raise mod.DownloadError("HTTP 404")
        payload = sentence.encode("utf-8") if "available-sentence" in url else appeal.encode("utf-8")
        part = destination.with_name(destination.name + ".part")
        part.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest(), len(payload)

    monkeypatch.setattr(mod, "download_file", fake_download)
    result = mod.download_texts(tmp_path, tmp_path / "report.md", pause=0, limit=0)
    assert result["attempted"] == 2
    assert result["statuses"] == {"download_error": 1, "confirmed_pair": 1}
    assert result["stopped_reason"] is None


def test_three_consecutive_network_errors_stop_after_checkpoint(tmp_path, monkeypatch):
    candidates = [candidate_for_download(f"error-{index}") for index in range(4)]
    mod.atomic_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", candidates)

    def fake_download(url, destination, pause):
        raise mod.DownloadError("synthetic network timeout")

    monkeypatch.setattr(mod, "download_file", fake_download)
    result = mod.download_texts(tmp_path, tmp_path / "report.md", pause=0, limit=0)
    assert result["attempted"] == 3
    assert result["statuses"] == {"download_error": 3}
    assert result["stopped_reason"] == "network_error_series"


def test_lower_case_number_must_match_inside_appeal():
    sentence, appeal = valid_pair_texts()
    assert mod.verify_pair(synthetic_candidate(), sentence, appeal) == []
    wrong_lower_number = appeal.replace("по делу 1-1/2024", "по делу 1-11/2024")
    assert "sentence_number_conflict" in mod.verify_pair(synthetic_candidate(), sentence, wrong_lower_number)


def test_spaced_resolution_marker_excludes_prior_request_and_requires_decision():
    text = "Защитник просит приговор отменить. П О С Т А Н О В И Л: жалобу оставить без удовлетворения."
    assert mod.extract_outcome(text) == "unknown"
    assert not mod.has_sentence_operative(text)
    assert mod.extract_outcome("П О С Т А Н О В И Л:") == "unknown"


def test_multiple_metadata_dates_require_one_shared_sentence_date():
    sentence, appeal = valid_pair_texts()
    candidate = synthetic_candidate()
    candidate["first"]["lastHearingDate"] = "02.02.2024"
    assert mod.verify_pair(candidate, sentence, appeal) == []
    different_dates = appeal.replace("01.02.2024", "02.02.2024")
    assert "sentence_date_mismatch" in mod.verify_pair(candidate, sentence, different_dates)


def test_lawful_sentence_term_change_is_not_procedural_deadline():
    text = "ОПРЕДЕЛИЛ: Приговор в части назначенного срока лишения свободы изменить, " + "доводы " * 100
    assert mod.extract_outcome(text) == "changed"


def test_content_location_is_not_range_version_validator(tmp_path):
    spec = mod.DatasetSpec("synthetic", "synthetic-dataset", 1, "2026-01-01", "synthetic.zip")
    payload = zip_payload()
    part = tmp_path / (spec.filename + ".part")
    part.write_bytes(payload[:100])
    mod.atomic_json(tmp_path / (spec.filename + ".part.meta.json"), {"version": {"content_location": spec.api_url}})
    complete = FakeResponse(spec.api_url, payload)
    opener = QueueOpener([complete])
    mod.download_dataset(spec, tmp_path, pause=0, opener=opener)
    assert opener.calls[0].get_header("Range") is None
    assert opener.calls[0].get_header("If-Range") is None
    assert (tmp_path / spec.filename).read_bytes() == payload


def test_http_416_same_size_with_only_content_location_restarts(tmp_path):
    spec = mod.DatasetSpec("synthetic", "synthetic-dataset", 1, "2026-01-01", "synthetic.zip")
    payload = zip_payload()
    part = tmp_path / (spec.filename + ".part")
    part.write_bytes(payload)
    mod.atomic_json(tmp_path / (spec.filename + ".part.meta.json"), {"version": {"content_location": spec.api_url}})
    error = HTTPError(spec.api_url, 416, "range", {"Content-Range": f"bytes */{len(payload)}", "Content-Location": spec.api_url}, io.BytesIO())
    complete = FakeResponse(spec.api_url, payload)
    opener = QueueOpener([error, complete])
    mod.download_dataset(spec, tmp_path, pause=0, opener=opener)
    assert len(opener.calls) == 2
    assert (tmp_path / spec.filename).read_bytes() == payload


def test_http_error_body_captcha_stops_immediately(tmp_path):
    error = HTTPError(mod.APPEALS.api_url, 403, "forbidden", {}, io.BytesIO(b"SmartCaptcha"))
    with pytest.raises(mod.CaptchaError):
        mod.download_file(mod.APPEALS.api_url, tmp_path / "captcha.bin", pause=0, opener=QueueOpener([error]))


def test_captcha_does_not_continue_to_next_candidate(tmp_path, monkeypatch):
    candidates = [candidate_for_download("captcha"), candidate_for_download("next")]
    mod.atomic_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", candidates)

    def fake_download(url, destination, pause):
        raise mod.CaptchaError("captcha")

    monkeypatch.setattr(mod, "download_file", fake_download)
    with pytest.raises(mod.CaptchaError):
        mod.download_texts(tmp_path, tmp_path / "report.md", pause=0, limit=0)
    progress = mod.read_json(tmp_path / "pair-progress.json", {})
    assert list(progress) == ["captcha"]
    assert progress["captcha"]["status"] == "captcha"


def write_dataset_zip(path, records):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.json", json.dumps(records, ensure_ascii=False))


def test_target_article_records_schema_is_integer(tmp_path):
    write_dataset_zip(tmp_path / mod.APPEALS.filename, [record()])
    write_dataset_zip(tmp_path / mod.FIRST_INSTANCE.filename, [first_record()])
    analysis = mod.analyze(tmp_path, tmp_path / "report.md")
    assert analysis["target_article_records"] == 1
    assert isinstance(analysis["target_article_records"], int)


def test_appeal_without_lower_case_number_can_be_confirmed():
    sentence, appeal = valid_pair_texts()
    without_reference = appeal.replace(" по делу 1-1/2024", "")
    assert "10-1/2024" in without_reference
    assert "1-1/2024" not in without_reference.replace("10-1/2024", "")
    assert mod.verify_pair(synthetic_candidate(), sentence, without_reference) == []


def test_multiple_conflicting_lower_references_require_manual_review():
    sentence, appeal = valid_pair_texts()
    first_wrong = appeal.replace("по делу 1-1/2024", "по делу 1-11/2024")
    second_wrong = first_wrong + " Также рассмотрена жалоба на приговор по делу 1-12/2024."
    errors = mod.verify_pair(synthetic_candidate(), sentence, second_wrong)
    assert "manual_review" in errors
    assert "sentence_number_conflict" not in errors



def test_conflicting_lower_number_in_operative_part_is_rejected():
    sentence, appeal = valid_pair_texts()
    appeal = appeal.replace(" по делу 1-1/2024", "")
    appeal = appeal.replace(
        "ОПРЕДЕЛИЛ: Приговор",
        "ОПРЕДЕЛИЛ: Приговор по делу 1-11/2024",
    )
    errors = mod.verify_pair(synthetic_candidate(), sentence, appeal)
    assert "sentence_number_conflict" in errors


def test_ocr_spaced_sentence_change_takes_priority_over_unchanged_remainder():
    text = (
        "ОПРЕДЕЛИЛ: п риговор районного суда изменить. "
        "В остальной части этот же приговор оставить без изменения."
    )
    assert mod.extract_outcome(text) == "changed"


def test_return_of_presentation_without_consideration_is_not_sentence_outcome():
    text = (
        "ОПРЕДЕЛИЛ: определение о возвращении без рассмотрения апелляционного представления "
        "оставить без изменения. Приговор районного суда оставить без изменения."
    )
    assert mod.extract_outcome(text) == "unknown"


def test_new_court_hearing_and_proceedings_are_not_confirmed_pairs():
    sentence, appeal = valid_pair_texts()
    for disposition in (
        "Приговор отменить, дело передать на новое судебное рассмотрение.",
        "Приговор изменить, дело в части ареста передать на новое судебное разбирательство.",
    ):
        changed = appeal.replace("Приговор оставить без изменения,", disposition)
        assert mod.new_trial_disposition(changed)
        assert "new_trial" in mod.verify_pair(synthetic_candidate(), sentence, changed)
