import io
import json

import pytest
from urllib.error import HTTPError

from tools import regional_courts as mod


ROBOTS = """User-agent: *
Disallow: /admin.php
Disallow: /*&cl=1
Disallow: /*&nc=1
Disallow: /*name=sud_delo
Disallow: /*name=bsr
"""


def test_robots_blocks_required_paths_without_fetch():
    rules = mod.parse_robots(ROBOTS)
    assert not mod.required_paths_allowed(mod.KRK, rules)
    assert not mod.allowed_by_robots(mod.KRK.form_url, rules)
    assert mod.allowed_by_robots(mod.KRK.robots_url, rules)


def test_other_rule_does_not_block_court():
    assert mod.required_paths_allowed(mod.VRN, ["/*name=other"])


def test_listing_parser_regional_links_and_total():
    html = """
    <table><tr><td><a href='/modules.php?name=sud_delo&name_op=case&case_id=101'>22-1/2025</a></td>
    <td><a href='/modules.php?name=sud_delo&name_op=doc&number=501'>акт</a></td></tr></table>
    Всего по запросу найдено — 42.
    """
    rows = mod.parse_listing(html, mod.KRK.base)
    assert len(rows) == 1
    assert rows[0]["case_url"].endswith("case_id=101")
    assert rows[0]["document_urls"][0].endswith("number=501")
    assert mod.total_found(html) == 42


def test_regional_number_identity_has_boundaries_and_zero_padding():
    assert mod.canonical_number("01-0001/2025") == ("01", 1, 2025)
    assert mod.canonical_number("1-1/25") == ("1", 1, 2025)
    assert mod.numbers_in_text("1-1/2024") != mod.numbers_in_text("1-11/2024")
    assert ("01", 1, 2024) in mod.numbers_in_text("01 - 0001 / 2024")
    assert mod.canonical_number("1-39/2025") in mod.numbers_in_text("1-214/2023 1-39/2025 1-96/2024")


def test_candidate_uses_act_type_not_only_number():
    assert mod.sentence_candidate("ОБВИНИТЕЛЬНЫЙ приговор ИЗМЕНЕН")
    assert not mod.sentence_candidate("материал о продлении срока содержания под стражей, приговор не вынесен")
    assert not mod.sentence_candidate("постановление в порядке исполнения приговора")


def test_event_identity_does_not_deduplicate_only_uid():
    url = mod.KRK.base + "/modules.php?name=sud_delo&name_op=case&case_id=101"
    text = "24RS0001-01-2025-000101-01 Дата рассмотрения 05.02.2026 АПЕЛЛЯЦИОННОЕ ОПРЕДЕЛЕНИЕ"
    first = mod.event_identity("krk", url, text)
    second = mod.event_identity("krk", url.replace("101", "102"), text)
    assert first["uid"] == second["uid"]
    assert mod.event_key(first) != mod.event_key(second)


def test_first_link_requires_uid_court_number_and_rejects_ambiguity():
    appeal = {"uid": "U1", "lower_court": "Суд А", "lower_number": "1-161/2025"}
    first = {"uid": "U1", "court": "Суд А", "number": "1-161/2025"}
    assert mod.link_first(appeal, [first]) == ("found", first)
    assert mod.link_first(appeal, [first, dict(first)]) == ("ambiguous", None)
    assert mod.link_first(appeal, [dict(first, number="1-162/2025")]) == ("not_found", None)


def valid_texts():
    sentence = (
        "ПРИГОВОР по делу 1-161/2025 от 27.10.2025. Суд ПРИГОВОРИЛ. Синтетический участник. "
        + "мотивировка " * 150
    )
    appeal = (
        "АПЕЛЛЯЦИОННОЕ ОПРЕДЕЛЕНИЕ 22-1/2026 от 05.02.2026. Рассмотрена жалоба на приговор от 27.10.2025. "
        "ОПРЕДЕЛИЛ: Приговор оставить без изменения, Синтетический участник. " + "доводы " * 150
    )
    return sentence, appeal


def metadata():
    return {
        "lower_number": "1-161/2025",
        "sentence_date": "27.10.2025",
        "act_date": "05.02.2026",
        "participants": {"Синтетический"},
    }


def test_pair_verification_regional_full_texts_and_outcome():
    sentence, appeal = valid_texts()
    assert mod.verify_pair(metadata(), sentence, appeal) == []
    assert mod.extract_outcome(appeal) == "unchanged"
    assert "sentence_date_mismatch" in mod.verify_pair(metadata(), sentence.replace("27.10.2025", "28.10.2025"), appeal)
    assert "sentence_bad" in mod.verify_pair(metadata(), sentence.replace("1-161/2025", "1-162/2025"), appeal)


def test_resolution_marker_is_required_and_spaced_form_works():
    assert mod.extract_outcome("защитник просит приговор отменить") == "unknown"
    text = "защитник просит приговор отменить. П О С Т А Н О В И Л: жалобу оставить без удовлетворения."
    assert mod.extract_outcome(text) == "unknown"
    changed = "ОПРЕДЕЛИЛ: Приговор в части назначенного срока изменить, " + "доводы " * 100
    assert mod.extract_outcome(changed) == "changed"


class FakeResponse:
    def __init__(self, url, payload, headers=None):
        self.url = url
        self.payload = io.BytesIO(payload)
        self.headers = headers or {"Content-Length": str(len(payload))}

    def read(self, size=-1):
        return self.payload.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeOpener:
    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append(request.full_url)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            payload, headers = item
            return FakeResponse(request.full_url, payload, headers)
        return FakeResponse(request.full_url, item)


def test_safe_get_checks_length_and_captcha():
    with pytest.raises(mod.DownloadError, match="Неполный"):
        mod.safe_get(mod.KRK.robots_url, mod.KRK, (), pause=0, opener=FakeOpener([(b"short", {"Content-Length": "100"})]))
    with pytest.raises(mod.CaptchaError):
        mod.safe_get(mod.KRK.robots_url, mod.KRK, (), pause=0, opener=FakeOpener([b"SmartCaptcha"]))


def test_http_error_captcha_is_distinct_from_404():
    error = HTTPError(mod.KRK.form_url, 403, "forbidden", {}, io.BytesIO(b"SmartCaptcha"))
    with pytest.raises(mod.CaptchaError):
        mod.safe_get(mod.KRK.form_url, mod.KRK, (), pause=0, opener=FakeOpener([error]))
    error_404 = HTTPError(mod.KRK.form_url, 404, "missing", {}, io.BytesIO(b"not found"))
    with pytest.raises(mod.DownloadError):
        mod.safe_get(mod.KRK.form_url, mod.KRK, (), pause=0, opener=FakeOpener([error_404]))


def test_checkpoint_resume_and_identity(tmp_path):
    identity = {"court": "krk", "production_id": "101", "act_date": "05.02.2026", "act_type": "Апелляционное определение", "uid": "U1"}
    checkpoint = mod.Checkpoint(tmp_path / "checkpoint.json")
    checkpoint.add(identity)
    assert len(checkpoint.pending()) == 1
    checkpoint.finish(identity, "confirmed_pair", "unchanged")
    assert checkpoint.pending() == []
    resumed = mod.Checkpoint(tmp_path / "checkpoint.json")
    assert resumed.data["candidates"]["krk|101|05.02.2026|Апелляционное определение"]["outcome"] == "unchanged"


def test_error_series_stops_after_three(tmp_path):
    opener = FakeOpener([HTTPError(mod.KRK.form_url, 500, "error", {}, io.BytesIO())] * 3)
    result = mod.collect_court(mod.KRK, [], tmp_path, pause=0, opener=opener)
    assert result["status"] == "network_error_series"
    assert len(opener.calls) == 3


def test_report_contains_only_aggregates_and_block_reason(tmp_path):
    path = tmp_path / "report.md"
    access = {
        "krk": {"status": "robots_blocked"},
        "vrn": {"status": "robots_blocked"},
    }
    mod.write_report(access, path, historical={"krk": 2, "vrn": 2})
    text = path.read_text(encoding="utf-8")
    assert "robots_blocked" in text
    assert "не определён" in text
    assert "name_op=case" not in text
    assert "77RS" not in text
