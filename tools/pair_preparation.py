from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

from tools import build_training_dataset as dataset
from tools import mosgorsud_open_data as official

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "corpus/regional/raw/collection-2026-11-10/unified-case-index.jsonl"
DEFAULT_OUTPUT = ROOT / "corpus/pairs/moscow"
TARGET_SOURCES = ("moscow_open_data", "mosgorsud", "expansion", "sudact", "recovery", "processed")
OFFICIAL_POOLS = ("mosgorsud", "expansion")
STATUS_RANK = {
    "confirmed_pair": 0,
    "duplicate": 1,
    "identity_failed": 2,
    "integrity_mismatch": 3,
    "new_trial": 4,
    "procedural_act": 5,
    "download_error": 6,
    "missing_appeal": 7,
    "no_compatible_attachment": 8,
    "missing_sentence": 9,
}


class AccessDenied(RuntimeError):
    pass


class DownloadFailed(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_bytes(path, payload.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def short_hash(value: str, prefix: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def project_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes project root")
    return path


def relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def canonical_date(value: Any) -> str:
    text = re.sub(r"\s*([./-])\s*", r"\1", str(value or "").strip())
    parsed = official.parse_date(text)
    return parsed.date().isoformat() if parsed else dataset.normalized_identity_part(value)


def canonical_court(value: Any) -> str:
    return dataset.normalized_identity_part(value)


def canonical_number(value: Any, year: Any = None) -> str:
    return dataset.canonical_case_number(value, year)


def index_identity(entry: dict[str, Any]) -> dict[str, str]:
    identity = entry["identity"]
    return {
        "uid": "|".join(sorted(identity.get("judicial_uids") or [])),
        "court": canonical_court(identity.get("first_court")),
        "number": canonical_number(identity.get("first_case_number"), identity.get("first_decision_date")),
        "date": canonical_date(identity.get("first_decision_date")),
        "type": "trial_verdict",
    }


def canonical_verdict_group_id(sentence_text: str, identity: dict[str, str]) -> str:
    normalized = dataset.normalized_verdict_text(sentence_text)
    text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    canonical = "|".join(identity.get(key, "") for key in ("court", "number", "date", "type"))
    return short_hash(text_hash + "|" + canonical, "vg_")


def candidate_event_key(candidate: dict[str, Any], appeal_text_sha: str = "") -> str:
    identity = candidate.get("identity") or {}
    return "|".join((
        canonical_court(identity.get("courtName")),
        canonical_number(identity.get("number"), identity.get("year")),
        canonical_date(identity.get("act_date")),
        dataset.normalized_identity_part(identity.get("documentType")),
        appeal_text_sha,
    ))


def candidate_case_id(verdict_group_id: str, candidate: dict[str, Any], appeal_text_sha: str = "") -> str:
    return short_hash(verdict_group_id + "|" + candidate_event_key(candidate, appeal_text_sha), "case_")


def index_artifact(entry: dict[str, Any]) -> dict[str, Any]:
    identity = entry["identity"]
    return {
        "text_path": identity.get("text_path"),
        "text_sha256": identity.get("text_sha256"),
        "raw_path": identity.get("raw_path"),
        "raw_sha256": identity.get("raw_sha256"),
        "text_source_sha256": identity.get("raw_sha256") if identity.get("raw_path") else None,
    }


def document_artifact(root: Path, pool: str, document: dict[str, Any]) -> dict[str, Any]:
    base = root / "corpus" / pool / "raw" / "texts"
    return {
        "text_path": relative_path(root, base / document["text_path"]) if document.get("text_path") else None,
        "text_sha256": document.get("text_sha256"),
        "raw_path": relative_path(root, base / document["path"]) if document.get("path") else None,
        "raw_sha256": document.get("sha256"),
        "text_source_sha256": document.get("text_source_sha256"),
    }


def inspect_artifact(root: Path, spec: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None, list[str]]:
    if not spec or not spec.get("text_path"):
        return "", None, []
    try:
        text_path = project_path(root, spec["text_path"])
        raw_path = project_path(root, spec["raw_path"]) if spec.get("raw_path") else None
    except (ValueError, KeyError):
        return "", None, ["integrity_mismatch"]
    errors = []
    if not text_path.is_file() or not spec.get("text_sha256") or sha256_file(text_path) != spec["text_sha256"]:
        errors.append("integrity_mismatch")
    if spec.get("raw_sha256") and (not raw_path or not raw_path.is_file() or sha256_file(raw_path) != spec["raw_sha256"]):
        errors.append("integrity_mismatch")
    if spec.get("raw_path") and not spec.get("raw_sha256"):
        errors.append("integrity_mismatch")
    if spec.get("text_source_sha256") and spec.get("raw_sha256") != spec["text_source_sha256"]:
        errors.append("integrity_mismatch")
    if errors:
        return "", None, sorted(set(errors))
    try:
        text = text_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return "", None, ["integrity_mismatch"]
    artifact = {"text_path": relative_path(root, text_path), "text_sha256": spec["text_sha256"]}
    if raw_path:
        artifact.update({"raw_path": relative_path(root, raw_path), "raw_sha256": spec["raw_sha256"]})
    return text, artifact, []


def load_index(path: Path, sources: Iterable[str] = TARGET_SOURCES) -> list[list[dict[str, Any]]]:
    selected = set(sources)
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for entry in read_jsonl(path):
        if entry.get("identity", {}).get("source") in selected:
            groups[str(entry["underlying_case_id"])].append(entry)
    return [groups[key] for key in sorted(groups)]


def split_group(group: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sentences = [entry for entry in group if entry["identity"].get("document_type") == "trial_verdict"]
    appeals = [entry for entry in group if entry["identity"].get("document_type") == "appeal_against_verdict"]
    return sentences, appeals


def confirmed_candidate_aliases(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    if not candidate.get("_official"):
        return []
    identity = candidate.get("identity") or {}
    identity_date = official.parse_date(identity.get("act_date"))
    identity_type = official.normalize_court(identity.get("documentType"))
    aliases = []
    for alias in candidate.get("aliases") or []:
        alias_date = official.parse_date(alias.get("date") or alias.get("dateFinal"))
        alias_type = official.normalize_court(alias.get("documentType"))
        if identity_date and alias_date == identity_date and identity_type and alias_type == identity_type:
            aliases.append(alias)
    return aliases


def load_official_candidates(root: Path) -> list[dict[str, Any]]:
    candidates = []
    for pool in OFFICIAL_POOLS:
        raw = root / "corpus" / pool / "raw"
        progress_path = raw / "pair-progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
        for original in read_jsonl(raw / "work" / "appeal-candidates.jsonl"):
            candidate = dict(original)
            candidate["_pool"] = pool
            candidate["_official"] = True
            candidate["_state"] = progress.get(str(candidate["appeal_id"]), {})
            candidate["aliases"] = confirmed_candidate_aliases(candidate)
            candidates.append(candidate)
    return candidates


def parse_russian_date(value: str) -> str | None:
    months = "|".join(official.MONTHS)
    match = re.search(rf"(\d{{1,2}})\s+({months})\s+(\d{{4}})", value.lower())
    return f"{match.group(1)}.{official.MONTHS.index(match.group(2)) + 1:02d}.{match.group(3)}" if match else None


def sudact_candidate(metadata: dict[str, Any]) -> dict[str, Any] | None:
    title = metadata.get("title") or ""
    numbers = re.findall(r"№\s*(\d+(?:/\d+)?\s*-\s*\d+\s*/\s*\d{2,4})", title)
    if not numbers:
        return None
    appeal_number = numbers[0]
    lower_number = numbers[-1]
    date = parse_russian_date(title)
    court = re.sub(r"\s*\([^)]*\)\s*", " ", metadata.get("court") or "")
    court = re.sub(r"\s*-\s*Уголовное\s*$", "", court, flags=re.I).strip()
    return {
        "appeal_id": "sudact:" + str(metadata.get("id") or ""),
        "identity": {
            "number": appeal_number,
            "year": (canonical_date(date) or "")[:4],
            "courtName": court,
            "act_date": date,
            "documentType": "Апелляционное постановление",
        },
        "aliases": [],
        "lower": [{"number": lower_number, "courtName": ""}],
        "first": {"number": lower_number, "dateFinal": None},
        "_pool": "sudact",
        "_official": False,
        "_regional": True,
        "_state": {},
        "_sudact_metadata": metadata,
    }


def load_sudact_candidates(root: Path) -> list[dict[str, Any]]:
    candidates = []
    for metadata in read_jsonl(root / "corpus/sudact/appeals.jsonl"):
        candidate = sudact_candidate(metadata)
        if candidate:
            candidates.append(candidate)
    return candidates


def candidate_uid(candidate: dict[str, Any]) -> str:
    first = candidate.get("first") or {}
    return str(first.get("uid") or (candidate.get("identity") or {}).get("uid") or "")


def candidate_matches(entry: dict[str, Any], candidate: dict[str, Any]) -> bool:
    identity = entry["identity"]
    first = candidate.get("first") or {}
    left = canonical_number(identity.get("first_case_number"), identity.get("first_decision_date"))
    right = canonical_number(first.get("number"), first.get("year"))
    if candidate.get("_regional"):
        return bool(left and right and left == right)
    uids = set(identity.get("judicial_uids") or [])
    uid = candidate_uid(candidate)
    if uids or uid:
        return bool(uids and uid and uid in uids)
    if not left or not right or left != right:
        return False
    return canonical_court(identity.get("first_court")) == canonical_court(first.get("courtName"))


def candidate_sort_key(entry: dict[str, Any], candidate: dict[str, Any]) -> tuple[Any, ...]:
    state = candidate.get("_state") or {}
    documents = state.get("documents") or {}
    return (
        not bool((documents.get("appeal") or {}).get("text_path")),
        candidate.get("_pool") != entry["identity"].get("source"),
        str(candidate.get("_pool")),
        str(candidate.get("appeal_id")),
    )


def matching_candidates(entry: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matched = [candidate for candidate in candidates if candidate_matches(entry, candidate)]
    events: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for candidate in matched:
        events[candidate_event_key(candidate)].append(candidate)
    return [sorted(group, key=lambda item: candidate_sort_key(entry, item))[0] for _, group in sorted(events.items())]


def merged_candidate(entry: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = dict(candidate)
    identity = entry["identity"]
    first = dict(candidate.get("first") or {})
    for key, value in (
        ("number", identity.get("first_case_number")),
        ("courtName", identity.get("first_court")),
        ("dateFinal", identity.get("first_decision_date")),
        ("lastHearingDate", identity.get("first_decision_date")),
        ("year", (canonical_date(identity.get("first_decision_date")) or "")[:4]),
    ):
        if not first.get(key):
            first[key] = value
    result["first"] = first
    if not result.get("lower"):
        result["lower"] = [{"number": first.get("number"), "courtName": first.get("courtName")}]
    return result


def essential_court_stems(value: Any) -> list[str]:
    text = re.sub(r"\([^)]*\)", " ", str(value or ""))
    ignored = {"города", "город", "области", "область", "края", "республики", "суд", "уголовное", "дела", "г"}
    return [
        token[:5] if len(token) > 5 else token
        for token in official.normalize_court(text).split()
        if token not in ignored and not token.isdigit()
    ]


def regional_court_match(text: str, court: Any) -> bool:
    compact = official.compact_text(text)
    stems = essential_court_stems(court)
    return bool(stems) and all(stem in compact for stem in stems)


def verify_regional_pair(candidate: dict[str, Any], sentence_text: str, appeal_text: str) -> list[str]:
    errors = []
    if len(sentence_text) < 1000 or len(appeal_text) < 1000:
        return ["text_too_short"]
    sentence_compact = official.compact_text(sentence_text)
    appeal_compact = official.compact_text(appeal_text)
    if "приговор" not in sentence_compact or "приговорил" not in sentence_compact:
        errors.append("sentence_type_not_confirmed")
    if "установил" not in sentence_compact:
        errors.append("sentence_full_text_not_confirmed")
    if "апелляцион" not in appeal_compact or "наприговор" not in appeal_compact or official.extract_outcome(appeal_text) == "unknown":
        errors.append("appeal_type_not_confirmed")
    first = candidate["first"]
    expected = official.number_identity_variants(first.get("number"), first.get("year"))
    if not expected or not expected & official.extract_case_numbers(sentence_text, first.get("year")):
        errors.append("sentence_number_mismatch")
    references = official.appealed_number_references(appeal_text, first.get("year"))
    if references and not references & expected:
        errors.append("sentence_number_conflict" if len(references) == 1 else "manual_review")
    identity = candidate["identity"]
    if not official.document_number_match(appeal_text, identity.get("number"), identity.get("year")):
        errors.append("appeal_number_mismatch")
    lower_court = (candidate.get("lower") or [{}])[0].get("courtName") or first.get("courtName")
    if not regional_court_match(sentence_text, lower_court) or not regional_court_match(appeal_text, lower_court):
        errors.append("lower_court_mismatch")
    if not regional_court_match(appeal_text, identity.get("courtName")):
        errors.append("appeal_court_mismatch")
    date = first.get("dateFinal") or first.get("lastHearingDate")
    if not official.parse_date(date):
        errors.append("sentence_date_missing")
    elif not all(any(official.compact_text(form) in compact for form in official.date_forms(date)) for compact in (sentence_compact, appeal_compact)):
        errors.append("sentence_date_mismatch")
    appeal_date = identity.get("act_date")
    if not official.parse_date(appeal_date):
        errors.append("appeal_date_missing")
    elif not any(official.compact_text(form) in appeal_compact for form in official.date_forms(appeal_date)):
        errors.append("appeal_date_mismatch")
    return errors


def verify_candidate_pair(candidate: dict[str, Any], sentence_text: str, appeal_text: str) -> list[str]:
    if candidate.get("_regional"):
        return verify_regional_pair(candidate, sentence_text, appeal_text)
    return official.verify_pair(candidate, sentence_text, appeal_text)


def new_trial(appeal_text: str, candidate: dict[str, Any], sentence_text: str) -> bool:
    if official.new_trial_disposition(appeal_text):
        return True
    sentence_date = official.parse_date((candidate.get("first") or {}).get("dateFinal"))
    appeal_date = official.parse_date((candidate.get("identity") or {}).get("act_date"))
    return bool(sentence_date and appeal_date and sentence_date > appeal_date)


def procedural_appeal(appeal_text: str) -> bool:
    operative = official.operative_text(appeal_text) or official.compact_text(appeal_text)
    return bool(re.search(r"мер[аы]\s*пресечения|срок\w*\s+обжалован|восстанов\w+\s+срок|ходатайств\w+\s+удовлетвор|возвращени\w*.{0,100}без\s+рассмотрен\w*|без\s+рассмотрен\w*.{0,100}возвращени\w*", operative, re.S))


def allowed_direct_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname in {"mos-gorsud.ru", "data.mos-gorsud.ru"}
        and parsed.path.startswith("/mgs/cases/docs/content/")
        and "/search" not in parsed.path.lower()
    )


def denial_detected(data: bytes, text: str = "") -> bool:
    probe = (data[:8192].decode("utf-8", "ignore") + "\n" + text[:8192]).lower()
    return bool(re.search(r"captcha|доступ\s+запрещ|access\s+denied|проверк\w+\s+что\s+вы\s+человек|too\s+many\s+requests", probe))


def download_artifact(
    root: Path,
    output: Path,
    url: str,
    sleeper: Callable[[float], None] = time.sleep,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    if not allowed_direct_url(url):
        raise DownloadFailed("non-official or non-direct URL")
    request = urllib.request.Request(url, headers={"User-Agent": "judicial-pair-preparation/1.0"})
    try:
        with opener(request, timeout=60) as response:
            final_url = response.geturl()
            if not allowed_direct_url(final_url):
                raise AccessDenied("redirect outside official direct document endpoint")
            data = response.read(50 * 1024 * 1024 + 1)
            if len(data) > 50 * 1024 * 1024:
                raise DownloadFailed("document too large")
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403, 429}:
            raise AccessDenied(str(exc)) from exc
        raise DownloadFailed(str(exc)) from exc
    finally:
        sleeper(4.0)
    if denial_detected(data):
        raise AccessDenied("access denial or CAPTCHA")
    (output / "raw/texts").mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="download-", suffix=".bin", dir=output / "raw/texts")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        text = official.extract_text(temporary)
    except official.UnsupportedFormat as exc:
        raise DownloadFailed(str(exc)) from exc
    finally:
        temporary.unlink(missing_ok=True)
    if denial_detected(data, text):
        raise AccessDenied("access denial or CAPTCHA")
    if len(text) < 1000:
        raise DownloadFailed("extracted text too short")
    raw_sha = hashlib.sha256(data).hexdigest()
    stem = raw_sha
    raw_path = output / "raw/texts" / (stem + ".bin")
    text_path = output / "raw/texts" / (stem + ".txt")
    atomic_bytes(raw_path, data)
    atomic_bytes(text_path, text.encode("utf-8"))
    return {
        "text_path": relative_path(root, text_path),
        "text_sha256": sha256_file(text_path),
        "raw_path": relative_path(root, raw_path),
        "raw_sha256": raw_sha,
        "text_source_sha256": raw_sha,
    }


def sudact_artifact(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate["_sudact_metadata"]
    path = root / "corpus/sudact/appeals" / (str(metadata["id"]) + ".txt")
    return {"text_path": relative_path(root, path), "text_sha256": sha256_file(path)} if path.is_file() else {}


def appeal_artifact(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate.get("_regional"):
        return sudact_artifact(root, candidate)
    document = ((candidate.get("_state") or {}).get("documents") or {}).get("appeal") or {}
    return document_artifact(root, str(candidate["_pool"]), document) if document.get("text_path") else {}


def downloadable_url(candidate: dict[str, Any]) -> str | None:
    if candidate.get("_regional") or (candidate.get("_state") or {}):
        return None
    attachments = candidate.get("appeal_attachments") or []
    if len(attachments) != 1:
        return None
    url = attachments[0].get("link")
    return url if isinstance(url, str) and allowed_direct_url(url) else None


def base_row(entry: dict[str, Any], sentence_text: str, candidate: dict[str, Any] | None = None, appeal_sha: str = "") -> dict[str, Any]:
    identity = index_identity(entry)
    verdict_group_id = canonical_verdict_group_id(sentence_text, identity)
    if candidate:
        case_id = candidate_case_id(verdict_group_id, candidate, appeal_sha)
        date = (candidate.get("identity") or {}).get("act_date")
        document_type = (candidate.get("identity") or {}).get("documentType") or "appeal_against_verdict"
    else:
        case_id = short_hash(str(entry["underlying_case_id"]) + "|missing-appeal", "case_")
        date = identity["date"]
        document_type = "missing_appeal"
    year_text = canonical_date(date)[:4]
    return {
        "schema_version": 1,
        "case_id": case_id,
        "verdict_group_id": verdict_group_id,
        "source": entry["identity"]["source"],
        "source_year": int(year_text) if year_text.isdigit() else None,
        "document_type": document_type,
        "artifacts": {"sentence": None, "appeal": None},
        "pair_status": "not_started",
        "outcome": "unknown",
        "identity_errors": [],
    }


def resume_valid(root: Path, row: dict[str, Any]) -> bool:
    for kind in ("sentence", "appeal"):
        artifact = (row.get("artifacts") or {}).get(kind)
        if not artifact:
            continue
        _, _, errors = inspect_artifact(root, {
            "text_path": artifact.get("text_path"),
            "text_sha256": artifact.get("text_sha256"),
            "raw_path": artifact.get("raw_path"),
            "raw_sha256": artifact.get("raw_sha256"),
            "text_source_sha256": artifact.get("raw_sha256") if artifact.get("raw_path") else None,
        })
        if errors:
            return False
    return True


def candidate_fingerprint(entry: dict[str, Any], candidate: dict[str, Any] | None) -> str:
    value = {"entry": entry, "candidate": candidate, "contract": 6}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def process_candidate(
    root: Path,
    output: Path,
    entry: dict[str, Any],
    candidate: dict[str, Any],
    sentence_text: str,
    sentence_artifact: dict[str, Any],
    download: bool,
    sleeper: Callable[[float], None],
    opener: Callable[..., Any],
) -> dict[str, Any]:
    candidate = merged_candidate(entry, candidate)
    spec = appeal_artifact(root, candidate)
    processing_errors = []
    if not spec and download and downloadable_url(candidate):
        try:
            spec = download_artifact(root, output, str(downloadable_url(candidate)), sleeper, opener)
        except DownloadFailed as exc:
            processing_errors.append(str(exc))
    preliminary_sha = str(spec.get("text_sha256") or "")
    row = base_row(entry, sentence_text, candidate, preliminary_sha)
    row["artifacts"]["sentence"] = sentence_artifact
    if not spec:
        state_status = (candidate.get("_state") or {}).get("status")
        row["pair_status"] = "no_compatible_attachment" if state_status == "no_compatible_attachment" else "missing_appeal"
        if processing_errors:
            row["pair_status"] = "download_error"
            row["processing_errors"] = processing_errors
        return row
    appeal_text, artifact, integrity_errors = inspect_artifact(root, spec)
    row["artifacts"]["appeal"] = artifact
    row["case_id"] = candidate_case_id(row["verdict_group_id"], candidate, str(spec.get("text_sha256") or ""))
    if integrity_errors:
        row["pair_status"] = "integrity_mismatch"
        row["identity_errors"] = integrity_errors
        return row
    candidate_sentence = ((candidate.get("_state") or {}).get("documents") or {}).get("sentence") or {}
    if candidate_sentence.get("text_path") and not candidate.get("_regional"):
        other_text, _, errors = inspect_artifact(root, document_artifact(root, str(candidate["_pool"]), candidate_sentence))
        if errors or dataset.normalized_verdict_text(other_text) != dataset.normalized_verdict_text(sentence_text):
            row["pair_status"] = "integrity_mismatch" if errors else "identity_failed"
            row["identity_errors"] = errors or ["sentence_text_conflict"]
            return row
    errors = verify_candidate_pair(candidate, sentence_text, appeal_text)
    outcome = official.extract_outcome(appeal_text)
    row["outcome"] = outcome
    row["identity_errors"] = errors
    if new_trial(appeal_text, candidate, sentence_text):
        row["pair_status"] = "new_trial"
        row["identity_errors"] = sorted(set(errors + ["new_trial"]))
    elif errors:
        row["pair_status"] = "procedural_act" if outcome == "unknown" and procedural_appeal(appeal_text) else "identity_failed"
    else:
        row["pair_status"] = "confirmed_pair"
    return row


def mark_duplicates(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["case_id"]].append(row)
    result = []
    duplicates = 0
    for _, group in sorted(groups.items(), key=lambda item: item[0]):
        group.sort(key=lambda row: (STATUS_RANK.get(row["pair_status"], 99), row["source"], json.dumps(row, sort_keys=True)))
        keeper = group[0]
        result.append(keeper)
        for duplicate in group[1:]:
            duplicate = dict(duplicate)
            duplicate["pair_status"] = "duplicate"
            duplicate["duplicate_of"] = keeper["case_id"]
            duplicate["outcome"] = "unknown"
            result.append(duplicate)
            duplicates += 1
    return sorted(result, key=lambda row: (row["source"], row["case_id"])), duplicates


def deduplicate_retrieval(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {}
    for row in rows:
        if row.get("pair_status") != "confirmed_pair":
            continue
        selected.setdefault(row["verdict_group_id"], row)
    return [selected[key] for key in sorted(selected)]


def aggregate(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(collections.Counter(tuple(str(row.get(field)) for field in fields) for row in rows).items())}


def table(counter: collections.Counter[Any], headers: tuple[str, ...]) -> list[str]:
    lines = ["| " + " | ".join(headers + ("Count",)) + " |", "| " + " | ".join("---" for _ in range(len(headers) + 1)) + " |"]
    for key, count in sorted(counter.items(), key=lambda item: str(item[0])):
        values = key if isinstance(key, tuple) else (key,)
        lines.append("| " + " | ".join(str(value) for value in values + (count,)) + " |")
    return lines


def render_report(rows: list[dict[str, Any]], candidate_count: int, duplicate_count: int) -> str:
    statuses = collections.Counter(row["pair_status"] for row in rows)
    confirmed = [row for row in rows if row["pair_status"] == "confirmed_pair"]
    identity_failures = statuses["identity_failed"] + statuses["integrity_mismatch"]
    lines = [
        "# Подготовка полных пар",
        "",
        "Отчёт содержит только агрегаты. Старые статусы подтверждения не использовались как доказательство.",
        "",
        f"- Кандидатов в индексе: **{candidate_count}**",
        f"- Событий в статусном файле: **{len(rows)}**",
        f"- Confirmed pairs: **{statuses['confirmed_pair']}**",
        f"- Восстановлено официальных пар вне сводного индекса: **{sum(row.get('discovery') == 'official_pair_recovery' for row in confirmed)}**",
        f"- Missing sentence: **{statuses['missing_sentence']}**",
        f"- Missing appeal: **{statuses['missing_appeal']}**",
        f"- No compatible attachment: **{statuses['no_compatible_attachment']}**",
        f"- Identity failures (включая integrity): **{identity_failures}**",
        f"- Duplicates: **{duplicate_count}**",
        "",
        "## По источнику и статусу",
        "",
        *table(collections.Counter((row["source"], row["pair_status"]) for row in rows), ("Source", "Status")),
        "",
        "## По году и статусу",
        "",
        *table(collections.Counter((row["source_year"], row["pair_status"]) for row in rows), ("Year", "Status")),
        "",
        "## По типу апелляционного акта",
        "",
        *table(collections.Counter(row["document_type"] for row in rows), ("Type",)),
        "",
        "## Outcomes подтверждённых пар",
        "",
        *table(collections.Counter(row["outcome"] for row in confirmed), ("Outcome",)),
        "",
        "Процессуальные акты, направления на новое рассмотрение и сокращённые приговоры без мотивировки не входят в confirmed pairs. Этот pipeline не меняет processed dataset; включение выполняется отдельной пересборкой dataset.",
        "",
    ]
    return "\n".join(lines)


def official_candidate_entry(root: Path, candidate: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any], list[str]]:
    documents = (candidate.get("_state") or {}).get("documents") or {}
    spec = document_artifact(root, str(candidate["_pool"]), documents["sentence"])
    sentence_text, sentence_artifact, errors = inspect_artifact(root, spec)
    first = candidate.get("first") or {}
    lower = (candidate.get("lower") or [{}])[0]
    uid = candidate_uid(candidate)
    identity = {
        "source": candidate["_pool"],
        "document_type": "trial_verdict",
        "judicial_uids": [uid] if uid else [],
        "first_court": first.get("courtName") or lower.get("courtName"),
        "first_case_number": first.get("number") or lower.get("number"),
        "first_decision_date": first.get("dateFinal") or first.get("lastHearingDate"),
    }
    entry = {"underlying_case_id": "official:" + str(candidate["appeal_id"]), "identity": identity}
    return entry, sentence_text, sentence_artifact, errors


def prepare(
    index_path: Path = DEFAULT_INDEX,
    output: Path = DEFAULT_OUTPUT,
    root: Path = ROOT,
    sources: Iterable[str] = TARGET_SOURCES,
    download: bool = False,
    sleeper: Callable[[float], None] = time.sleep,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> list[dict[str, Any]]:
    groups = load_index(index_path, sources)
    candidates = load_official_candidates(root) + load_sudact_candidates(root)
    progress_path = output / "raw/pair-progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"schema_version": 1, "entries": {}}
    entries = progress.setdefault("entries", {})
    resumed = {}
    for saved in entries.values():
        fingerprint = saved.get("fingerprint")
        if not fingerprint:
            continue
        current = resumed.get(fingerprint)
        rank = STATUS_RANK.get(saved["row"].get("pair_status"), 99)
        current_rank = STATUS_RANK.get(current["row"].get("pair_status"), 99) if current else 99
        if current is None or rank < current_rank:
            resumed[fingerprint] = saved
    rows = []
    for group in groups:
        sentences, index_appeals = split_group(group)
        representative = sentences[0] if sentences else group[0]
        matched = matching_candidates(representative, candidates)
        sentence_text = ""
        sentence_artifact = None
        sentence_errors = []
        if len(sentences) == 1:
            sentence_text, sentence_artifact, sentence_errors = inspect_artifact(root, index_artifact(sentences[0]))
        elif len(sentences) > 1:
            sentence_errors = ["ambiguous_sentence"]
        if not matched and index_appeals:
            sentence_errors.append("index_appeal_requires_source_adapter")
        selected = matched or [None]
        for candidate in selected:
            fingerprint = candidate_fingerprint(representative, candidate)
            saved = resumed.get(fingerprint)
            retry_download = bool(
                download and candidate and downloadable_url(candidate)
                and saved and saved["row"].get("pair_status") in {"missing_appeal", "download_error"}
            )
            if saved and not retry_download and resume_valid(root, saved["row"]):
                rows.append(saved["row"])
                continue
            if not sentences:
                row = base_row(representative, "", candidate)
                row["pair_status"] = "missing_sentence"
            elif sentence_errors:
                row = base_row(representative, sentence_text, candidate)
                row["artifacts"]["sentence"] = sentence_artifact
                row["pair_status"] = "integrity_mismatch" if "integrity_mismatch" in sentence_errors else "identity_failed"
                row["identity_errors"] = sentence_errors
            elif candidate is None:
                row = base_row(representative, sentence_text)
                row["artifacts"]["sentence"] = sentence_artifact
                row["pair_status"] = "missing_appeal"
            else:
                row = process_candidate(root, output, representative, candidate, sentence_text, sentence_artifact, download, sleeper, opener)
            entries[row["case_id"]] = {"fingerprint": fingerprint, "row": row}
            resumed[fingerprint] = entries[row["case_id"]]
            atomic_json(progress_path, progress)
            rows.append(row)
    indexed_hashes = {
        (row["artifacts"]["sentence"]["text_sha256"], row["artifacts"]["appeal"]["text_sha256"])
        for row in rows
        if (row["artifacts"].get("sentence") or {}).get("text_sha256")
        and (row["artifacts"].get("appeal") or {}).get("text_sha256")
    }
    for candidate in candidates:
        documents = (candidate.get("_state") or {}).get("documents") or {}
        if not all((documents.get(kind) or {}).get("text_path") for kind in ("sentence", "appeal")):
            continue
        appeal_spec = document_artifact(root, str(candidate["_pool"]), documents["appeal"])
        sentence_spec = document_artifact(root, str(candidate["_pool"]), documents["sentence"])
        event_hashes = (sentence_spec.get("text_sha256"), appeal_spec.get("text_sha256"))
        if event_hashes in indexed_hashes:
            continue
        entry, sentence_text, sentence_artifact, errors = official_candidate_entry(root, candidate)
        if errors or not sentence_artifact:
            continue
        fingerprint = candidate_fingerprint(entry, candidate)
        saved = resumed.get(fingerprint)
        if saved and resume_valid(root, saved["row"]):
            row = saved["row"]
        else:
            row = process_candidate(root, output, entry, candidate, sentence_text, sentence_artifact, False, sleeper, opener)
        if row["pair_status"] != "confirmed_pair":
            continue
        row["discovery"] = "official_pair_recovery"
        entries[row["case_id"]] = {"fingerprint": fingerprint, "row": row}
        resumed[fingerprint] = entries[row["case_id"]]
        atomic_json(progress_path, progress)
        rows.append(row)
        indexed_hashes.add(event_hashes)
    rows, duplicate_count = mark_duplicates(rows)
    atomic_jsonl(output / "raw/pair-status.jsonl", rows)
    atomic_bytes(output / "report.md", render_report(rows, len(groups), duplicate_count).encode("utf-8"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and independently verify full sentence/appeal pairs")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--download", action="store_true", help="Fetch missing direct official open-data attachments sequentially")
    args = parser.parse_args()
    rows = prepare(args.index, args.output, args.root, download=args.download)
    counts = collections.Counter(row["pair_status"] for row in rows)
    print(json.dumps({"candidates": len(rows), "statuses": counts}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
