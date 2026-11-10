from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mosgorsud_open_data import (
    extract_outcome,
    number_identity_variants,
    parse_date,
    verify_pair,
)

DEFAULT_INDEX = ROOT / "corpus/regional/raw/collection-2026-11-10/unified-case-index.jsonl"
DEFAULT_RELATED = ROOT / "corpus/regional/raw/collection-2026-11-09"
DEFAULT_OUTPUT = ROOT / "corpus/pairs/voronezh-krasnoyarsk"
SOURCES = ("voronezh", "krasnoyarsk")
SENTENCE_TYPES = {"trial_verdict"}
APPEAL_TYPES = {"appeal_against_verdict", "appellate_verdict"}
PROCEDURAL_TYPES = {"procedural_appeal", "cassation_ruling", "criminal_appeal_other", "manual_review"}
OFFICIAL_HOSTS = {
    "voronezh": "oblsud.vrn.sudrf.ru",
    "krasnoyarsk": "kraevoy.krk.sudrf.ru",
}
COURT_FILLERS = {"г", "город", "города", "область", "области", "край", "края", "суд", "судебного", "участка"}
UID_RE = re.compile(r"^[0-9A-Z]{2}[0-9A-Z]{2}\d{4}-\d{2}-\d{4}-\d{6}-\d{2}$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL row is not an object: {path}")
                rows.append(payload)
    return rows


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_text(path: Path, text: str) -> None:
    atomic_write(path, text.encode("utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_text(path, serialized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value: Any, root: Path = ROOT, base_dir: Path | None = None) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        base = root if base_dir is None or candidate.parts[0] == "corpus" else base_dir
        resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("artifact path escapes project root") from exc
    return resolved


def relative_path(path: Path, root: Path = ROOT) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def canonical_court(value: Any) -> str:
    tokens = re.findall(r"[0-9a-zа-я]+", str(value or "").lower().replace("ё", "е"))
    return " ".join(token for token in tokens if token not in COURT_FILLERS)


def formal_uids(metadata: dict[str, Any]) -> set[str]:
    values = {metadata.get("judicial_uid")}
    for field in ("judicial_uids_found", "judicial_uids_observed", "judicial_uids"):
        value = metadata.get(field)
        if isinstance(value, list):
            values.update(value)
    return {value for value in values if isinstance(value, str) and UID_RE.fullmatch(value)}


def official_case_keys(case: dict[str, Any]) -> set[tuple[str, tuple[tuple[int, ...], int, int]]]:
    result = set()
    identity = case["identity"]
    values = [(identity.get("first_court"), identity.get("first_case_number"))]
    for key in case.get("case_keys") or []:
        if isinstance(key, str) and key.startswith("first_case:") and "|" in key:
            court, number = key.removeprefix("first_case:").rsplit("|", 1)
            values.append((court, number))
    for court, number in values:
        canonical = canonical_court(court)
        for parsed in number_identity_variants(number):
            if canonical:
                result.add((canonical, parsed))
    return result


def document_case_keys(document: dict[str, Any]) -> set[tuple[str, tuple[tuple[int, ...], int, int]]]:
    metadata = document["metadata"]
    canonical = canonical_court(metadata.get("lower_court"))
    return {(canonical, parsed) for parsed in number_identity_variants(metadata.get("lower_case_number")) if canonical}


def load_cases(index_path: Path = DEFAULT_INDEX) -> list[dict[str, Any]]:
    cases = [row for row in read_jsonl(index_path) if row.get("identity", {}).get("source") in SOURCES]
    identifiers = [row.get("underlying_case_id") for row in cases]
    if len(cases) != 84 or any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("expected 84 identified regional cases")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate underlying_case_id in unified index")
    return sorted(cases, key=lambda row: (row["identity"]["source"], row["underlying_case_id"]))


def load_documents(related_dir: Path = DEFAULT_RELATED) -> tuple[list[dict[str, Any]], dict[str, int]]:
    documents: list[dict[str, Any]] = []
    duplicate_ids = 0
    seen_ids: set[tuple[str, str]] = set()
    for source in SOURCES:
        paths = sorted((related_dir / source).glob("integration-records-*.jsonl"))
        if len(paths) != 1:
            raise ValueError(f"expected one integration record file for {source}")
        for row in read_jsonl(paths[0]):
            metadata = row.get("metadata")
            document_id = row.get("document_id") or (metadata or {}).get("document_id")
            if not isinstance(metadata, dict) or not isinstance(document_id, str) or not document_id:
                raise ValueError("regional document is missing metadata or document_id")
            key = (source, document_id)
            if key in seen_ids:
                duplicate_ids += 1
                continue
            seen_ids.add(key)
            documents.append({
                **row,
                "source": source,
                "document_id": document_id,
                "_artifact_base": str((related_dir / source).resolve()),
            })
    text_groups = Counter(
        (row["source"], row.get("normalized_text_sha256") or row.get("text_sha256"))
        for row in documents
        if row.get("normalized_text_sha256") or row.get("text_sha256")
    )
    duplicate_text_groups = sum(count > 1 for count in text_groups.values())
    return documents, {
        "duplicate_document_ids": duplicate_ids,
        "duplicate_text_groups_not_deduplicated": duplicate_text_groups,
    }


def document_matches(case: dict[str, Any], document: dict[str, Any]) -> set[str]:
    identity = case["identity"]
    methods = set()
    if document.get("text_sha256") == identity.get("text_sha256"):
        methods.add("unified_primary_sha256")
    if formal_uids(identity) & formal_uids(document["metadata"]):
        methods.add("official_judicial_uid")
    if official_case_keys(case) & document_case_keys(document):
        methods.add("official_first_court_and_number")
    return methods


def link_documents(
    cases: list[dict[str, Any]], documents: list[dict[str, Any]]
) -> dict[str, list[tuple[dict[str, Any], set[str]]]]:
    links: dict[str, list[tuple[dict[str, Any], set[str]]]] = {
        case["underlying_case_id"]: [] for case in cases
    }
    for case in cases:
        source = case["identity"]["source"]
        for document in documents:
            if document["source"] != source:
                continue
            methods = document_matches(case, document)
            if methods:
                links[case["underlying_case_id"]].append((document, methods))
    document_link_count = Counter(
        (document["source"], document["document_id"])
        for linked in links.values()
        for document, _ in linked
    )
    for case_id, linked in links.items():
        links[case_id] = [
            (document, methods | ({"ambiguous_case_link"} if document_link_count[(document["source"], document["document_id"])] > 1 else set()))
            for document, methods in linked
        ]
    return links


def direct_document_url(source: str, document: dict[str, Any]) -> str | None:
    url = document["metadata"].get("source_url")
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if (
        parsed.scheme != "https"
        or parsed.hostname != OFFICIAL_HOSTS[source]
        or "/search" in parsed.path.lower()
        or query.get("name_op") != ["doc"]
    ):
        return None
    return url


def artifact_for_document(
    document: dict[str, Any], root: Path = ROOT
) -> tuple[dict[str, Any], list[str]]:
    errors = []
    configured_base = document.get("_artifact_base")
    base_dir = Path(configured_base) if isinstance(configured_base, str) else None
    text_path = project_path(document.get("text_path"), root, base_dir)
    raw_path = project_path(document.get("raw_path"), root, base_dir)
    if text_path is None or not text_path.is_file():
        errors.append("text_artifact_missing")
    if raw_path is None or not raw_path.is_file():
        errors.append("raw_artifact_missing")
    text = None
    if text_path is not None and text_path.is_file():
        actual_text_sha = sha256_file(text_path)
        if actual_text_sha != document.get("text_sha256"):
            errors.append("text_sha256_mismatch")
        else:
            text = text_path.read_text(encoding="utf-8")
            if len(text) < 1000:
                errors.append("text_too_short")
            decoded_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if document.get("decoded_text_sha256") not in {None, decoded_sha}:
                errors.append("decoded_text_sha256_mismatch")
    if raw_path is not None and raw_path.is_file():
        if sha256_file(raw_path) != document.get("raw_sha256"):
            errors.append("raw_sha256_mismatch")
    source = document["source"]
    direct_url = direct_document_url(source, document)
    if direct_url is None:
        errors.append("official_direct_url_missing_or_invalid")
    artifact = {
        "document_id": document["document_id"],
        "path": relative_path(text_path, root) if text_path else None,
        "sha256": document.get("text_sha256"),
        "raw_path": relative_path(raw_path, root) if raw_path else None,
        "raw_sha256": document.get("raw_sha256"),
        "direct_url": direct_url,
    }
    return (artifact, text), errors


def candidate_metadata(case: dict[str, Any], appeal: dict[str, Any]) -> dict[str, Any]:
    identity = case["identity"]
    appeal_metadata = appeal["metadata"]
    first_date = parse_date(identity.get("first_decision_date"))
    appeal_date = parse_date(appeal_metadata.get("decision_date"))
    return {
        "first": {
            "number": identity.get("first_case_number"),
            "year": first_date.year if first_date else None,
            "dateFinal": identity.get("first_decision_date"),
            "lastHearingDate": identity.get("first_decision_date"),
        },
        "identity": {
            "number": appeal_metadata.get("case_number"),
            "year": appeal_date.year if appeal_date else None,
            "act_date": appeal_metadata.get("decision_date"),
            "documentType": appeal.get("document_type"),
            "courtName": appeal_metadata.get("court"),
        },
        "lower": [{"courtName": identity.get("first_court") or appeal_metadata.get("lower_court")}],
        "aliases": [],
    }


def verify_regional_pair(
    case: dict[str, Any],
    sentence: dict[str, Any],
    appeal: dict[str, Any],
    sentence_text: str,
    appeal_text: str,
) -> list[str]:
    identity = case["identity"]
    errors = verify_pair(candidate_metadata(case, appeal), sentence_text, appeal_text)
    uid_confirmed = bool(
        formal_uids(identity)
        & formal_uids(sentence["metadata"])
        & formal_uids(appeal["metadata"])
    )
    if uid_confirmed and not identity.get("first_case_number"):
        errors = [
            error
            for error in errors
            if error not in {"sentence_number_mismatch", "sentence_number_conflict"}
        ]
    if uid_confirmed and not (identity.get("first_court") or appeal["metadata"].get("lower_court")):
        errors = [error for error in errors if error != "lower_court_mismatch"]
    if sentence.get("document_type") not in SENTENCE_TYPES:
        errors.append("sentence_event_type_rejected")
    if appeal.get("document_type") not in APPEAL_TYPES:
        errors.append("appeal_event_type_rejected")
    if "ambiguous_case_link" in sentence.get("match_methods", set()) or "ambiguous_case_link" in appeal.get("match_methods", set()):
        errors.append("ambiguous_case_link")
    sentence_date = parse_date(identity.get("first_decision_date"))
    appeal_date = parse_date(appeal["metadata"].get("decision_date"))
    if sentence_date and appeal_date and sentence_date > appeal_date:
        errors.append("appeal_precedes_sentence")
    compact_appeal = re.sub(r"[^0-9a-zа-я]+", "", appeal_text.lower().replace("ё", "е"))
    if "приговорновогорассмотрения" in compact_appeal or (
        "новогорассмотрения" in compact_appeal and "апелляцион" not in compact_appeal
    ):
        errors.append("new_trial_not_accepted")
    return list(dict.fromkeys(errors))


def pair_id(case_id: str, sentence_id: str | None, appeal_id: str | None, event_id: str | None = None) -> str:
    value = "|".join((case_id, sentence_id or "", appeal_id or "", event_id or ""))
    return "pair_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def base_status(
    case: dict[str, Any],
    status: str,
    sentence: dict[str, Any] | None,
    appeal: dict[str, Any] | None,
    errors: list[str],
    outcome: str = "unknown",
    excluded: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sentence_id = sentence["document_id"] if sentence else None
    appeal_id = appeal["document_id"] if appeal else None
    event_id = excluded["document_id"] if excluded else None
    return {
        "schema_version": 1,
        "pair_id": pair_id(case["underlying_case_id"], sentence_id, appeal_id, event_id),
        "verdict_group_id": case["underlying_case_id"],
        "source": case["identity"]["source"],
        "pair_status": status,
        "outcome": outcome,
        "identity_errors": errors,
        "sentence_artifact": sentence.get("artifact") if sentence else None,
        "appeal_artifact": appeal.get("artifact") if appeal else None,
        "excluded_artifact": excluded.get("artifact") if excluded else None,
    }


def prepare_pairs(
    cases: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    duplicate_stats: dict[str, int] | None = None,
    root: Path = ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    links = link_documents(cases, documents)
    statuses: list[dict[str, Any]] = []
    for case in cases:
        linked = links[case["underlying_case_id"]]
        if not linked:
            statuses.append(base_status(case, "identity_failed", None, None, ["primary_document_not_linked"]))
            continue
        prepared = []
        for original, methods in linked:
            (artifact, text), errors = artifact_for_document(original, root)
            prepared.append({**original, "match_methods": methods, "artifact": artifact, "text": text, "artifact_errors": errors})
        sentences = [item for item in prepared if item["document_type"] in SENTENCE_TYPES]
        appeals = [item for item in prepared if item["document_type"] in APPEAL_TYPES]
        excluded = [item for item in prepared if item not in sentences and item not in appeals]
        for item in excluded:
            status = "excluded_procedural" if item["document_type"] in PROCEDURAL_TYPES else "excluded_unsupported_event"
            statuses.append(base_status(case, status, None, None, [item["document_type"]], excluded=item))
        for appeal in appeals:
            outcome = extract_outcome(appeal["text"]) if appeal["text"] is not None else "unknown"
            if not sentences:
                statuses.append(base_status(case, "missing_sentence", None, appeal, appeal["artifact_errors"], outcome))
                continue
            compatible = []
            for sentence in sentences:
                errors = sentence["artifact_errors"] + appeal["artifact_errors"]
                if errors:
                    compatible.append((sentence, list(dict.fromkeys(errors)), True))
                    continue
                errors = verify_regional_pair(case, sentence, appeal, sentence["text"], appeal["text"])
                compatible.append((sentence, errors, False))
            valid = [item for item in compatible if not item[1]]
            if len(valid) > 1:
                statuses.append(base_status(case, "identity_failed", None, appeal, ["multiple_valid_sentence_candidates"], outcome))
                continue
            chosen = valid[0] if len(valid) == 1 else compatible[0]
            sentence, errors, artifact_invalid = chosen
            status = "artifact_invalid" if artifact_invalid else ("confirmed_pair" if not errors else "identity_failed")
            statuses.append(base_status(case, status, sentence, appeal, errors, outcome))
            for extra_sentence, extra_errors, extra_invalid in (item for item in compatible if item is not chosen):
                extra_status = "artifact_invalid" if extra_invalid else "identity_failed"
                statuses.append(base_status(case, extra_status, extra_sentence, appeal, extra_errors or ["alternative_sentence_not_selected"], outcome))
        if not appeals:
            for sentence in sentences:
                if sentence["artifact_errors"]:
                    statuses.append(base_status(case, "artifact_invalid", sentence, None, sentence["artifact_errors"]))
                else:
                    statuses.append(base_status(case, "missing_appeal", sentence, None, []))
    statuses.sort(key=lambda row: (row["source"], row["verdict_group_id"], row["pair_id"]))
    summary = summarize(statuses, duplicate_stats or {})
    return statuses, summary


def summarize(rows: list[dict[str, Any]], duplicate_stats: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cases": {},
        "events": dict(sorted(Counter(row["pair_status"] for row in rows).items())),
        "outcomes_confirmed": {outcome: 0 for outcome in ("unchanged", "changed", "cancelled", "unknown")},
        "duplicates": duplicate_stats,
    }
    for source in SOURCES:
        source_rows = [row for row in rows if row["source"] == source]
        confirmed = [row for row in source_rows if row["pair_status"] == "confirmed_pair"]
        result["cases"][source] = {
            "candidates": len({row["verdict_group_id"] for row in source_rows}),
            "statuses": dict(sorted(Counter(row["pair_status"] for row in source_rows).items())),
            "confirmed_pairs": len(confirmed),
            "outcomes": dict(sorted(Counter(row["outcome"] for row in confirmed).items())),
        }
        for row in confirmed:
            result["outcomes_confirmed"][row["outcome"]] = result["outcomes_confirmed"].get(row["outcome"], 0) + 1
    result["candidates"] = sum(item["candidates"] for item in result["cases"].values())
    result["confirmed_pairs"] = sum(item["confirmed_pairs"] for item in result["cases"].values())
    return result


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Воронеж и Красноярск: подготовка полных пар",
        "",
        "Публикуемый отчёт содержит только агрегаты. Реквизиты дел, УИД, ссылки и хеши остаются только в локальном статус-файле. Разные апелляционные события не объединяются по UID.",
        "",
        "Полная пара подтверждается только при двух полных текстах, совпавших SHA, официальной identity по UID либо суду и номеру первой инстанции, датам и типам актов. Процессуальные акты и новый приговор нового рассмотрения не принимаются.",
        "",
        "## Итоги по источникам",
        "",
        "| Источник | Исходные дела | Подтверждённые пары | Отсутствует приговор | Отсутствует апелляция | Identity failures | Некорректные artifacts | Процессуальные исключены |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in SOURCES:
        item = summary["cases"].get(source, {"candidates": 0, "statuses": {}, "confirmed_pairs": 0})
        statuses = item["statuses"]
        lines.append(
            f"| {source} | {item['candidates']} | {item['confirmed_pairs']} | "
            f"{statuses.get('missing_sentence', 0)} | {statuses.get('missing_appeal', 0)} | "
            f"{statuses.get('identity_failed', 0)} | {statuses.get('artifact_invalid', 0)} | "
            f"{statuses.get('excluded_procedural', 0)} |"
        )
    lines.extend(
        [
            f"| **Итого** | **{summary['candidates']}** | **{summary['confirmed_pairs']}** | "
            f"**{summary['events'].get('missing_sentence', 0)}** | **{summary['events'].get('missing_appeal', 0)}** | "
            f"**{summary['events'].get('identity_failed', 0)}** | **{summary['events'].get('artifact_invalid', 0)}** | "
            f"**{summary['events'].get('excluded_procedural', 0)}** |",
            "",
            "Строки отсутствующих сторон относятся к событиям: при нескольких апелляциях по одному делу каждая сохраняется отдельно. Поэтому их сумма может превышать число исходных дел.",
            "",
            "## Дубликаты",
            "",
            f"- Повторные идентификаторы документов, исключённые при загрузке: **{summary['duplicates'].get('duplicate_document_ids', 0)}**.",
            f"- Группы совпадающего текста с разными идентификаторами документов: **{summary['duplicates'].get('duplicate_text_groups_not_deduplicated', 0)}**. Они не удаляются автоматически, поскольку это могут быть разные события.",
            "",
            "## Outcomes только подтверждённых пар",
            "",
            "| Outcome | Количество |",
            "|---|---:|",
        ]
    )
    for outcome in ("unchanged", "changed", "cancelled", "unknown"):
        lines.append(f"| {outcome} | {summary['outcomes_confirmed'].get(outcome, 0)} |")
    lines.extend(
        [
            "",
            "Outcome определяется только по резолютивной части апелляции. Labels у неподтверждённых кандидатов не засчитываются как outcomes полных пар.",
            "",
            "## Ограничения",
            "",
            "Отсутствие второго текста в локальной коллекции не доказывает отсутствие судебного акта на официальном сайте. Никакие поисковые или robots-blocked маршруты не использовались. Результат не включён в processed dataset и требует проверки lead перед дальнейшим использованием.",
            "",
        ]
    )
    return "\n".join(lines)


def render_readme() -> str:
    return """# Подготовка региональных полных пар

Локальный каталог содержит `pair-status.jsonl` с реквизитами artifacts и SHA, агрегированный `report.md` и эту инструкцию. Статус-файл может содержать чувствительные реквизиты и не предназначен для публикации.

## Метод

1. Из `unified-case-index.jsonl` выбираются 81 дело `voronezh` и 3 дела `krasnoyarsk`.
2. Связанные integration records коллекции 2026-11-09 присоединяются по точному официальному judicial UID либо канонической связке суда и номера первой инстанции. UUID производства UID не считается. Алиасы и перенумерации без официального доказательства не создаются.
3. Каждый document ID обрабатывается как отдельное событие. Разные апелляции одного UID не дедуплицируются. Повторяется только тот же document ID; совпадающий текст с другим document ID сохраняется для проверки.
4. До подтверждения проверяются наличие приговора и апелляции, полные тексты не короче 1000 символов, фактические SHA текста и raw-файла, прямой официальный document URL, типы, суд, номера и даты в обоих текстах. Используются строгие `verify_pair` и outcome по последней резолютивной части.
5. Процессуальные акты, кассационные события вне задачи апелляции и новый приговор нового рассмотрения не принимаются. Неизвестный outcome не подтверждает пару.

## Resume

Повторный запуск заново проверяет фактические SHA и перезаписывает статус-файл атомарно. Уже скачанные документы читаются только из исходных коллекций и не копируются. Если artifact отсутствует или повреждён, событие получает `artifact_invalid`; произвольное докачивание не происходит. Разрешённые direct URLs сохраняются в статус-файле для ручного resume lead. В текущем локальном прогоне HTTP-запросы не выполняются, поэтому паузы и CAPTCHA не возникают. Для любого последующего сетевого resume допустимы только последовательные GET по этим прямым document URLs с паузой не менее 4 секунд и немедленной остановкой при CAPTCHA; `/search` и robots-blocked маршруты запрещены.

```bash
python tools/prepare_regional_pairs.py run
python tools/prepare_regional_pairs.py report
```

`run` выполняет полную локальную ревалидацию. `report` заново строит только агрегаты из существующего `pair-status.jsonl`. Наличие статуса `confirmed_pair` не означает включение в processed dataset: сначала требуется проверка lead.
"""


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    cases = load_cases()
    documents, duplicate_stats = load_documents()
    rows, summary = prepare_pairs(cases, documents, duplicate_stats)
    write_jsonl(output_dir / "pair-status.jsonl", rows)
    atomic_text(output_dir / "report.md", render_report(summary))
    atomic_text(output_dir / "README.md", render_readme())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare strict Voronezh/Krasnoyarsk sentence-appeal pairs")
    parser.add_argument("command", choices=("run", "report"), default="run", nargs="?")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "run":
        summary = run(args.output)
    else:
        rows = read_jsonl(args.output / "pair-status.jsonl")
        _, duplicate_stats = load_documents()
        summary = summarize(rows, duplicate_stats)
        atomic_text(args.output / "report.md", render_report(summary))
        atomic_text(args.output / "README.md", render_readme())
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
