from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mosgorsud_open_data import (
    DEFAULT_RAW as MOSCOW_RAW,
    CaptchaError,
    DownloadError,
    UnsupportedFormat,
    act_date,
    attachment_combinations,
    court_matches,
    ensure_document,
    extract_outcome,
    first_index,
    has_article_290,
    iter_zip_records,
    normalize_number,
    parse_date,
    read_candidates,
    read_json,
    revalidate_progress,
    refresh_artifact_flags,
    sentence_attachments,
    status_summary,
    verify_pair,
)
from tools.build_training_dataset import atomic_json, atomic_write, read_jsonl, write_jsonl

RAW = ROOT / "corpus" / "recovery" / "raw"
REPORT = ROOT / "corpus" / "recovery" / "report.md"
EXPANSION_RAW = ROOT / "corpus" / "expansion" / "raw"
ALLOWED_STATUSES = {"verification_failed", "no_compatible_attachment", "download_error", "manual_review"}
PRIORITY = (
    "lost_alias",
    "wrong_attachment",
    "format_mismatch",
    "partial_download",
    "other",
)


def priority(candidate: dict[str, Any], state: dict[str, Any]) -> str:
    if candidate.get("aliases"):
        return "lost_alias"
    errors = set(state.get("errors", []))
    if errors & {"sentence_number_mismatch", "lower_court_mismatch", "appeal_court_mismatch", "appeal_type_not_confirmed", "sentence_type_not_confirmed"}:
        return "wrong_attachment"
    if errors & {"appeal_number_mismatch", "sentence_date_mismatch", "appeal_date_mismatch"}:
        return "format_mismatch"
    if state.get("status") == "download_error" or len(state.get("documents", {})) < 2:
        return "partial_download"
    return "other"


def select_candidates(limit: int = 50) -> list[dict[str, Any]]:
    selected = []
    for raw_dir in (MOSCOW_RAW, EXPANSION_RAW):
        progress = read_json(raw_dir / "pair-progress.json", {})
        candidates = {str(item["appeal_id"]): item for item in read_candidates(raw_dir / "work" / "appeal-candidates.jsonl")}
        for appeal_id, state in progress.items():
            if state.get("status") not in ALLOWED_STATUSES:
                continue
            candidate = candidates.get(appeal_id)
            if not candidate:
                continue
            selected.append({
                "candidate": candidate,
                "state": state,
                "raw_dir": raw_dir,
                "priority": priority(candidate, state),
                "source": "mosgorsud" if raw_dir == MOSCOW_RAW else "expansion",
            })
    selected.sort(key=lambda item: (PRIORITY.index(item["priority"]), item["candidate"]["identity"].get("year") or 0, item["candidate"]["appeal_id"]))
    return selected[:limit]


def backup_first(candidate: dict[str, Any], index: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any] | None:
    uid = candidate["identity"].get("uid")
    if not uid:
        return None
    appeal_date = parse_date(candidate["identity"].get("act_date"))
    current_id = (candidate.get("first") or {}).get("_id")
    matches = []
    for lower in candidate.get("lower") or []:
        for variant in normalize_number(lower.get("number")):
            for first in index.get((uid, variant), []):
                if first.get("_id") == current_id or not court_matches(lower, first) or not sentence_attachments(first):
                    continue
                decision_date = parse_date(first.get("dateFinal") or first.get("lastHearingDate"))
                if appeal_date and decision_date and decision_date <= appeal_date:
                    matches.append(first)
    return matches[0] if len(matches) == 1 else None


def load_backup_index() -> dict[tuple[str, str], list[dict[str, Any]]]:
    return first_index(record for record in iter_zip_records(MOSCOW_RAW / "first-instance-2026-11-08.json.zip") if has_article_290(record))


def recover(raw_dir: Path = RAW, pause: float = 4, limit: int = 50, retry_errors: bool = False) -> dict[str, Any]:
    selected = select_candidates(limit)
    progress_path = raw_dir / "pair-progress.json"
    progress = read_json(progress_path, {})
    index = load_backup_index()
    attempted = 0
    write_jsonl(raw_dir / "work" / "appeal-candidates.jsonl", [item["candidate"] for item in selected])
    for item in selected:
        candidate = item["candidate"]
        source_dir = item["raw_dir"]
        appeal_id = str(candidate["appeal_id"])
        current = progress.get(appeal_id, {"status": "processing", "documents": {}, "attempts": []})
        progress[appeal_id] = current
        attempted += 1
        if current.get("status") == "confirmed_pair":
            continue
        current["status"] = "processing"
        current["source_status"] = item["state"].get("status")
        current["priority"] = item["priority"]
        first = backup_first(candidate, index)
        if first:
            candidate["first"] = first
            candidate["first_attachments"] = sentence_attachments(first)
        current.setdefault("documents", {})
        current.setdefault("attempts", [])
        combinations = attachment_combinations(candidate)
        if not combinations:
            current["status"] = "no_compatible_attachment"
            current["errors"] = ["no_compatible_attachment_combination"]
            atomic_json(progress_path, progress)
            continue
        errors = []
        unsupported = []
        try:
            for sentence_attachment, appeal_attachment in combinations:
                try:
                    _, sentence_text = ensure_document(raw_dir, appeal_id, "sentence", sentence_attachment, current, progress_path, progress, pause)
                    _, appeal_text = ensure_document(raw_dir, appeal_id, "appeal", appeal_attachment, current, progress_path, progress, pause)
                    current["both_files_downloaded"] = True
                    current["both_texts_extracted"] = True
                    errors = verify_pair(candidate, sentence_text, appeal_text)
                    current["outcome"] = extract_outcome(appeal_text)
                    current["attempts"].append({
                        "sentence_url": sentence_attachment["link"], "appeal_url": appeal_attachment["link"],
                        "errors": errors, "outcome": current["outcome"],
                    })
                    current["errors"] = errors
                    if not errors:
                        current["status"] = "confirmed_pair"
                        atomic_json(progress_path, progress)
                        break
                    current["status"] = "manual_review" if errors == ["manual_review"] else "verification_failed"
                    atomic_json(progress_path, progress)
                except UnsupportedFormat as exc:
                    unsupported.append(str(exc))
                    current["attempts"].append({
                        "sentence_url": sentence_attachment["link"], "appeal_url": appeal_attachment["link"],
                        "errors": [str(exc)], "outcome": "unknown",
                    })
                    refresh_artifact_flags(raw_dir, current)
                    atomic_json(progress_path, progress)
            else:
                if errors:
                    current["status"] = "manual_review" if errors == ["manual_review"] else "verification_failed"
                    current["errors"] = errors
                else:
                    current["status"] = "unsupported_format"
                    current["errors"] = unsupported
                refresh_artifact_flags(raw_dir, current)
                atomic_json(progress_path, progress)
        except DownloadError as exc:
            current["status"] = "captcha" if isinstance(exc, CaptchaError) else "download_error"
            current["errors"] = [str(exc)]
            refresh_artifact_flags(raw_dir, current)
            atomic_json(progress_path, progress)
            if isinstance(exc, CaptchaError):
                raise
            if "HTTP 404" not in str(exc):
                break
    write_report(raw_dir)
    return {"attempted": attempted, "statuses": status_summary(progress)}


def write_report(raw_dir: Path = RAW, report_path: Path = REPORT) -> None:
    progress = revalidate_progress(raw_dir)
    selection = select_candidates(50)
    statuses = Counter(item.get("status", "not_started") for item in progress.values())
    outcomes = Counter(item.get("outcome", "unknown") for item in progress.values() if item.get("status") == "confirmed_pair")
    years = Counter()
    for item in selection:
        state = progress.get(str(item["candidate"]["appeal_id"]), {})
        if state.get("status") == "confirmed_pair":
            years[str(item["candidate"]["identity"].get("year") or "other/no-year")] += 1
    reasons = Counter()
    for item in selection:
        state = progress.get(str(item["candidate"]["appeal_id"]), {})
        if state.get("status") != "confirmed_pair":
            for error in state.get("errors", []) or [state.get("status", "not_checked")]:
                reasons[error] += 1
    lines = [
        "# Ревизия отклонённых кандидатов Москвы",
        "",
        f"Просмотрено кандидатов: {len(selection)}. Порядок: потерянные алиасы → не тот attachment → формат номера/даты → частичные загрузки → остальные. Исходное распределение сохранено.",
        "Подтверждение только по двум полным текстам, SHA и текущей identity. Цитата приговора в апелляции полной парой не считается. 207 основных и 138 expansion кандидатов не смешиваются с recovery.",
        "",
        "## Итог",
        "",
        f"- **Recovered full pairs: {statuses.get('confirmed_pair', 0)}**",
        f"- Не восстановлено: {len(selection) - statuses.get('confirmed_pair', 0)}",
        f"- verification_failed: {statuses.get('verification_failed', 0)}",
        f"- no_compatible_attachment: {statuses.get('no_compatible_attachment', 0)}",
        f"- unsupported_format: {statuses.get('unsupported_format', 0)}",
        f"- download_error: {statuses.get('download_error', 0)}",
        f"- manual_review: {statuses.get('manual_review', 0)}",
        f"- captcha: {statuses.get('captcha', 0)}",
        "",
        "## Причины невосстановления",
        "",
        "| Причина | Количество |",
        "|---|---:|",
    ]
    for reason, count in sorted(reasons.items()):
        lines.append(f"| {reason} | {count} |")
    lines += ["", "## Outcome восстановленных пар", "", "| Outcome | Количество |", "|---|---:|"]
    for outcome in ("unchanged", "changed", "cancelled", "unknown"):
        lines.append(f"| {outcome} | {outcomes.get(outcome, 0)} |")
    lines += ["", "## Восстановленные пары по годам", "", "| Год | Количество |", "|---|---:|"]
    for year, count in sorted(years.items()):
        lines.append(f"| {year} | {count} |")
    atomic_write(report_path, "\n".join(lines).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Строгая ревизия отклонённых кандидатов Москвы")
    parser.add_argument("command", choices=("select", "recover", "report"))
    parser.add_argument("--pause", type=float, default=4)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.command == "select":
        selected = select_candidates(args.limit)
        print(json.dumps({"selected": len(selected), "by_priority": dict(Counter(item["priority"] for item in selected))}, ensure_ascii=False))
    elif args.command == "recover":
        print(json.dumps(recover(pause=args.pause, limit=args.limit, retry_errors=args.retry_errors), ensure_ascii=False))
    else:
        write_report()
        print(json.dumps({"report": str(REPORT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
