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
    APPEALS,
    FIRST_INSTANCE,
    act_date,
    appellate_attachments,
    article_290,
    attachment_combinations,
    confirmed_aliases,
    ensure_document,
    extract_outcome,
    first_index,
    has_article_290,
    in_target_period,
    iter_zip_records,
    link_first_instance,
    read_json,
    record_years,
    revalidate_progress,
    refresh_artifact_flags,
    sentence_attachments,
    sentence_number,
    status_summary,
    verify_pair,
)
from tools.build_training_dataset import atomic_json, atomic_write, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "corpus" / "expansion" / "raw"
REPORT = ROOT / "corpus" / "expansion" / "report.md"


def outside_candidates(raw_dir: Path = RAW) -> dict[str, Any]:
    first_records = [record for record in iter_zip_records(MOSCOW_RAW / FIRST_INSTANCE.filename) if has_article_290(record)]
    index = first_index(first_records)
    rows = []
    totals = Counter()
    for record in iter_zip_records(MOSCOW_RAW / APPEALS.filename):
        if in_target_period(record) or not has_article_290(record) or not sentence_number(record):
            continue
        attachments = appellate_attachments(record)
        link_status, first = link_first_instance(record, index)
        first_links = sentence_attachments(first) if first else []
        if not attachments or not first_links:
            continue
        years = record_years(record)
        label = "2026" if 2026 in {years["number"], years["registration"], *years["act"]} else "other/no-year"
        totals[label] += 1
        rows.append({
            "appeal_id": record["_id"],
            "identity": {
                "uid": record.get("uid"), "number": record.get("number"), "year": record.get("year"),
                "courtCode": record.get("courtCode"), "courtName": record.get("courtName"),
                "dateReg": record.get("dateReg"), "act_date": act_date(record),
                "documentType": record.get("documentType"), "years": years,
            },
            "aliases": confirmed_aliases(record),
            "lower": record.get("lower") or [],
            "participants": record.get("participants") or [],
            "appeal_attachments": attachments,
            "first_status": link_status,
            "first": first,
            "first_attachments": first_links,
            "years": [label],
        })
    write_jsonl(raw_dir / "work" / "appeal-candidates.jsonl", rows)
    result = {"outside_both_links": len(rows), "by_year": dict(totals)}
    atomic_json(raw_dir / "work" / "selection.json", result)
    return result


def download_expansion(raw_dir: Path = RAW, pause: float = 4, limit: int = 0, retry_errors: bool = False) -> dict[str, Any]:
    candidates = read_jsonl(raw_dir / "work" / "appeal-candidates.jsonl")
    progress_path = raw_dir / "pair-progress.json"
    progress = read_json(progress_path, {})
    attempted = 0
    consecutive_errors = 0
    stopped_reason = None
    terminal = {"confirmed_pair", "verification_failed", "unsupported_format", "no_compatible_attachment", "manual_review"}
    if not retry_errors:
        terminal.update({"download_error", "captcha"})
    for candidate in candidates:
        appeal_id = str(candidate["appeal_id"])
        current = progress.get(appeal_id, {})
        if current.get("status") in terminal:
            continue
        if limit and attempted >= limit:
            break
        attempted += 1
        current["status"] = "processing"
        current.setdefault("documents", {})
        current.setdefault("attempts", [])
        progress[appeal_id] = current
        atomic_json(progress_path, progress)
        combinations = attachment_combinations(candidate)
        if not combinations:
            current["status"] = "no_compatible_attachment"
            current["errors"] = ["no_compatible_attachment_combination"]
            atomic_json(progress_path, progress)
            continue
        failed_errors = []
        unsupported_errors = []
        try:
            for sentence_attachment, appeal_attachment in combinations:
                try:
                    _, sentence_text = ensure_document(raw_dir, appeal_id, "sentence", sentence_attachment, current, progress_path, progress, pause)
                    _, appeal_text = ensure_document(raw_dir, appeal_id, "appeal", appeal_attachment, current, progress_path, progress, pause)
                    current["both_files_downloaded"] = True
                    current["both_texts_extracted"] = True
                    errors = verify_pair(candidate, sentence_text, appeal_text)
                    outcome = extract_outcome(appeal_text)
                    current["outcome"] = outcome
                    current["attempts"].append({
                        "sentence_url": sentence_attachment["link"], "appeal_url": appeal_attachment["link"],
                        "errors": errors, "outcome": outcome,
                    })
                    current["errors"] = errors
                    if not errors:
                        current["status"] = "confirmed_pair"
                        atomic_json(progress_path, progress)
                        break
                    current["status"] = "processing"
                    failed_errors = errors
                    atomic_json(progress_path, progress)
                except UnsupportedFormat as exc:
                    unsupported_errors.append(str(exc))
                    current["attempts"].append({
                        "sentence_url": sentence_attachment["link"], "appeal_url": appeal_attachment["link"],
                        "errors": [str(exc)], "outcome": "unknown",
                    })
                    refresh_artifact_flags(raw_dir, current)
                    atomic_json(progress_path, progress)
            else:
                if failed_errors:
                    current["status"] = "manual_review" if failed_errors == ["manual_review"] else "verification_failed"
                    current["errors"] = failed_errors
                else:
                    current["status"] = "unsupported_format"
                    current["errors"] = unsupported_errors or ["no_compatible_attachment_combination"]
                refresh_artifact_flags(raw_dir, current)
                atomic_json(progress_path, progress)
        except DownloadError as exc:
            current["status"] = "captcha" if isinstance(exc, CaptchaError) else "download_error"
            current["errors"] = [str(exc)]
            refresh_artifact_flags(raw_dir, current)
            atomic_json(progress_path, progress)
            if isinstance(exc, CaptchaError):
                raise
            if "HTTP 404" in str(exc):
                continue
            consecutive_errors += 1
            if consecutive_errors >= 3:
                stopped_reason = "network_error_series"
                break
        else:
            consecutive_errors = 0
    revalidate_progress(raw_dir)
    write_report(raw_dir)
    return {"attempted": attempted, "statuses": status_summary(progress), "stopped_reason": stopped_reason}


def write_report(raw_dir: Path = RAW, report_path: Path = REPORT) -> None:
    progress = revalidate_progress(raw_dir)
    selection = read_json(raw_dir / "work" / "selection.json", {})
    candidates = read_jsonl(raw_dir / "work" / "appeal-candidates.jsonl")
    candidate_ids = {str(candidate["appeal_id"]) for candidate in candidates}
    progress = {key: value for key, value in progress.items() if key in candidate_ids}
    statuses = Counter(item.get("status", "not_started") for item in progress.values())
    outcomes = Counter(item.get("outcome", "unknown") for item in progress.values() if item.get("status") == "confirmed_pair")
    year_counts = Counter()
    for candidate in candidates:
        state = progress.get(str(candidate["appeal_id"]), {})
        if state.get("status") == "confirmed_pair":
            years = candidate["identity"].get("years", {})
            year = years.get("number") or candidate["identity"].get("year")
            if not year:
                year = str(candidate["identity"].get("act_date", "")[:4] or candidate["identity"].get("dateReg", "")[:4] or "other/no-year")
            year_counts[str(year)] += 1
    remaining = max(0, len(candidates) - sum(1 for item in progress.values() if item.get("status") in {
        "confirmed_pair", "verification_failed", "unsupported_format", "no_compatible_attachment", "manual_review", "download_error", "captcha",
    }))
    lines = [
        "# Расширение полных пар Москвы вне 2022–2025",
        "",
        "Источник — те же официальные открытые наборы Мосгорсуда. 207 ранее обработанных кандидатов не повторяются. Подтверждение требует оба полных текста, SHA и текущую identity. Прямые реквизиты дел не публикуются.",
        "",
        "## Кандидаты с обеими ссылками",
        "",
        "| Год | Количество |",
        "|---|---:|",
    ]
    for year, count in selection.get("by_year", {}).items():
        lines.append(f"| {year} | {count} |")
    lines += [
        f"| **Итого** | **{selection.get('outside_both_links', len(candidates))}** |",
        "",
        "## Статусы",
        "",
        "| Статус | Количество |",
        "|---|---:|",
        f"| confirmed_pair | {statuses.get('confirmed_pair', 0)} |",
        f"| verification_failed | {statuses.get('verification_failed', 0)} |",
        f"| no_compatible_attachment | {statuses.get('no_compatible_attachment', 0)} |",
        f"| unsupported_format | {statuses.get('unsupported_format', 0)} |",
        f"| download_error | {statuses.get('download_error', 0)} |",
        f"| captcha | {statuses.get('captcha', 0)} |",
        f"| manual_review | {statuses.get('manual_review', 0)} |",
        f"| remaining | {remaining} |",
        "",
        "## Подтверждённые пары по годам",
        "",
        "| Год | Количество |",
        "|---|---:|",
    ]
    for year in ("2018", "2019", "2020", "2021", "2026", "other/no-year"):
        lines.append(f"| {year} | {year_counts.get(year, 0)} |")
    lines += ["", "## Outcome подтверждённых пар", "", "| Outcome | Количество |", "|---|---:|"]
    for outcome in ("unchanged", "changed", "cancelled", "unknown"):
        lines.append(f"| {outcome} | {outcomes.get(outcome, 0)} |")
    lines += ["", f"**Новых подтверждённых полных пар: {statuses.get('confirmed_pair', 0)}.**", ""]
    atomic_write(report_path, "\n".join(lines).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Новые полные пары Москвы вне 2022–2025")
    parser.add_argument("command", choices=("select", "download", "report"))
    parser.add_argument("--pause", type=float, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.command == "select":
        print(json.dumps(outside_candidates(), ensure_ascii=False))
    elif args.command == "download":
        if not (RAW / "work" / "selection.json").exists():
            outside_candidates()
        print(json.dumps(download_expansion(pause=args.pause, limit=args.limit, retry_errors=args.retry_errors), ensure_ascii=False))
    else:
        write_report()
        print(json.dumps({"report": str(REPORT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
