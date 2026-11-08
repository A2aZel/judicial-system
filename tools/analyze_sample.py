"""Грубый разбор выборки приговоров СудАкт для оценки корпуса.

Это не извлечение для обучения, а проверка данных: сколько среди
документов настоящих приговоров по ст. 290, есть ли УИД, какие режимы и
наказания встречаются, насколько длинные и как обезличены тексты.
Правила намеренно простые; их точность нужно проверить вручную на части выборки.

Запуск из корня проекта:
    python tools/analyze_sample.py
"""

import csv
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

DOCS_DIR = Path("corpus/sudact/docs")
META_PATH = Path("corpus/sudact/docs.jsonl")
OUT_CSV = Path("corpus/sudact/analysis.csv")
OUT_SUMMARY = Path("corpus/sudact/summary.json")

UID_RE = re.compile(r"(?<!\d)(\d{2}RS\d{4}-\d{2}-\d{4}-\d{6}-\d{2})(?!\d)")
# С двоеточием: «приговорил» без него встречается в описании прежних судимостей.
VERDICT_MARK_RE = re.compile(r"П\s*Р\s*И\s*Г\s*О\s*В\s*О\s*Р\s*И\s*Л\s*:", re.I)
ART_290_RE = re.compile(r"ст(?:\.|атьи|атьей|атье|атья)?\s*290(?![.\d])")
PART_290_RE = re.compile(
    r"ч(?:\.|асти|астью|асть)?\s*(\d)\s*ст(?:\.|атьи|атьей|атье)?\s*290(?![.\d])"
    r"|ст(?:\.|атьи|атьей|атье)?\s*290\s*ч(?:\.|асти|асть)?\s*(\d)"
)
OTHER_ART_RE = re.compile(r"ст(?:\.|атьи|атьей)?\s*(291\.2|291\.1|291|285|286|159|160|204)(?![.\d])")
CLAUSE_ART_RE = re.compile(r"ст(?:\.|атьи|атьей|атье)?\s*(\d{2,3}(?:\.\d)?)(?![.\d])")
REDACTION_RE = re.compile(r"<данные изъяты>|\(данные изъяты\)|\[данные изъяты\]|<\.\.\.>|\bФИО\d*\b|\(обезличено\)", re.I)
TERM_RE = re.compile(r"лишени[яе] свободы\s+(?:на\s+)?(?:срок\s+)?([^;]{0,80})", re.I)
YEARS_RE = re.compile(r"(\d+)\s*(?:\([^)]*\)\s*)?(?:год|лет)")
MONTHS_RE = re.compile(r"(\d+)\s*(?:\([^)]*\)\s*)?месяц")
# Формулировка осуждения: «признать … виновным в совершении …, предусмотренного ч. 3 ст. 290 УК РФ».
# Статьи берём только отсюда: ниже в резолютивной части 290 встречается и в чужих делах.
CONVICTION_CLAUSE_RE = re.compile(r"виновн\w*.{0,200}?предусмотренн\w*(.{0,400}?)(?:УК\s*Р|\bи\s+назнач)", re.I | re.S)
TERM_REMOVED_RE = re.compile(r"(?:на срок|сроком на)\s+(?:год|лет|месяц)", re.I)
AMOUNT_REMOVED_RE = re.compile(r"(?:в размере|в сумме|сумму)\s+рубл", re.I)
FINE_RE = re.compile(r"штраф\w*\s+в\s+размере\s+([\d\s ]{4,})\s*(?:\([^)]*\)\s*)?руб", re.I)


def has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.I) is not None


def term_months(fragment: str) -> int | None:
    years = YEARS_RE.search(fragment)
    months = MONTHS_RE.search(fragment)
    if not years and not months:
        return None
    return (int(years.group(1)) * 12 if years else 0) + (int(months.group(1)) if months else 0)


def analyse(meta: dict, text: str) -> dict:
    title = meta["title"]
    doc_type = title.split(" ", 1)[0]
    mark = VERDICT_MARK_RE.search(text)
    resolutive = text[mark.end():mark.end() + 6000] if mark else ""
    # итоговое наказание при совокупности обычно идёт после слова «окончательно»
    final_idx = resolutive.lower().rfind("окончательн")
    final_part = resolutive[final_idx:] if final_idx >= 0 else resolutive
    term = TERM_RE.search(final_part)
    fine = FINE_RE.search(final_part)
    clauses = " | ".join(m.group(0) for m in CONVICTION_CLAUSE_RE.finditer(resolutive))
    parts_290 = sorted({a or b for a, b in PART_290_RE.findall(clauses)})
    uid = UID_RE.search(text)
    return {
        "id": meta["id"],
        "period": meta.get("period"),
        "doc_type": doc_type,
        "court_kind": meta["court"].rsplit(" - ", 1)[-1] if " - " in meta["court"] else "",
        "chars": len(text),
        "words": len(text.split()),
        "uid": uid.group(1) if uid else "",
        "has_resolutive": bool(mark),
        "convicted_290": bool(ART_290_RE.search(clauses)),
        "parts_290": "|".join(parts_290),
        "other_articles": "|".join(sorted(set(OTHER_ART_RE.findall(clauses)))),
        "clause_articles": "|".join(sorted(set(CLAUSE_ART_RE.findall(clauses)) - {"30", "33"})),
        "acquittal": has(r"оправда", resolutive[:1500]),
        "special_order": has(r"особом порядке|особого порядка|ст\.?\s*316 УПК|главы 40 УПК", text),
        "pretrial_agreement": has(r"досудебн\w* соглашени|главы 40\.1|ст\.?\s*317\.7", text),
        "art_64": has(r"ст\.?\s*64 УК|статьи 64 УК", resolutive),
        "art_73": has(r"ст\.?\s*73 УК|статьи 73 УК|условн", resolutive),
        "art_69": has(r"ст\.?\s*69 УК|статьи 69 УК|совокупност", resolutive),
        "prison_months": term_months(term.group(1)) if term else None,
        "fine_rub": int(re.sub(r"\D", "", fine.group(1))) if fine else None,
        "fine_multiple": has(r"кратн", resolutive),
        "ban_on_positions": has(r"прав\w* занимать", resolutive),
        "confiscation": has(r"конфиск", resolutive),
        "redactions": len(REDACTION_RE.findall(text)),
        # обезличивание иногда вырезает суммы и даты: «в размере рублей», «ДД.ММ.ГГГГ»
        "amounts_removed": len(AMOUNT_REMOVED_RE.findall(text)),
        "term_removed": bool(TERM_REMOVED_RE.search(final_part)),
        "dates_removed": text.count("ДД.ММ.ГГГГ"),
    }


def share(rows: list[dict], key: str) -> float:
    return round(sum(1 for r in rows if r[key]) / len(rows), 3) if rows else 0.0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    metas =[json.loads(line) for line in META_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for meta in metas:
        path = DOCS_DIR / f"{meta['id']}.txt"
        if path.exists():
            rows.append(analyse(meta, path.read_text(encoding="utf-8")))

    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    verdicts = [r for r in rows if r["doc_type"] == "Приговор"]
    convicted = [r for r in verdicts if r["convicted_290"]]
    chars = [r["chars"] for r in convicted] or [0]
    prison = [r["prison_months"] for r in convicted if r["prison_months"]]
    summary = {
        "documents": len(rows),
        "verdicts": len(verdicts),
        "convicted_290": len(convicted),
        "convicted_290_share_of_verdicts": round(len(convicted) / len(verdicts), 3) if verdicts else 0,
        "by_period": dict(Counter(r["period"] for r in convicted)),
        "parts_290": dict(Counter(r["parts_290"] or "не распознана" for r in convicted)),
        "with_uid": share(convicted, "uid"),
        "with_uid_by_period": {
            p: share([r for r in convicted if r["period"] == p], "uid")
            for p in sorted({r["period"] for r in convicted})
        },
        "special_order": share(convicted, "special_order"),
        "pretrial_agreement": share(convicted, "pretrial_agreement"),
        "art_64": share(convicted, "art_64"),
        "art_73_or_conditional": share(convicted, "art_73"),
        "art_69_multiple_counts": share(convicted, "art_69"),
        "fine_multiple": share(convicted, "fine_multiple"),
        "ban_on_positions": share(convicted, "ban_on_positions"),
        "confiscation": share(convicted, "confiscation"),
        "with_prison_term": round(len(prison) / len(convicted), 3) if convicted else 0,
        "prison_months_median": statistics.median(prison) if prison else None,
        "chars_median": statistics.median(chars),
        "chars_p90": sorted(chars)[int(len(chars) * 0.9) - 1] if len(chars) > 1 else chars[0],
        "chars_max": max(chars),
        "over_8192_tokens_est": share([{"x": r["chars"] / 3.5 > 8192} for r in convicted], "x"),
        "redactions_median": statistics.median([r["redactions"] for r in convicted]) if convicted else 0,
        "no_redactions": share([{"x": r["redactions"] == 0} for r in convicted], "x"),
        "with_amounts_removed": share([{"x": r["amounts_removed"] > 0} for r in convicted], "x"),
        "with_dates_removed": share([{"x": r["dates_removed"] > 0} for r in convicted], "x"),
        "with_term_removed": share(convicted, "term_removed"),
        "acquittals_290": sum(1 for r in verdicts if r["acquittal"] and not r["convicted_290"]),
        "conviction_clause_not_found": sum(1 for r in verdicts if not r["clause_articles"] and not r["acquittal"]),
        "articles_in_non_290_verdicts": dict(Counter(
            a for r in verdicts if not r["convicted_290"] for a in (r["clause_articles"] or "не найдено").split("|")
        ).most_common(12)),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
