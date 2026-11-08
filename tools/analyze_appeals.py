"""Грубый разбор выборки апелляционных актов СудАкт по ст. 290.

Проверяем, годятся ли апелляционные акты как источник меток «изменён ли приговор и почему»:
сколько из них — жалобы на приговор, а не на меру пресечения; какие исходы и мотивы
встречаются; можно ли по тексту найти приговор первой инстанции (УИД, суд и дата).
Правила простые; точность нужно проверить вручную.

Запуск из корня проекта:
    python tools/analyze_appeals.py
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

DOCS_DIR = Path("corpus/sudact/appeals")
META_PATH = Path("corpus/sudact/appeals.jsonl")
OUT_CSV = Path("corpus/sudact/appeals_analysis.csv")
OUT_SUMMARY = Path("corpus/sudact/appeals_summary.json")

UID_RE = re.compile(r"(?<!\d)(\d{2}RS\d{4}-\d{2}-\d{4}-\d{6}-\d{2})(?!\d)")
# С двоеточием и последнее вхождение: «постановил» встречается и в описательной части.
# Коллегия пишет «ОПРЕДЕЛИЛА:», единоличный судья — «ПОСТАНОВИЛ:».
DECISION_MARK_RE = re.compile(r"(?:П\s*О\s*С\s*Т\s*А\s*Н\s*О\s*В\s*И\s*Л|О\s*П\s*Р\s*Е\s*Д\s*Е\s*Л\s*И\s*Л(?:\s*А)?)\s*:", re.I)
FIRST_INSTANCE_RE = re.compile(
    r"приговор\w*\s+.{0,160}?(?:суда|судьи).{0,160}?от\s+(?:\d{1,2}\s*[а-я]+\s+\d{4}|\d{2}\.\d{2}\.\d{4})", re.I | re.S
)
ART_290_RE = re.compile(r"ст(?:\.|атьи|атьей|атье)?\s*290(?![.\d])")

# Мотивы, которые относятся к наказанию (п. 3–4 ст. 389.15 УПК РФ)
REASONS = {
    "too_harsh": r"чрезмерн\w* суров|несправедлив\w* вследствие суровости",
    "too_lenient": r"чрезмерн\w* мягк|вследствие (?:чрезмерной )?мягкости",
    "misapplied_law": r"неправильн\w* применени\w* уголовного закона",
    "art_64_73": r"ст\.?\s*(?:64|73) УК|статьи (?:64|73) УК",
    "mitigating": r"смягчающ\w* обстоятельств",
    "additional_punishment": r"дополнительн\w* наказани",
    "confiscation": r"конфиск",
    "time_served": r"ст\.?\s*72 УК|зач[её]т\w* времени",
}


def analyse(meta: dict, text: str) -> dict:
    marks = list(DECISION_MARK_RE.finditer(text))
    resolutive = text[marks[-1].end():marks[-1].end() + 3000].lower() if marks else ""
    head = text[:6000].lower()
    # резолютивная часть жалобы на приговор начинается со слова «приговор …»
    about_verdict = "приговор" in resolutive[:150]
    outcome = ""
    if about_verdict:
        first = resolutive[:800]
        if "отменить" in first:
            outcome = "отменён"
        elif "изменить" in first:
            outcome = "изменён"
        elif "без изменения" in first:
            outcome = "без изменения"
    row = {
        "id": meta["id"],
        "period": meta.get("period"),
        "title": meta["title"].split(" №")[0],
        "about_verdict": about_verdict,
        "outcome": outcome,
        "art_290": bool(ART_290_RE.search(text)),
        "prosecutor_appeal": "апелляционн" in head and "представлени" in head,
        "defence_appeal": bool(re.search(r"апелляционн\w* жалоб", head)),
        "uid": bool(UID_RE.search(text)),
        # «приговор <суд> от <дата> в отношении …» — ключ для связки без УИД
        "first_instance_ref": bool(FIRST_INSTANCE_RE.search(resolutive[:400])),
        "chars": len(text),
    }
    for key, pattern in REASONS.items():
        row[f"reason_{key}"] = bool(re.search(pattern, text, re.I))
    return row


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    metas = [json.loads(line) for line in META_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [
        analyse(meta, (DOCS_DIR / f"{meta['id']}.txt").read_text(encoding="utf-8"))
        for meta in metas
        if (DOCS_DIR / f"{meta['id']}.txt").exists()
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    verdict_appeals = [r for r in rows if r["about_verdict"]]
    on_290 = [r for r in verdict_appeals if r["art_290"]]
    summary = {
        "candidates": len(rows),
        "about_verdict": len(verdict_appeals),
        "about_verdict_art_290": len(on_290),
        "outcomes": dict(Counter(r["outcome"] or "не распознан" for r in verdict_appeals)),
        "prosecutor_appeal": sum(r["prosecutor_appeal"] for r in verdict_appeals),
        "defence_appeal": sum(r["defence_appeal"] for r in verdict_appeals),
        "with_uid": sum(r["uid"] for r in verdict_appeals),
        "with_first_instance_ref": sum(r["first_instance_ref"] for r in verdict_appeals),
        "reasons_in_changed": dict(Counter(
            key for r in verdict_appeals if r["outcome"] in ("изменён", "отменён")
            for key in REASONS if r[f"reason_{key}"]
        )),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
