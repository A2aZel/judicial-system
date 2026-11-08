"""Скачивает тексты документов СудАкт по списку из sudact_listing.js.

Страницы документов (/regular/doc/<id>/) открыты для роботов в robots.txt.
Скрипт идёт последовательно с паузой, пропускает уже скачанное и
останавливается, если сайт начинает отдавать капчу или ошибки.

Запуск из корня проекта:
    python tools/fetch_sudact_docs.py corpus/sudact/listing.json --only-verdicts
"""

import argparse
import html
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://sudact.ru/regular/doc/{id}/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-sample/0.1"
DOCS_DIR = Path("corpus/sudact/docs")
META_PATH = Path("corpus/sudact/docs.jsonl")


class Blocked(Exception):
    """Сайт перестал отдавать документы: капча или серия ошибок."""


def fetch(doc_id: str) -> str:
    request = urllib.request.Request(BASE_URL.format(id=doc_id), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read().decode("utf-8", "replace")


def html_to_text(fragment: str) -> str:
    fragment = re.sub(r"<script.*?</script>|<style.*?</style>|<!--.*?-->", " ", fragment, flags=re.S)
    fragment = re.sub(r"<br\s*/?>|</p>|</div>|</center>", "\n", fragment, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", fragment))
    text = re.sub(r"[ \t ]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def parse(page: str) -> dict:
    if "check_captcha" in page or ("captcha" in page.lower() and "<h1>" not in page):
        raise Blocked("captcha")
    title = re.search(r"<h1>(.*?)</h1>", page, re.S)
    court = re.search(r'<div class="b-justice">(.*?)</div>', page, re.S)
    start = page.find('<hr class="hr-h1">')
    end = page.find("<h3>Суд:</h3>", start)
    if not title or start < 0:
        raise ValueError("document layout not recognised")
    return {
        "title": html_to_text(title.group(1)),
        "court": html_to_text(court.group(1)) if court else "",
        "text": html_to_text(page[start:end if end > 0 else None]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("listing", type=Path)
    parser.add_argument("--only-verdicts", action="store_true", help="качать только «Приговор …»")
    parser.add_argument("--pause", type=float, default=4.0, help="базовая пауза между запросами, с")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--meta", type=Path, default=META_PATH)
    args = parser.parse_args()
    docs_dir, meta_path = args.docs_dir, args.meta

    listing = [row for row in json.loads(args.listing.read_text(encoding="utf-8")) if row.get("id")]
    if args.only_verdicts:
        listing = [row for row in listing if (row.get("type") or row.get("title", "")).startswith("Приговор")]
    seen = {}
    for row in listing:
        seen.setdefault(row["id"], row)
    queue = [row for doc_id, row in seen.items() if not (docs_dir / f"{doc_id}.txt").exists()]
    if args.limit:
        queue = queue[: args.limit]

    docs_dir.mkdir(parents=True, exist_ok=True)
    print(f"to fetch: {len(queue)} of {len(seen)} unique", flush=True)
    failures_in_row = 0
    for n, row in enumerate(queue, 1):
        doc_id = row["id"]
        try:
            doc = parse(fetch(doc_id))
        except Blocked as exc:
            print(f"STOP at {doc_id}: {exc}", flush=True)
            return 2
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            failures_in_row += 1
            print(f"fail {doc_id}: {exc}", flush=True)
            if failures_in_row >= 3:
                print("STOP: 3 failures in a row", flush=True)
                return 2
            time.sleep(args.pause * 3)
            continue
        failures_in_row = 0
        (docs_dir / f"{doc_id}.txt").write_text(doc["text"], encoding="utf-8")
        meta = {
            "id": doc_id,
            "url": BASE_URL.format(id=doc_id),
            "title": doc["title"],
            "court": doc["court"],
            "period": row.get("period"),
            "chars": len(doc["text"]),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with meta_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        print(f"{n}/{len(queue)} {doc_id} {meta['chars']} chars", flush=True)
        time.sleep(args.pause + random.uniform(0, args.pause / 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
