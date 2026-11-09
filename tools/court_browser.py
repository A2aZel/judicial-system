from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "work" / "court-browser-profile"
COURTS = {
    "krk": "https://kraevoy.krk.sudrf.ru/",
    "vrn": "https://oblsud.vrn.sudrf.ru/",
    "svd": "https://oblsud.svd.sudrf.ru/",
    "sar": "https://oblsud.sar.sudrf.ru/",
}
ALLOWED_COURT_HOSTS = {urlparse(url).hostname for url in COURTS.values()}


def court_url(key: str) -> str:
    url = COURTS[key]
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_COURT_HOSTS or parsed.path != "/":
        raise ValueError("only official court homepages are allowed")
    return url


async def open_courts(
    keys: list[str],
    profile: Path,
    executable: Path | None,
    headless: bool,
) -> None:
    profile.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as playwright:
        options: dict[str, Any] = {
            "headless": headless,
            "viewport": {"width": 1440, "height": 960},
            "locale": "ru-RU",
            "ignore_https_errors": False,
        }
        if executable:
            options["executable_path"] = str(executable)
        context = await playwright.chromium.launch_persistent_context(
            str(profile),
            **options,
        )
        try:
            for key in keys:
                page = await context.new_page()
                await page.goto(court_url(key), wait_until="domcontentloaded", timeout=60_000)
                print(f"Opened {key}: {await page.title()}", flush=True)
            print(
                "Browser is ready for manual work. Automated search/download is not running. "
                "Solve CAPTCHA and export documents yourself where required.",
                flush=True,
            )
            await asyncio.Event().wait()
        finally:
            await context.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open official court homepages for manual work")
    parser.add_argument("courts", nargs="+", choices=sorted(COURTS))
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.headless:
        raise SystemExit("Use a visible browser for manual CAPTCHA work")
    asyncio.run(open_courts(args.courts, args.profile, args.executable, args.headless))


if __name__ == "__main__":
    main()
