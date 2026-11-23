import csv
import logging
import math
from pathlib import Path

from judicial_ai.corpus.punishment_outcome import (
    ARTICLE_290_PARTS,
    UNKNOWN_OUTCOME,
    explicit_principal_outcome,
)
from judicial_ai.corpus.records import CaseRecord

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "convicted_290",
    "parts_290",
    "special_order",
    "pretrial_agreement",
    "art_64",
    "art_69",
    "art_73",
    "prison_months",
    "fine_rub",
    "fine_multiple",
    "ban_on_positions",
    "confiscation",
}


class CorpusError(ValueError):
    pass


def _bool(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError("required boolean is empty or unknown")


def _optional_bool(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    return _bool(value)


def _float(value: str | None, maximum: float) -> float | None:
    if value is None or not value.strip():
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or parsed > maximum:
        raise ValueError("numeric value is non-finite or out of range")
    return parsed


def _int(value: str | None, minimum: int, maximum: int) -> int | None:
    parsed = _float(value, maximum)
    if parsed is None:
        return None
    if not parsed.is_integer() or parsed < minimum:
        raise ValueError("counter must be an integer in range")
    return int(parsed)


class CsvCorpusRepository:
    def __init__(self, path: Path):
        self.path = path
        self.records, self.skipped_records = self._load(path)

    @staticmethod
    def _load(path: Path) -> tuple[tuple[CaseRecord, ...], int]:
        if not path.is_file():
            raise CorpusError("corpus file is not available")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise CorpusError(f"corpus is missing required columns: {sorted(missing)}")
            records: list[CaseRecord] = []
            skipped = 0
            for order, row in enumerate(reader):
                try:
                    if not _bool(row["convicted_290"]):
                        continue
                    parts = tuple(part for part in (row["parts_290"] or "").split("|") if part)
                    if not parts:
                        continue
                    art_69 = _bool(row["art_69"])
                    principal = UNKNOWN_OUTCOME
                    if (
                        len(parts) == 1
                        and parts[0] in ARTICLE_290_PARTS
                        and not art_69
                        and not row.get("other_articles")
                    ):
                        principal = explicit_principal_outcome(
                            (row.get("principal_punishment") or "").strip().lower() or None,
                            _float(row.get("principal_value"), 1_000_000_000_000),
                            _optional_bool(row.get("principal_conditional")),
                        )
                    principal_punishment, principal_value, principal_conditional = principal
                    records.append(
                        CaseRecord(
                            article_parts=parts,
                            special_order=_bool(row["special_order"]),
                            pretrial_agreement=_bool(row["pretrial_agreement"]),
                            art_64=_bool(row["art_64"]),
                            art_69=art_69,
                            amount_rub=_float(row.get("amount_rub"), 1_000_000_000_000),
                            episodes=_int(row.get("episodes"), 1, 1000),
                            mitigating_count=_int(row.get("mitigating_count"), 0, 30),
                            aggravating_count=_int(row.get("aggravating_count"), 0, 30),
                            prison_months=_float(row.get("prison_months"), 1200),
                            fine_rub=_float(row.get("fine_rub"), 1_000_000_000_000),
                            fine_multiple=_optional_bool(row.get("fine_multiple")),
                            ban_on_positions=_optional_bool(row.get("ban_on_positions")),
                            ban_years=_float(row.get("ban_years"), 100),
                            conditional=_optional_bool(row.get("art_73")),
                            confiscation=_optional_bool(row.get("confiscation")),
                            source_order=order,
                            principal_punishment=principal_punishment,
                            principal_value=principal_value,
                            principal_conditional=principal_conditional,
                            recommendation_group=(row.get("verdict_group_id") or "").strip() or None,
                        )
                    )
                except (TypeError, ValueError, OverflowError):
                    skipped += 1
        logger.info(
            "Corpus loaded: records=%d skipped=%d", len(records), skipped
        )
        return tuple(records), skipped
