import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from judicial_ai.corpus.bribe_amount import MAX_AMOUNT_RUB, extract_bribe_amount
from judicial_ai.corpus.punishment_outcome import (
    ARTICLE_290_PARTS,
    UNKNOWN_OUTCOME,
    explicit_principal_outcome,
    extract_principal_outcome,
)
from judicial_ai.corpus.records import CaseRecord


_JUDGE_SIGNATURE = re.compile(
    r"(?m)^[ \t]*(?:судья|Судья|СУДЬЯ)[ \t]*:?[ \t]+"
    r"(?:[А-ЯЁA-Z]\.[ \t]*){1,2}(?:[А-ЯЁA-Z][а-яёa-z-]{1,40})?[ \t]*\Z"
)


class ProcessedRepositoryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProcessedCase:
    record: CaseRecord
    narrative: str | None
    text_sha256: str | None


def text_sha256(narrative: str) -> str:
    return hashlib.sha256(narrative.encode("utf-8")).hexdigest()


def _recommendation_text_hash(narrative: str, source: Any) -> str:
    source = source.strip() if isinstance(source, str) else ""
    body = _JUDGE_SIGNATURE.sub("", narrative) if source else narrative
    normalized = " ".join(body.split())
    return text_sha256(f"{source}\n{normalized}")


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value


def _bool(value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError("expected boolean")
    return value


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else _bool(value)


def _number(value: Any, maximum: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("expected number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or parsed > maximum:
        raise ValueError("number is non-finite or out of range")
    return parsed


def _counter(value: Any, minimum: int, maximum: int) -> int | None:
    parsed = _number(value, maximum)
    if parsed is None:
        return None
    if not parsed.is_integer() or parsed < minimum:
        raise ValueError("expected integer counter")
    return int(parsed)


def _parts(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("article_parts must be non-empty")
    parts = tuple(part.strip() for part in value if isinstance(part, str) and part.strip())
    if len(parts) != len(value):
        raise ValueError("invalid article part")
    return parts


def _factor_count(value: Any) -> int:
    if not isinstance(value, list) or len(value) > 30:
        raise ValueError("invalid factors")
    return len(value)


class ProcessedRepository:
    def __init__(self, path: Path):
        self.path = path
        self.cases, self.skipped_records = self._load(path)
        self.records = tuple(case.record for case in self.cases)

    @staticmethod
    def _load(path: Path) -> tuple[tuple[ProcessedCase, ...], int]:
        if not path.is_file():
            raise ProcessedRepositoryError("processed repository is not available")
        cases: list[ProcessedCase] = []
        skipped = 0
        with path.open(encoding="utf-8") as handle:
            for order, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    case = ProcessedRepository._parse(payload, order)
                    cases.append(case)
                except (TypeError, ValueError, OverflowError, json.JSONDecodeError):
                    skipped += 1
        return tuple(cases), skipped

    @staticmethod
    def _parse(payload: Any, order: int) -> ProcessedCase:
        payload = _mapping(payload)
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported schema version")
        case_id = payload.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id is required")
        process = _mapping(payload.get("process"))
        facts = _mapping(payload.get("facts"))
        punishment = _mapping(payload.get("punishment"))
        narrative = facts.get("narrative")
        if narrative is not None and not isinstance(narrative, str):
            raise ValueError("narrative must be a string")
        narrative = narrative.strip() if narrative else None
        conditional = _optional_bool(punishment.get("conditional"))
        parts = _parts(payload.get("article_parts"))
        art_69 = _optional_bool(process.get("art_69"))
        amount_rub = _number(facts.get("amount_rub"), MAX_AMOUNT_RUB)
        facts_trusted = _optional_bool(facts.get("recommendation_facts_trusted")) is True
        recommendation_amount_rub = None
        principal = UNKNOWN_OUTCOME
        if (
            len(parts) == 1
            and parts[0] in ARTICLE_290_PARTS
            and art_69 is not True
            and not payload.get("other_articles")
            and payload.get("convicted_290") is not False
        ):
            principal = explicit_principal_outcome(
                punishment.get("principal_punishment"),
                _number(punishment.get("principal_value"), 1_000_000_000_000),
                _optional_bool(punishment.get("principal_conditional")),
            )
            if principal == UNKNOWN_OUTCOME and narrative:
                principal = extract_principal_outcome(narrative, parts[0])
            if facts_trusted and amount_rub is not None and amount_rub > 0:
                recommendation_amount_rub = amount_rub
            elif principal[0] is not None and narrative:
                recommendation_amount_rub = extract_bribe_amount(narrative)
        principal_punishment, principal_value, principal_conditional = principal
        group = payload.get("verdict_group_id")
        group = (group.strip() or None) if isinstance(group, str) else None
        record = CaseRecord(
            article_parts=parts,
            special_order=_optional_bool(process.get("special_order")),
            pretrial_agreement=_optional_bool(process.get("pretrial_agreement")),
            art_64=_optional_bool(process.get("art_64")),
            art_69=art_69,
            amount_rub=amount_rub,
            episodes=_counter(facts.get("episodes"), 1, 1000),
            mitigating_count=_factor_count(facts.get("mitigating", [])),
            aggravating_count=_factor_count(facts.get("aggravating", [])),
            prison_months=_number(punishment.get("prison_months"), 1200),
            fine_rub=_number(punishment.get("fine_rub"), 1_000_000_000_000),
            fine_multiple=_optional_bool(punishment.get("fine_multiple")),
            ban_on_positions=_optional_bool(punishment.get("ban_on_positions")),
            ban_years=_number(punishment.get("ban_years"), 100),
            conditional=(
                conditional
                if conditional is not None
                else _optional_bool(process.get("art_73"))
            ),
            confiscation=_optional_bool(punishment.get("confiscation")),
            source_order=order,
            principal_punishment=principal_punishment,
            principal_value=principal_value,
            principal_conditional=principal_conditional,
            recommendation_facts_trusted=facts_trusted,
            recommendation_group=group,
            recommendation_text_hash=(
                _recommendation_text_hash(narrative, payload.get("source")) if narrative else None
            ),
            recommendation_amount_rub=recommendation_amount_rub,
        )
        return ProcessedCase(
            record=record,
            narrative=narrative,
            text_sha256=text_sha256(narrative) if narrative else None,
        )
