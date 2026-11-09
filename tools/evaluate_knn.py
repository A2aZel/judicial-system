import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judicial_ai.analogues.dense import (
    DENSE_MODEL_ID,
    DENSE_MODEL_REVISION,
    DenseAnalogueFinder,
    DenseEncoder,
    Encoder,
    load_dense_index,
)
from judicial_ai.analogues.knn import KNNAnalogueFinder, KNNResult, Neighbor
from judicial_ai.case_card import CaseCard
from judicial_ai.corpus.processed_repository import ProcessedCase, ProcessedRepository

TOP_K = 5
TARGETS = ("prison_months", "fine_rub")
RANKED_METHODS = ("structured", "dense", "lexical", "dense_structured")
METHODS = ("cohort_median",) + RANKED_METHODS
METHOD_LABELS = {
    "cohort_median": "Cohort median",
    "structured": "Structured kNN",
    "dense": "Dense kNN",
    "lexical": "Lexical BM25-like",
    "dense_structured": "Dense + structured",
}
OUTCOMES = ("cancelled", "changed", "unchanged")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Candidate:
    case_id: str
    verdict_group_id: str
    case: ProcessedCase


@dataclass(frozen=True, slots=True)
class QueryExample:
    case_id: str
    verdict_group_id: str
    outcome: str
    article_parts: tuple[str, ...]
    article_part: str
    card: CaseCard
    narrative: str
    text_sha256: str
    prison_months: float | None
    fine_rub: float | None
    source_year: int | None = None


class CachedEncoder:
    def __init__(self, encoder: Encoder):
        self._encoder = encoder
        self._cache: dict[str, np.ndarray] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._encoder, name)

    def configure(self, preprocessing: dict[str, Any]) -> None:
        self._encoder.configure(preprocessing)

    def info(self) -> dict[str, object]:
        return self._encoder.info()

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        if missing:
            vectors = np.asarray(self._encoder.encode(missing), dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(missing):
                raise ValueError("invalid encoder output")
            self._cache.update(zip(missing, vectors, strict=True))
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([self._cache[text] for text in texts]).astype(np.float32)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL row in {path} is not an object")
                rows.append(payload)
    return rows


def _query_card(record: Any, article_part: str, narrative: str) -> CaseCard:
    return CaseCard(
        article_part=article_part,
        amount_rub=record.amount_rub,
        episodes=record.episodes,
        special_order=record.special_order,
        pretrial_agreement=record.pretrial_agreement,
        art_64=record.art_64,
        art_69=record.art_69,
        mitigating_count=record.mitigating_count,
        aggravating_count=record.aggravating_count,
        narrative=narrative,
    )


def load_risk_queries(path: Path) -> tuple[tuple[QueryExample, ...], list[dict[str, str]]]:
    queries: list[QueryExample] = []
    skipped: list[dict[str, str]] = []
    seen_groups: set[str] = set()
    seen_hashes: set[str] = set()
    for order, row in enumerate(_read_jsonl(path)):
        case_id = row.get("case_id")
        group = row.get("verdict_group_id")
        outcome = row.get("outcome")
        features = row.get("features")
        if (
            not isinstance(case_id, str)
            or not isinstance(group, str)
            or outcome not in OUTCOMES
            or not isinstance(features, dict)
        ):
            skipped.append({"case_id": str(case_id), "reason": "invalid_risk_schema"})
            continue
        parts = features.get("article_parts")
        if not isinstance(parts, list) or not parts or not all(isinstance(part, str) for part in parts):
            skipped.append({"case_id": case_id, "reason": "article_part_missing"})
            continue
        payload = {
            "schema_version": row.get("schema_version"),
            "case_id": case_id,
            "article_parts": parts,
            "process": features.get("process"),
            "facts": features.get("facts"),
            "punishment": features.get("punishment"),
        }
        try:
            processed = ProcessedRepository._parse(payload, order)
        except (TypeError, ValueError, OverflowError):
            skipped.append({"case_id": case_id, "reason": "invalid_risk_values"})
            continue
        if not processed.narrative or not processed.text_sha256:
            skipped.append({"case_id": case_id, "reason": "narrative_missing"})
            continue
        if group in seen_groups:
            skipped.append({"case_id": case_id, "reason": "duplicate_verdict_group"})
            continue
        if processed.text_sha256 in seen_hashes:
            skipped.append({"case_id": case_id, "reason": "duplicate_text_hash"})
            continue
        seen_groups.add(group)
        seen_hashes.add(processed.text_sha256)
        record = processed.record
        queries.append(
            QueryExample(
                case_id=case_id,
                verdict_group_id=group,
                outcome=outcome,
                article_parts=tuple(parts),
                article_part=parts[0],
                card=_query_card(record, parts[0], processed.narrative),
                narrative=processed.narrative,
                text_sha256=processed.text_sha256,
                prison_months=record.prison_months,
                fine_rub=record.fine_rub,
                source_year=row.get("source_year") if type(row.get("source_year")) is int else None,
            )
        )
    return tuple(queries), skipped


def load_retrieval_candidates(path: Path) -> tuple[tuple[Candidate, ...], dict[str, int]]:
    raw_rows = _read_jsonl(path)
    repository = ProcessedRepository(path)
    candidates: list[Candidate] = []
    seen_groups: set[str] = set()
    seen_hashes: set[str] = set()
    duplicate_records = 0
    for case in repository.cases:
        raw = raw_rows[case.record.source_order]
        case_id = raw.get("case_id")
        group = raw.get("verdict_group_id")
        if not isinstance(case_id, str) or not isinstance(group, str):
            raise ValueError("valid processed record is missing its identifiers")
        if group in seen_groups or (case.text_sha256 and case.text_sha256 in seen_hashes):
            duplicate_records += 1
            continue
        seen_groups.add(group)
        if case.text_sha256:
            seen_hashes.add(case.text_sha256)
        candidates.append(Candidate(case_id, group, case))
    stats = {
        "source_records": len(raw_rows),
        "parser_valid_records": len(repository.cases),
        "parser_skipped_records": repository.skipped_records,
        "duplicate_records_skipped": duplicate_records,
        "evaluation_documents": len(candidates),
        "unique_verdict_groups": len(seen_groups),
        "unique_text_hashes": len(seen_hashes),
    }
    return tuple(candidates), stats


def is_eligible(query: QueryExample, candidate: Candidate) -> bool:
    return (
        candidate.case_id != query.case_id
        and candidate.verdict_group_id != query.verdict_group_id
        and candidate.case.text_sha256 != query.text_sha256
    )


def eligible_candidates(
    query: QueryExample, candidates: Iterable[Candidate]
) -> tuple[Candidate, ...]:
    return tuple(candidate for candidate in candidates if is_eligible(query, candidate))


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(text.casefold())
        if len(token) >= 2 and any(character.isalpha() for character in token)
    ]


def lexical_rank(
    query_text: str, candidates: Iterable[Candidate], top_k: int = TOP_K
) -> tuple[Neighbor, ...]:
    documents = list(candidates)
    if not documents:
        return ()
    tokens = [tokenize(candidate.case.narrative or "") for candidate in documents]
    query_tokens = tokenize(query_text)
    average_length = sum(map(len, tokens)) / len(documents)
    if not query_tokens or average_length == 0:
        return tuple(
            Neighbor(candidate.case.record, 1.0) for candidate in documents[:top_k]
        )
    document_frequency: Counter[str] = Counter()
    for document in tokens:
        document_frequency.update(set(document))
    inverse_frequency = {
        term: math.log1p((len(documents) - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
    }
    query_frequency = Counter(query_tokens)
    k1 = 1.5
    b = 0.75
    ranked: list[Neighbor] = []
    for candidate, document in zip(documents, tokens, strict=True):
        frequencies = Counter(document)
        score = 0.0
        for term, query_weight in query_frequency.items():
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            normalization = k1 * (1.0 - b + b * len(document) / average_length)
            score += (
                query_weight
                * inverse_frequency.get(term, 0.0)
                * frequency
                * (k1 + 1.0)
                / (frequency + normalization)
            )
        ranked.append(Neighbor(candidate.case.record, 1.0 / (1.0 + score)))
    ranked.sort(key=lambda neighbor: (neighbor.distance, neighbor.record.source_order))
    return tuple(ranked[:top_k])


def known_median(records: Iterable[Any], target: str) -> float | None:
    values = [
        float(value)
        for record in records
        if (value := getattr(record, target)) is not None
    ]
    return statistics.median(values) if values else None


def neighbor_median(neighbors: Iterable[Neighbor], target: str) -> float | None:
    return known_median((neighbor.record for neighbor in neighbors), target)


def regression_metrics(
    rows: Iterable[dict[str, Any]], target: str
) -> dict[str, Any]:
    selected = list(rows)
    known = [row for row in selected if row["actual"][target] is not None]
    scored = [row for row in known if row["predicted"][target] is not None]
    transform = math.log1p if target == "fine_rub" else float
    errors = [
        transform(float(row["predicted"][target]))
        - transform(float(row["actual"][target]))
        for row in scored
    ]
    absolute = [abs(error) for error in errors]
    return {
        "n_queries": len(selected),
        "n_known": len(known),
        "n_scored": len(scored),
        "coverage": len(scored) / len(known) if known else None,
        "mae": statistics.fmean(absolute) if absolute else None,
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors))
        if errors
        else None,
        "median_absolute_error": statistics.median(absolute) if absolute else None,
    }


def _excerpt(text: str, limit: int = 1800) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()[:limit]


def _counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def evaluate(
    queries: tuple[QueryExample, ...],
    candidates: tuple[Candidate, ...],
    index_path: Path,
    manifest_path: Path,
    encoder: Encoder,
    top_k: int = TOP_K,
    dense_weight: float = 0.7,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if top_k != TOP_K:
        raise ValueError("the evaluation protocol requires top_k=5")
    cached_encoder = CachedEncoder(encoder)
    cases = tuple(candidate.case for candidate in candidates)
    dense_finder = DenseAnalogueFinder(
        cases, index_path, manifest_path, cached_encoder, dense_weight=1.0
    )
    combined_finder = DenseAnalogueFinder(
        cases, index_path, manifest_path, cached_encoder, dense_weight=dense_weight
    )
    embeddings, _ = load_dense_index(
        index_path, manifest_path, encoder.model_id, encoder.revision
    )
    rows: list[dict[str, Any]] = []
    review_queries: list[dict[str, Any]] = []
    technical = {
        method: {
            "queries": len(queries),
            "queries_with_neighbors": 0,
            "returned_neighbors": 0,
            "non_null_distances": 0,
            "null_distances": 0,
            "self_exclusion_violations": 0,
            "cohort_filter_violations": 0,
        }
        for method in RANKED_METHODS
    }
    fallbacks = {"dense": Counter(), "dense_structured": Counter()}
    missing_embeddings = 0
    structured_unranked = 0
    by_order = {candidate.case.record.source_order: candidate for candidate in candidates}

    for query in queries:
        eligible = eligible_candidates(query, candidates)
        structured_finder = KNNAnalogueFinder(
            tuple(candidate.case.record for candidate in eligible)
        )
        structured = structured_finder.search(query.card, top_k)
        dense = dense_finder.search(query.card, structured, structured_finder, top_k)
        combined = combined_finder.search(query.card, structured, structured_finder, top_k)
        cohort_orders = {record.source_order for record in structured.cohort}
        cohort_candidates = tuple(
            candidate
            for candidate in eligible
            if candidate.case.record.source_order in cohort_orders
        )
        lexical = lexical_rank(query.narrative, cohort_candidates, top_k)
        results = {
            "structured": structured,
            "dense": dense.result,
            "lexical": KNNResult(
                lexical,
                structured.cohort,
                structured.candidate_count,
                bool(structured.cohort),
                structured.active_numeric_features,
                structured.active_boolean_features,
                (),
            ),
            "dense_structured": combined.result,
        }
        statuses = {
            "structured": "ranked" if structured.ranking_available else "unranked",
            "dense": dense.status,
            "lexical": "ranked" if structured.cohort else "empty_cohort",
            "dense_structured": combined.status,
        }
        fallback_reasons = {
            "structured": None,
            "dense": dense.fallback_reason,
            "lexical": None if structured.cohort else "empty_cohort",
            "dense_structured": combined.fallback_reason,
        }
        if structured.cohort and not structured.ranking_available:
            structured_unranked += 1
        for method, search in (("dense", dense), ("dense_structured", combined)):
            if search.fallback_reason:
                fallbacks[method][search.fallback_reason] += 1
        missing_embeddings += sum(
            candidate.case.text_sha256 not in embeddings
            for candidate in cohort_candidates
        )

        for method in RANKED_METHODS:
            neighbors = results[method].neighbors
            stats = technical[method]
            stats["queries_with_neighbors"] += bool(neighbors)
            stats["returned_neighbors"] += len(neighbors)
            stats["non_null_distances"] += sum(
                neighbor.distance is not None for neighbor in neighbors
            )
            stats["null_distances"] += sum(
                neighbor.distance is None for neighbor in neighbors
            )
            for neighbor in neighbors:
                candidate = by_order.get(neighbor.record.source_order)
                if candidate is None or not is_eligible(query, candidate):
                    stats["self_exclusion_violations"] += 1
                if neighbor.record.source_order not in cohort_orders:
                    stats["cohort_filter_violations"] += 1

        actual = {target: getattr(query, target) for target in TARGETS}
        predictions = {
            "cohort_median": {
                target: known_median(structured.cohort, target) for target in TARGETS
            }
        }
        for method in RANKED_METHODS:
            predictions[method] = {
                target: neighbor_median(results[method].neighbors, target)
                for target in TARGETS
            }
        for method in METHODS:
            rows.append(
                {
                    "method": method,
                    "article_part": query.article_part,
                    "outcome": query.outcome,
                    "actual": actual,
                    "predicted": predictions[method],
                }
            )

        review_methods = {}
        for method in RANKED_METHODS:
            analogues = []
            for neighbor in results[method].neighbors:
                candidate = by_order[neighbor.record.source_order]
                analogues.append(
                    {
                        "analogue_id": candidate.case_id,
                        "verdict_group_id": candidate.verdict_group_id,
                        "text_sha256": candidate.case.text_sha256,
                        "distance": neighbor.distance,
                        "excerpt": _excerpt(candidate.case.narrative or ""),
                        "prison_months": neighbor.record.prison_months,
                        "fine_rub": neighbor.record.fine_rub,
                    }
                )
            review_methods[method] = {
                "status": statuses[method],
                "fallback_reason": fallback_reasons[method],
                "top_analogues": analogues,
            }
        review_queries.append(
            {
                "query_id": query.case_id,
                "verdict_group_id": query.verdict_group_id,
                "text_sha256": query.text_sha256,
                "article_part": query.article_part,
                "all_article_parts": list(query.article_parts),
                "outcome": query.outcome,
                "excerpt": _excerpt(query.narrative),
                "actual": actual,
                "cohort_size": structured.candidate_count,
                "methods": review_methods,
            }
        )

    scopes = [("overall", lambda row: True)]
    for part in sorted({query.article_part for query in queries}):
        scopes.append((f"article_part:{part}", lambda row, part=part: row["article_part"] == part))
    for outcome in OUTCOMES:
        scopes.append((f"outcome:{outcome}", lambda row, outcome=outcome: row["outcome"] == outcome))
    metrics: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGETS}
    for target in TARGETS:
        for scope, predicate in scopes:
            for method in METHODS:
                selected = [
                    row for row in rows if row["method"] == method and predicate(row)
                ]
                metrics[target].append(
                    {"scope": scope, "method": method, **regression_metrics(selected, target)}
                )

    summary = {
        "protocol": {
            "queries": "confirmed_risk_examples_leave_one_verdict_group_out",
            "top_k": top_k,
            "dense_weight": dense_weight,
            "article_part": "first_listed_part",
            "prediction": "median_of_known_targets_only",
            "fine_error_space": "log1p",
        },
        "counts": {
            "evaluated_queries": len(queries),
            "evaluated_verdict_groups": len({query.verdict_group_id for query in queries}),
            "queries_with_cohort": sum("cohort_size" in item and item["cohort_size"] > 0 for item in review_queries),
            "queries_without_cohort": sum(item["cohort_size"] == 0 for item in review_queries),
            "known_targets": {
                target: sum(getattr(query, target) is not None for query in queries)
                for target in TARGETS
            },
        },
        "metrics": metrics,
        "technical": {
            "ranking": technical,
            "dense_fallbacks": {
                method: _counter_payload(counter) for method, counter in fallbacks.items()
            },
            "structured_unranked_queries": structured_unranked,
            "missing_cohort_embeddings": missing_embeddings,
        },
    }
    review_pack = {
        "schema_version": 1,
        "confidentiality": "local_manual_review_only_do_not_publish",
        "model": {"repository": encoder.model_id, "revision": encoder.revision},
        "protocol": summary["protocol"],
        "queries": review_queries,
    }
    return summary, review_pack


def _format_number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _format_coverage(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _metric_row(row: dict[str, Any], digits: int) -> str:
    return (
        f"| {row['scope']} | {METHOD_LABELS[row['method']]} | {row['n_queries']} | "
        f"{row['n_known']} | {row['n_scored']} | {_format_coverage(row['coverage'])} | "
        f"{_format_number(row['mae'], digits)} | {_format_number(row['rmse'], digits)} | "
        f"{_format_number(row['median_absolute_error'], digits)} |"
    )


def _overall(summary: dict[str, Any], target: str, method: str) -> dict[str, Any]:
    return next(
        row
        for row in summary["metrics"][target]
        if row["scope"] == "overall" and row["method"] == method
    )


def render_report(
    summary: dict[str, Any],
    risk_records: int,
    skipped_queries: list[dict[str, str]],
    retrieval: dict[str, int],
) -> str:
    counts = summary["counts"]
    lines = [
        "# Независимая оценка kNN",
        "",
        "## Протокол и итоговые counts",
        "",
        f"- Risk-записей на входе: **{risk_records}**; после групповой дедупликации и проверки схемы оценено **{counts['evaluated_queries']}** запросов из **{counts['evaluated_verdict_groups']}** групп приговоров.",
        f"- Пропущено запросов: **{len(skipped_queries)}**; причины: "
        + ", ".join(f"{reason}={count}" for reason, count in sorted(Counter(item["reason"] for item in skipped_queries).items()))
        + ".",
        f"- Retrieval: **{retrieval['source_records']}** исходных, **{retrieval['parser_valid_records']}** валидных для production parser, **{retrieval['parser_skipped_records']}** пропущено parser; после дедупликации осталось **{retrieval['evaluation_documents']}** документов.",
        f"- Непустая cohort: **{counts['queries_with_cohort']}** из {counts['evaluated_queries']}; пустая: **{counts['queries_without_cohort']}**.",
        f"- Известные targets среди оценённых запросов: prison_months=**{counts['known_targets']['prison_months']}**, fine_rub=**{counts['known_targets']['fine_rub']}**.",
        "- Для каждого запроса исключены собственный hash текста, идентификатор дела и вся его verdict-группа. Дубликаты удалены также внутри candidate-side.",
        "- Все методы используют одинаковую жёсткую cohort по первой перечисленной части статьи и процессуальным режимам; `null` не заменяется на `false`. Для ranking выбрано top_k=5.",
        "- Cohort median использует всю cohort; остальные методы — медиану известных targets среди top-5. Неизвестный target не превращается в ноль. Coverage считается от запросов с известным фактическим target.",
        "- Для fine_rub ошибки рассчитаны в пространстве `log1p(руб.)`; для prison_months — в месяцах. Cosine и combined distance являются только техническими мерами ranking, не юридической релевантностью.",
        "",
    ]
    if counts.get("by_year"):
        lines.extend(
            [
                "## Counts по годам",
                "",
                "Risk source — все подтверждённые записи; evaluated — после групповой дедупликации и проверки схемы. Retrieval source приведён до parser-фильтрации.",
                "",
                "| Год | Risk source | Evaluated | С cohort | Prison known | Fine known | Retrieval source |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for year, row in counts["by_year"].items():
            lines.append(
                f"| {year} | {row['risk_source']} | {row['evaluated_queries']} | "
                f"{row['queries_with_cohort']} | {row['prison_known']} | {row['fine_known']} | "
                f"{row['retrieval_source']} |"
            )
        lines.append("")
    lines.extend(["## Общие метрики", ""])
    for target, title, digits in (
        ("prison_months", "prison_months, месяцы", 3),
        ("fine_rub", "fine_rub, ошибки log1p(руб.)", 4),
    ):
        lines.extend(
            [
                f"### {title}",
                "",
                "| Метод | Запросов | Известных targets | Покрыто | Coverage | MAE | RMSE | Median AE |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["metrics"][target]:
            if row["scope"] == "overall":
                label = METHOD_LABELS[row["method"]]
                lines.append(
                    _metric_row(row, digits).replace(
                        f"| overall | {label} |", f"| {label} |", 1
                    )
                )
        lines.append("")

    lines.extend(["## Метрики по частям статьи и outcomes", ""])
    for target, title, digits in (
        ("prison_months", "prison_months, месяцы", 3),
        ("fine_rub", "fine_rub, ошибки log1p(руб.)", 4),
    ):
        lines.extend(
            [
                f"### {title}",
                "",
                "| Срез | Метод | Запросов | Известных targets | Покрыто | Coverage | MAE | RMSE | Median AE |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["metrics"][target]:
            if row["scope"] != "overall":
                lines.append(_metric_row(row, digits))
        lines.append("")

    lines.extend(
        [
            "## Техническая проверка ranking",
            "",
            "| Метод | Запросов с neighbors | Возвращено neighbors | Distance non-null | Distance null | Self exclusions нарушено | Cohort filter нарушено |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in RANKED_METHODS:
        row = summary["technical"]["ranking"][method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['queries_with_neighbors']} | {row['returned_neighbors']} | "
            f"{row['non_null_distances']} | {row['null_distances']} | "
            f"{row['self_exclusion_violations']} | {row['cohort_filter_violations']} |"
        )
    lines.extend(
        [
            "",
            f"- Structured без активных числовых признаков (unranked fallback): **{summary['technical']['structured_unranked_queries']}** запросов; у такого списка допустимы `distance=null`.",
            f"- Dense fallback: **{json.dumps(summary['technical']['dense_fallbacks']['dense'], ensure_ascii=False)}**; dense+structured: **{json.dumps(summary['technical']['dense_fallbacks']['dense_structured'], ensure_ascii=False)}**.",
            f"- Отсутствующих embeddings внутри cohort: **{summary['technical']['missing_cohort_embeddings']}**.",
            "",
            "## Сравнение и ограничения",
            "",
        ]
    )
    for target, label, digits in (
        ("prison_months", "prison_months", 3),
        ("fine_rub", "fine_rub", 4),
    ):
        dense = _overall(summary, target, "dense")
        structured = _overall(summary, target, "structured")
        baseline = _overall(summary, target, "cohort_median")
        lines.append(
            f"- По {label} MAE dense={_format_number(dense['mae'], digits)}, structured={_format_number(structured['mae'], digits)}, cohort median={_format_number(baseline['mae'], digits)}; это описание малой выборки, а не доказанное преимущество."
        )
    lines.extend(
        [
            "- Выборка мала и нерепрезентативна: только подтверждённые полные пары одного пилота; внешней, временной или региональной проверки нет. Число известных targets существенно меньше числа запросов, поэтому RMSE неустойчив к отдельным делам.",
            "- Пустые cohort и срезы с малым `n_known` означают отсутствие доказанного покрытия, а не нулевую ошибку. Прочерк в метрике означает отсутствие наблюдений.",
            "- Одинаковые candidate-side и hard filters делают сравнение честным технически, но не устраняют ошибки извлечения признаков и targets из текстов.",
            "- Dense+structured использует заранее заданный production вес 0.7; подбор веса по этой выборке не проводился. Доверительные интервалы и тесты значимости не дают надёжного вывода при таких counts.",
            "- Локальный review pack с идентификаторами и excerpts предназначен только для ручной проверки lead. Он не публикуется; автоматическая оценка не заменяет эту проверку.",
            "- **Production quality не доказано**: нужны больше независимых дел с известными targets, групповая внешняя/временная проверка и ручная оценка юридической уместности аналогов.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-controlled processed kNN evaluation")
    parser.add_argument("--risk", type=Path, default=Path("corpus/processed/risk_examples.jsonl"))
    parser.add_argument("--retrieval", type=Path, default=Path("corpus/processed/retrieval_cases.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("corpus/processed/dense-index.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("corpus/processed/model-card.json"))
    parser.add_argument("--report", type=Path, default=Path("corpus/processed/evaluation.md"))
    parser.add_argument("--review-pack", type=Path, default=Path("work/knn-review-pack.json"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    risk_rows = _read_jsonl(args.risk)
    retrieval_rows = _read_jsonl(args.retrieval)
    risk_records = len(risk_rows)
    queries, skipped_queries = load_risk_queries(args.risk)
    candidates, retrieval = load_retrieval_candidates(args.retrieval)
    encoder = DenseEncoder(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
    summary, review_pack = evaluate(
        queries,
        candidates,
        args.index,
        args.manifest,
        encoder,
    )
    summary["counts"]["risk_records"] = risk_records
    summary["counts"]["unique_risk_verdict_groups_before_dedup"] = len(
        {row["verdict_group_id"] for row in risk_rows}
    )
    summary["counts"]["skipped_queries"] = dict(
        sorted(Counter(item["reason"] for item in skipped_queries).items())
    )
    summary["counts"]["retrieval"] = retrieval

    def year_of(row: dict[str, Any]) -> str:
        value = row.get("source_year")
        return str(value) if type(value) is int else "unknown"

    years = sorted(
        {year_of(row) for row in risk_rows + retrieval_rows}
        | {str(query.source_year) if query.source_year is not None else "unknown" for query in queries}
    )
    query_reviews = list(zip(queries, review_pack["queries"], strict=True))
    by_year = {}
    for year in years:
        selected = [
            (query, review)
            for query, review in query_reviews
            if (str(query.source_year) if query.source_year is not None else "unknown") == year
        ]
        by_year[year] = {
            "risk_source": sum(year_of(row) == year for row in risk_rows),
            "retrieval_source": sum(year_of(row) == year for row in retrieval_rows),
            "evaluated_queries": len(selected),
            "queries_with_cohort": sum(review["cohort_size"] > 0 for _, review in selected),
            "prison_known": sum(query.prison_months is not None for query, _ in selected),
            "fine_known": sum(query.fine_rub is not None for query, _ in selected),
        }
    summary["counts"]["by_year"] = by_year
    review_pack["counts_by_year"] = by_year
    args.review_pack.parent.mkdir(parents=True, exist_ok=True)
    args.review_pack.write_text(
        json.dumps(review_pack, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_report(summary, risk_records, skipped_queries, retrieval), encoding="utf-8"
    )
    public_summary = {
        "counts": summary["counts"],
        "overall": {
            target: [
                row for row in rows if row["scope"] == "overall"
            ]
            for target, rows in summary["metrics"].items()
        },
        "technical": summary["technical"],
    }
    print(json.dumps(public_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
