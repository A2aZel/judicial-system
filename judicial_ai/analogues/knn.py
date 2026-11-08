import math
import statistics
from dataclasses import dataclass

from judicial_ai.case_card import CaseCard
from judicial_ai.corpus.records import CaseRecord

NUMERIC_FEATURES = (
    "amount_rub",
    "episodes",
    "mitigating_count",
    "aggravating_count",
)
BOOLEAN_FEATURES = (
    "special_order",
    "pretrial_agreement",
    "art_64",
    "art_69",
)


@dataclass(frozen=True, slots=True)
class Neighbor:
    record: CaseRecord
    distance: float | None


@dataclass(frozen=True, slots=True)
class KNNResult:
    neighbors: tuple[Neighbor, ...]
    cohort: tuple[CaseRecord, ...]
    candidate_count: int
    ranking_available: bool
    active_numeric_features: tuple[str, ...]
    active_boolean_features: tuple[str, ...]
    warnings: tuple[str, ...]


class KNNAnalogueFinder:
    def __init__(self, records: tuple[CaseRecord, ...]):
        self._records = records

    @property
    def record_count(self) -> int:
        return len(self._records)

    def search(self, case: CaseCard, top_k: int) -> KNNResult:
        candidates = tuple(
            record
            for record in self._records
            if case.article_part in record.article_parts
            and all(getattr(record, name) == getattr(case, name) for name in BOOLEAN_FEATURES)
        )
        active_numeric = tuple(
            name
            for name in NUMERIC_FEATURES
            if any(getattr(record, name) is not None for record in candidates)
        )
        warnings: list[str] = []
        if not candidates:
            warnings.append(
                "Аналоги не найдены после точной фильтрации по части статьи и процессуальным режимам."
            )
            return KNNResult((), (), 0, False, (), BOOLEAN_FEATURES, tuple(warnings))

        if not active_numeric:
            warnings.append(
                "В CSV нет активных признаков для различения дел внутри cohort: ранжирование отключено, distance=null. Список является неранжированной подвыборкой, а статистика рассчитана по всей cohort."
            )
            return KNNResult(
                tuple(Neighbor(record, None) for record in candidates[:top_k]),
                candidates,
                len(candidates),
                False,
                (),
                BOOLEAN_FEATURES,
                tuple(warnings),
            )

        scaled_candidates: dict[int, list[float]] = {}
        query_values: list[float] = []
        missing_query: list[str] = []
        imputed_records: set[str] = set()
        for name in active_numeric:
            known = [
                float(getattr(record, name))
                for record in candidates
                if getattr(record, name) is not None
            ]
            transformed = [self._transform(name, value) for value in known]
            center = statistics.median(transformed)
            scale = self._scale(transformed)
            raw_query = getattr(case, name)
            if raw_query is None:
                transformed_query = center
                missing_query.append(name)
            else:
                transformed_query = self._transform(name, float(raw_query))
            query_values.append((transformed_query - center) / scale)
            for order, record in enumerate(candidates):
                raw_value = getattr(record, name)
                if raw_value is None:
                    transformed_value = center
                    imputed_records.add(name)
                else:
                    transformed_value = self._transform(name, float(raw_value))
                scaled_candidates.setdefault(order, []).append(
                    (transformed_value - center) / scale
                )

        if missing_query:
            warnings.append(
                "Для числовых признаков "
                + ", ".join(missing_query)
                + " использована медиана сопоставимой cohort: исходное значение не задано."
            )
        if imputed_records:
            warnings.append(
                "Пропуски числовых признаков в аналогах заполнены медианой сопоставимой cohort: "
                + ", ".join(sorted(imputed_records))
                + "."
            )

        neighbors = []
        for order, record in enumerate(candidates):
            numeric_distance = sum(
                (value - query_values[index]) ** 2
                for index, value in enumerate(scaled_candidates[order])
            )
            neighbors.append(Neighbor(record, math.sqrt(numeric_distance)))
        neighbors.sort(key=lambda neighbor: (neighbor.distance, neighbor.record.source_order))
        selected = tuple(neighbors[:top_k])
        return KNNResult(
            selected,
            candidates,
            len(candidates),
            True,
            active_numeric,
            BOOLEAN_FEATURES,
            tuple(warnings),
        )

    @staticmethod
    def _transform(name: str, value: float) -> float:
        return math.log1p(value) if name == "amount_rub" else value

    @staticmethod
    def _scale(values: list[float]) -> float:
        ordered = sorted(values)
        q1 = KNNAnalogueFinder._quantile(ordered, 0.25)
        q3 = KNNAnalogueFinder._quantile(ordered, 0.75)
        iqr = q3 - q1
        if iqr > 0:
            return iqr
        deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
        return deviation if deviation > 0 else 1.0

    @staticmethod
    def _quantile(values: list[float], probability: float) -> float:
        if len(values) == 1:
            return values[0]
        position = (len(values) - 1) * probability
        lower = int(position)
        fraction = position - lower
        if lower + 1 == len(values):
            return values[lower]
        return values[lower] + fraction * (values[lower + 1] - values[lower])
