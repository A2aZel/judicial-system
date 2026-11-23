import csv
import logging
import time
import uuid
from pathlib import Path

from judicial_ai import __version__
from judicial_ai.analogues.dense import DenseAnalogueFinder, DenseEncoder, DenseSearch
from judicial_ai.analogues.knn import KNNAnalogueFinder
from judicial_ai.analogues.recommendation import recommend_punishment
from judicial_ai.analogues.summary import summarize_punishment
from judicial_ai.case_card import AnalysisCard, CaseCard
from judicial_ai.config import Settings
from judicial_ai.corpus.csv_repository import CorpusError, CsvCorpusRepository
from judicial_ai.corpus.processed_repository import (
    ProcessedRepository,
    ProcessedRepositoryError,
)
from judicial_ai.corpus.records import CaseRecord
from judicial_ai.legal.validator import LegalValidator
from judicial_ai.models.base import AppealRiskModel
from judicial_ai.models.bert import BertModelAdapter
from judicial_ai.schemas.analysis import (
    Analogue,
    AnalysisRequest,
    AnalysisResponse,
    ModelInfo,
)

logger = logging.getLogger(__name__)
SYNTHETIC_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "synthetic_analysis.csv"


class AnalysisService:
    def __init__(
        self,
        settings: Settings,
        records: tuple[CaseRecord, ...],
        risk_model: AppealRiskModel,
        corpus_available: bool = True,
        skipped_records: int = 0,
        dense_finder: DenseAnalogueFinder | None = None,
        startup_warnings: tuple[str, ...] = (),
        corpus_source: str = "analysis_csv",
    ):
        self.settings = settings
        self._finder = KNNAnalogueFinder(records)
        self._risk_model = risk_model
        self._legal_validator = LegalValidator()
        self._corpus_available = corpus_available
        self._skipped_records = skipped_records
        self._dense_finder = dense_finder
        self._startup_warnings = startup_warnings
        self._corpus_source = corpus_source
        self._recommendation_source = (
            "synthetic"
            if corpus_available and corpus_source == "analysis_csv"
            and settings.corpus_path.resolve() == SYNTHETIC_FIXTURE.resolve()
            else "case_corpus"
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "AnalysisService":
        records: tuple[CaseRecord, ...] = ()
        corpus_available = True
        skipped = 0
        dense_finder = None
        startup_warnings: list[str] = []
        corpus_source = "analysis_csv"
        use_processed = settings.processed_repository_path.is_file() and (
            "processed_repository_path" in settings.model_fields_set
            or settings.corpus_path == Settings.model_fields["corpus_path"].default
        )
        if use_processed:
            try:
                repository = ProcessedRepository(settings.processed_repository_path)
                if not repository.records:
                    raise ProcessedRepositoryError("processed repository has no valid records")
                records = repository.records
                skipped = repository.skipped_records
                corpus_source = "processed_jsonl"
                if settings.dense_enabled:
                    encoder = DenseEncoder(
                        model_id=settings.dense_model_id,
                        revision=settings.dense_model_revision,
                        cache_dir=settings.bert_cache_dir,
                        batch_size=settings.dense_batch_size,
                        max_tokens=settings.dense_max_tokens,
                    )
                    dense_finder = DenseAnalogueFinder(
                        repository.cases,
                        settings.dense_index_path,
                        settings.dense_manifest_path,
                        encoder,
                        dense_weight=settings.dense_weight,
                    )
            except (ProcessedRepositoryError, OSError, UnicodeError) as exc:
                logger.warning("Processed repository unavailable: %s", type(exc).__name__)
                startup_warnings.append(
                    "Processed repository недоступен или отклонён; использован analysis.csv fallback."
                )
                records = ()
                skipped = 0
        if corpus_source == "analysis_csv":
            try:
                repository = CsvCorpusRepository(settings.corpus_path)
                records = repository.records
                skipped = repository.skipped_records
            except (CorpusError, OSError, UnicodeError, csv.Error) as exc:
                corpus_available = False
                logger.warning("Corpus unavailable: %s", type(exc).__name__)
        risk_model = BertModelAdapter(
            model_id=settings.bert_model_id,
            revision=settings.bert_revision,
            risk_checkpoint=settings.bert_risk_checkpoint,
            cache_dir=settings.bert_cache_dir,
            risk_enabled=settings.bert_risk_enabled,
            approved_checkpoint_sha256=settings.bert_risk_checkpoint_sha256,
        )
        return cls(
            settings,
            records,
            risk_model,
            corpus_available,
            skipped,
            dense_finder,
            tuple(startup_warnings),
            corpus_source,
        )

    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        started = time.monotonic()
        analysis_id = str(uuid.uuid4())
        case = CaseCard.from_input(request.case)
        legal_checks = (
            self._legal_validator.validate_case(case)
            if request.draft_punishment is None
            else self._legal_validator.validate(AnalysisCard(case, request.draft_punishment))
        )
        top_k = request.top_k or self.settings.default_top_k
        structured = self._finder.search(case, top_k)
        dense_search = (
            self._dense_finder.search(case, structured, self._finder, top_k)
            if self._dense_finder is not None
            else None
        )
        result = dense_search.result if dense_search else structured
        analogues = [
            Analogue(
                distance=(
                    round(neighbor.distance, 6)
                    if neighbor.distance is not None
                    else None
                ),
                prison_months=neighbor.record.prison_months,
                fine_rub=neighbor.record.fine_rub,
                fine_multiple=neighbor.record.fine_multiple,
                ban_on_positions=neighbor.record.ban_on_positions,
                ban_years=neighbor.record.ban_years,
                conditional=neighbor.record.conditional,
                confiscation=neighbor.record.confiscation,
            )
            for neighbor in result.neighbors
        ]
        warnings = [
            "Результат ограничен формальными проверками и поиском похожих дел; это не полная юридическая валидация и не доказательство причинной связи.",
            "Поле conditional получено из автоматически извлечённого признака art_73 и требует проверки по первоисточнику.",
            *self._startup_warnings,
            *result.warnings,
        ]
        if any(not record.recommendation_facts_trusted for record in structured.cohort):
            warnings.append(
                "Сумма, число эпизодов и обстоятельства из полных текстов приговоров "
                "могут требовать проверки. Эти поля используются для просмотра аналогов."
            )
        if not self._corpus_available:
            warnings.append("Корпус недоступен; поиск аналогов не выполнялся.")
        if self._skipped_records:
            warnings.append(
                f"При загрузке корпуса пропущено записей с некорректными полями: {self._skipped_records}."
            )
        risk = self._risk_model.predict_appeal_risk(case.narrative)
        response = AnalysisResponse(
            analysis_id=analysis_id,
            legal_checks=legal_checks,
            analogues=analogues,
            punishment_summary=summarize_punishment(
                result.cohort, len(result.neighbors)
            ),
            recommendation=recommend_punishment(
                case, structured.cohort, self._recommendation_source
            ),
            appeal_risk=risk,
            warnings=warnings,
            model_info=ModelInfo(
                engine_version=__version__,
                knn=self._analysis_knn_info(result, dense_search),
                bert=self._risk_model.info(),
                dense=self._dense_info(),
            ),
        )
        logger.info(
            "Analysis completed: analysis_id=%s candidates=%d returned=%d legal_errors=%d risk=%s dense=%s duration_ms=%.1f",
            analysis_id,
            result.candidate_count,
            len(analogues),
            sum(check.status == "error" for check in legal_checks),
            risk.status,
            dense_search.status if dense_search else "not_configured",
            (time.monotonic() - started) * 1000,
        )
        return response

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            engine_version=__version__,
            knn=self._knn_info(),
            bert=self._risk_model.info(),
            dense=self._dense_info(),
        )

    def health(self) -> dict[str, object]:
        return {
            "status": "ok" if self._corpus_available and self._finder.record_count else "degraded",
            "version": __version__,
            "corpus": {
                "status": "ready" if self._corpus_available and self._finder.record_count else "unavailable",
                "records": self._finder.record_count,
                "source": self._corpus_source,
            },
            "appeal_risk": self._risk_model.info()["risk_status"],
            "dense": self._dense_info()["status"],
        }

    def _analysis_knn_info(
        self, result, dense_search: DenseSearch | None
    ) -> dict[str, object]:
        return {
            **self._knn_info(),
            "candidate_count": result.candidate_count,
            "ranking_available": result.ranking_available,
            "fallback_reason": (
                dense_search.fallback_reason
                if dense_search
                else ("no_active_features" if not result.ranking_available else None)
            ),
            "statistics_scope": "cohort",
            "active_numeric_features": list(result.active_numeric_features),
            "active_boolean_features": list(result.active_boolean_features),
            "distance_metric": (
                dense_search.distance_metric
                if dense_search
                else ("euclidean_robust_scaled" if result.ranking_available else None)
            ),
        }

    def _knn_info(self) -> dict[str, object]:
        return {
            "status": "ready" if self._corpus_available and self._finder.record_count else "unavailable",
            "algorithm": "deterministic_knn",
            "loaded_records": self._finder.record_count,
            "hard_filters": [
                "article_part",
                "special_order",
                "pretrial_agreement",
                "art_64",
                "art_69",
            ],
        }

    def _dense_info(self) -> dict[str, object]:
        if self._dense_finder is not None:
            return self._dense_finder.info()
        reason = "disabled" if not self.settings.dense_enabled else "processed_repository_not_used"
        return {
            "repository": self.settings.dense_model_id,
            "revision": self.settings.dense_model_revision,
            "status": "disabled" if not self.settings.dense_enabled else "repository_unavailable",
            "encoder_status": "lazy",
            "index_status": "unavailable",
            "fallback_reason": reason,
            "distance_metric": None,
            "dense_weight": self.settings.dense_weight,
        }
