import csv
import logging
import time
import uuid

from judicial_ai import __version__
from judicial_ai.analogues.knn import KNNAnalogueFinder
from judicial_ai.analogues.summary import summarize_punishment
from judicial_ai.case_card import AnalysisCard, CaseCard
from judicial_ai.config import Settings
from judicial_ai.corpus.csv_repository import CorpusError, CsvCorpusRepository
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


class AnalysisService:
    def __init__(
        self,
        settings: Settings,
        records: tuple[CaseRecord, ...],
        risk_model: AppealRiskModel,
        corpus_available: bool = True,
        skipped_records: int = 0,
    ):
        self.settings = settings
        self._finder = KNNAnalogueFinder(records)
        self._risk_model = risk_model
        self._legal_validator = LegalValidator()
        self._corpus_available = corpus_available
        self._skipped_records = skipped_records

    @classmethod
    def from_settings(cls, settings: Settings) -> "AnalysisService":
        records: tuple[CaseRecord, ...] = ()
        corpus_available = True
        skipped = 0
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
        return cls(settings, records, risk_model, corpus_available, skipped)

    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        started = time.monotonic()
        analysis_id = str(uuid.uuid4())
        card = AnalysisCard(
            case=CaseCard.from_input(request.case),
            draft=request.draft_punishment,
        )
        legal_checks = self._legal_validator.validate(card)
        result = self._finder.search(
            card.case, request.top_k or self.settings.default_top_k
        )
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
            *result.warnings,
        ]
        if not self._corpus_available:
            warnings.append("Корпус недоступен; поиск аналогов не выполнялся.")
        if self._skipped_records:
            warnings.append(
                f"При загрузке корпуса пропущено записей с некорректными полями: {self._skipped_records}."
            )
        risk = self._risk_model.predict_appeal_risk(card.case.narrative)
        response = AnalysisResponse(
            analysis_id=analysis_id,
            legal_checks=legal_checks,
            analogues=analogues,
            punishment_summary=summarize_punishment(
                result.cohort, len(result.neighbors)
            ),
            appeal_risk=risk,
            warnings=warnings,
            model_info=ModelInfo(
                engine_version=__version__,
                knn={
                    **self._knn_info(),
                    "candidate_count": result.candidate_count,
                    "ranking_available": result.ranking_available,
                    "statistics_scope": "cohort",
                    "active_numeric_features": list(result.active_numeric_features),
                    "active_boolean_features": list(result.active_boolean_features),
                    "distance_metric": (
                        "euclidean_robust_scaled"
                        if result.ranking_available
                        else None
                    ),
                },
                bert=self._risk_model.info(),
            ),
        )
        logger.info(
            "Analysis completed: analysis_id=%s candidates=%d returned=%d legal_errors=%d risk=%s duration_ms=%.1f",
            analysis_id,
            result.candidate_count,
            len(analogues),
            sum(check.status == "error" for check in legal_checks),
            risk.status,
            (time.monotonic() - started) * 1000,
        )
        return response

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            engine_version=__version__,
            knn=self._knn_info(),
            bert=self._risk_model.info(),
        )

    def health(self) -> dict[str, object]:
        return {
            "status": "ok" if self._corpus_available and self._finder.record_count else "degraded",
            "version": __version__,
            "corpus": {
                "status": "ready" if self._corpus_available and self._finder.record_count else "unavailable",
                "records": self._finder.record_count,
            },
            "appeal_risk": self._risk_model.info()["risk_status"],
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
