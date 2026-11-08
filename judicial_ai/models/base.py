from typing import Protocol

from judicial_ai.schemas.analysis import AppealRisk


class AppealRiskModel(Protocol):
    def predict_appeal_risk(self, narrative: str | None) -> AppealRisk: ...

    def info(self) -> dict[str, object]: ...
