from dataclasses import dataclass

from judicial_ai.factors import canonical_factors
from judicial_ai.schemas.case import CaseInput, DraftPunishment


@dataclass(frozen=True, slots=True)
class CaseCard:
    article_part: str
    amount_rub: float | None
    episodes: int | None
    special_order: bool
    pretrial_agreement: bool
    art_64: bool
    art_69: bool
    mitigating_count: int
    aggravating_count: int
    narrative: str | None
    mitigating_factors: frozenset[str] = frozenset()
    aggravating_factors: frozenset[str] = frozenset()

    @classmethod
    def from_input(cls, case: CaseInput) -> "CaseCard":
        return cls(
            article_part=case.article_part,
            amount_rub=case.amount_rub,
            episodes=case.episodes,
            special_order=case.special_order,
            pretrial_agreement=case.pretrial_agreement,
            art_64=case.art_64,
            art_69=case.art_69,
            mitigating_count=len(case.mitigating),
            aggravating_count=len(case.aggravating),
            narrative=case.narrative,
            mitigating_factors=canonical_factors(case.mitigating, "mitigating"),
            aggravating_factors=canonical_factors(case.aggravating, "aggravating"),
        )


@dataclass(frozen=True, slots=True)
class AnalysisCard:
    case: CaseCard
    draft: DraftPunishment
