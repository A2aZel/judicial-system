from judicial_ai.case_card import AnalysisCard, CaseCard
from judicial_ai.legal.validator import LegalValidator
from judicial_ai.schemas.case import CaseInput, DraftPunishment


def test_consistency_errors_and_limits_are_explicit() -> None:
    case = CaseCard.from_input(
        CaseInput(
            article_part="3",
            episodes=1,
            special_order=False,
            pretrial_agreement=False,
            art_64=False,
            art_69=True,
        )
    )
    card = AnalysisCard(
        case=case,
        draft=DraftPunishment(
            fine_rub=100000,
            fine_multiple=2,
            conditional=True,
        ),
    )
    checks = {check.code: check for check in LegalValidator().validate(card)}
    assert checks["conditional_requires_prison_term"].status == "error"
    assert checks["fine_representation_unambiguous"].status == "error"
    assert checks["episode_regime_consistency"].status == "warning"
    assert checks["statutory_sanction_limits"].status == "not_checked"


def test_complete_formal_draft_passes_checked_rules() -> None:
    case = CaseCard.from_input(
        CaseInput(
            article_part="3",
            episodes=1,
            special_order=False,
            pretrial_agreement=False,
            art_64=False,
            art_69=False,
        )
    )
    card = AnalysisCard(case=case, draft=DraftPunishment(prison_months=24))
    checks = {check.code: check for check in LegalValidator().validate(card)}
    assert checks["primary_punishment_present"].status == "passed"
    assert checks["conditional_requires_prison_term"].status == "passed"
