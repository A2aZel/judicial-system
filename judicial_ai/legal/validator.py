from judicial_ai.case_card import AnalysisCard
from judicial_ai.schemas.analysis import LegalCheck


class LegalValidator:
    """Checks structural completeness and consistency, not statutory compliance."""

    def validate(self, card: AnalysisCard) -> list[LegalCheck]:
        checks: list[LegalCheck] = []
        draft = card.draft
        case = card.case
        has_primary_punishment = any(
            value is not None and value > 0
            for value in (
                draft.prison_months,
                draft.fine_rub,
                draft.fine_multiple,
                draft.ban_years,
            )
        )
        checks.append(
            LegalCheck(
                code="primary_punishment_present",
                status="passed" if has_primary_punishment else "error",
                message=(
                    "Указано хотя бы одно основное наказание."
                    if has_primary_punishment
                    else "Не указано основное наказание; конфискация сама по себе не заменяет эту проверку."
                ),
            )
        )
        conditional_consistent = not draft.conditional or (
            draft.prison_months is not None and draft.prison_months > 0
        )
        checks.append(
            LegalCheck(
                code="conditional_requires_prison_term",
                status="passed" if conditional_consistent else "error",
                message=(
                    "Условный режим и срок лишения свободы формально согласованы."
                    if conditional_consistent
                    else "Для условного режима нужен положительный срок лишения свободы."
                ),
            )
        )
        fine_consistent = not (draft.fine_rub is not None and draft.fine_multiple is not None)
        checks.append(
            LegalCheck(
                code="fine_representation_unambiguous",
                status="passed" if fine_consistent else "error",
                message=(
                    "Форма записи штрафа не противоречива."
                    if fine_consistent
                    else "Укажите штраф либо в рублях, либо в кратной величине, чтобы избежать неоднозначности."
                ),
            )
        )
        if case.art_69 and case.episodes == 1:
            episode_status = "warning"
            episode_message = "Указан режим ст. 69, но число эпизодов равно одному; требуется ручная проверка."
        else:
            episode_status = "not_checked" if case.episodes is None else "passed"
            episode_message = (
                "Число эпизодов не задано; соответствие режиму ст. 69 не проверялось."
                if case.episodes is None
                else "Явного формального противоречия между числом эпизодов и режимом ст. 69 не обнаружено."
            )
        checks.append(
            LegalCheck(
                code="episode_regime_consistency",
                status=episode_status,
                message=episode_message,
            )
        )
        checks.append(
            LegalCheck(
                code="statutory_sanction_limits",
                status="not_checked",
                message="Санкции и обязательные ограничения по конкретной редакции закона не проверялись.",
            )
        )
        return checks
