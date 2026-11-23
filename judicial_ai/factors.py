from collections.abc import Iterable
from typing import Literal


FactorGroup = Literal["mitigating", "aggravating"]

_LABELS = {
    "mitigating": frozenset({
        "явка с повинной",
        "содействие расследованию",
        "возмещение вреда",
        "раскаяние",
        "наличие малолетних детей",
        "состояние здоровья",
    }),
    "aggravating": frozenset({
        "рецидив преступлений",
        "группа лиц",
        "особо активная роль",
        "причинение тяжких последствий",
    }),
}


def canonical_factors(values: Iterable[str], group: FactorGroup) -> frozenset[str]:
    normalized = (" ".join(value.lower().replace("ё", "е").split()) for value in values)
    return _LABELS[group].intersection(normalized)


def verified_factor_list(value: object, group: FactorGroup) -> frozenset[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) > 30 or any(not isinstance(item, str) for item in value):
        raise ValueError("verified factors must be a list of strings")
    if not value:
        return frozenset()
    return canonical_factors(value, group) or None
