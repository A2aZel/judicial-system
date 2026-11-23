import re

from judicial_ai.factors import FactorGroup


_DISPOSITION = re.compile(r"\bп\s*р\s*и\s*г\s*о\s*в\s*о\s*р\s*и\s*л[аи]?\s*:", re.I)
_FINDINGS = re.compile(r"\bу\s*с\s*т\s*а\s*н\s*о\s*в\s*и\s*л[аи]?\s*:", re.I)
_SENTENCE_BREAK = re.compile(r"(?<=[!?;]) +|(?<=\.) +(?=[А-ЯЁ](?:[а-яё]| ))")
_QUOTED = re.compile(r'«[^»]*»|"[^"]*"')
_ATTRIBUTED = re.compile(
    r"\b(?:по (?:версии|мнению|словам|позиции|показаниям)|"
    r"согласно (?:обвинени\w*|обвинительн\w+\s+заключени\w*|показани\w*|разъяснени\w*)|"
    r"в обвинительн\w+\s+заключени\w*|"
    r"(?:прокурор\w*|защитник\w*|адвокат\w*|подсудим\w*|свидетел\w*)\s+"
    r"(?:полага\w*|счита\w*|прос\w*|утвержда\w*|заявля\w*|указ\w*|поясн\w*)|"
    r"якобы|предположительн\w*|предполагаем\w*|если)\b"
)
_UNCONFIRMED = re.compile(
    r"\b(?:не\b(?!\s+только\b)|отсутстви\w*|исключ\w*|опроверг\w*|оспарива\w*|"
    r"предполагаем\w*|возможн\w*|ходатайств\w*|необоснованн\w*|недоказан\w*|"
    r"неподтвержден\w*|неустановлен\w*|или|либо)\b"
)
_EXCEPTION = re.compile(r"\b(?:кроме|за исключением|иных|других)\b")
_RETRACTION = re.compile(
    r"\b(?:подлеж\w*\s+исключени\w*|суд\s+исключил[аи]?|"
    r"не\s+(?:призна(?:н[оаы]?|ет)|подтвер(?:дил[аи]?|жден[оаы]?)))\b"
)
_NEXT_CLAUSE = re.compile(
    r",\s*(?:(?:а(?!\s+также\b)|но|однако|при этом)\s+|"
    r"(?=(?:отягчающ|смягчающ)\w*\s+(?:наказание\s+)?обстоятельств|"
    r"обстоятельств\w*\s*,?\s*(?:отягчающ|смягчающ)))"
)
_VERB = r"(?:призна(?:ет|л|ла)|учитыва(?:ет|л|ла)|относит|отнес(?:ла)?)"
_COURT = r"суд\s+(?:(?:также|также прямо)\s+)?"
_SUBJECT = r"(?:\s+(?:подсудим\w*|осужденн\w*))?"
_LEAD = r"^\s*(?:(?:при назначении наказания|при определении вида и размера наказания)\s*,?\s+)?"
_LIST_SEPARATOR = re.compile(r"\s*,\s*|\s+(?:а также|и)\s+")
_ITEM_LEAD = re.compile(
    r"^(?:(?:а также|в том числе|искренн\w*|активн\w*|добровольн\w*|"
    r"полн\w*|частичн\w*)\s+)*"
)


def _circumstances(stem: str) -> str:
    return rf"(?:обстоятельств\w*\s*,?\s*{stem}\w*\s+наказание|{stem}\w*\s+(?:наказание\s+)?обстоятельств\w*)"


def _recognition(stem: str) -> re.Pattern[str]:
    marker = _circumstances(stem)
    return re.compile(
        rf"{_LEAD}(?:(?:(?:в качестве|к)\s+)?{marker}{_SUBJECT}\s*,?\s*{_COURT}{_VERB}|"
        rf"{_COURT}в качестве\s+{marker}{_SUBJECT}\s*,?\s*{_VERB})\b[\s:,—-]*"
    )


def _explicit_absence(stem: str) -> re.Pattern[str]:
    marker = _circumstances(stem)
    return re.compile(
        rf"{_LEAD}(?:{marker}{_SUBJECT}\s*,?\s*(?:судом\s+)?не\s+установлен[оы]|"
        rf"суд\s+не\s+установил[аи]?\s+{marker})\b"
    )


_GROUPS: dict[FactorGroup, str] = {"mitigating": "смягчающ", "aggravating": "отягчающ"}
_RECOGNITION = {group: _recognition(stem) for group, stem in _GROUPS.items()}
_NO_FACTORS = {group: _explicit_absence(stem) for group, stem in _GROUPS.items()}
_PATTERNS = {
    "mitigating": {
        "явка с повинной": r"явк[уаеи]\s+с\s+повинн\w*",
        "содействие расследованию": (
            r"(?:содействи\w*|способствовани\w*)\s+(?:раскрыти\w*|расследовани\w*)"
        ),
        "возмещение вреда": (
            r"возмещени\w*\s+(?:(?:причинен\w*|имущественн\w*|моральн\w*)\s+){0,3}(?:вред\w*|ущерб\w*)"
        ),
        "раскаяние": r"раскаяни[еяиюем]+",
        "наличие малолетних детей": r"(?:наличи\w*\s+)?малолетн\w*\s+(?:дет\w*|ребен\w*)",
        "состояние здоровья": r"состояни\w*\s+здоровья(?:\s+подсудим\w*|(?=[.!?;\s]*$))",
    },
    "aggravating": {
        "рецидив преступлений": r"рецидив\w*(?:\s+преступлен\w*)?",
        "группа лиц": r"(?:совершени\w*\s+(?:преступлен\w*\s+)?(?:в составе\s+)?)?групп(?:а|ы|ой|ою|у|е)\s+лиц",
        "особо активная роль": r"особо\s+активн\w*\s+рол[ьиью]+",
        "причинение тяжких последствий": r"причинени\w*\s+тяжк\w*\s+последстви\w*",
    },
}
_NAMED_FACTORS = {
    group: {name: re.compile(rf"\b(?:{pattern})\b") for name, pattern in patterns.items()}
    for group, patterns in _PATTERNS.items()
}


def _group_factors(
    clauses: list[str], group: FactorGroup
) -> frozenset[str] | None:
    present: set[str] = set()
    retracted: set[str] = set()
    explicit_absence = False
    factors = _NAMED_FACTORS[group]
    for clause in clauses:
        if _NO_FACTORS[group].search(clause) and not _EXCEPTION.search(clause):
            explicit_absence = True
        if _RETRACTION.search(clause):
            retracted.update(name for name, pattern in factors.items() if pattern.search(clause))
        anchor = _RECOGNITION[group].search(clause)
        if anchor is None or _UNCONFIRMED.search(clause):
            continue
        recognized = clause[anchor.end():]
        exception = _EXCEPTION.search(recognized)
        if exception:
            recognized = recognized[:exception.start()]
        for item in _LIST_SEPARATOR.split(recognized):
            item = _ITEM_LEAD.sub("", item.strip())
            present.update(name for name, pattern in factors.items() if pattern.match(item))
    if explicit_absence:
        return None if present else frozenset()
    return frozenset(present - retracted) or None


def extract_sentencing_factors(
    narrative: str,
) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    dispositions = list(_DISPOSITION.finditer(narrative))
    if len(dispositions) != 1:
        return None, None
    headings = list(_FINDINGS.finditer(narrative, 0, dispositions[0].start()))
    if len(headings) != 1:
        return None, None
    findings = narrative[headings[0].end():dispositions[0].start()]
    findings = " ".join(_QUOTED.sub(" [цитата] ", findings).split())
    clauses = []
    for sentence in _SENTENCE_BREAK.split(findings):
        normalized = sentence.lower().replace("ё", "е")
        if not _ATTRIBUTED.search(normalized) and not any(mark in normalized for mark in '«»"'):
            clauses.extend(_NEXT_CLAUSE.split(normalized))
    return _group_factors(clauses, "mitigating"), _group_factors(clauses, "aggravating")
