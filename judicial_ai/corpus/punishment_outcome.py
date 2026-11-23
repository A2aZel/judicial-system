import re
from typing import Literal


PrincipalOutcome = tuple[Literal["imprisonment", "fine"] | None, float | None, bool | None]
UNKNOWN_OUTCOME: PrincipalOutcome = (None, None, None)
ARTICLE_290_PARTS = frozenset("123456")

_MAX_PRISON_MONTHS = 1200
_MAX_FINE_RUB = 1_000_000_000_000

_DISPOSITION = re.compile(r"\bп\s*р\s*и\s*г\s*о\s*в\s*о\s*р\s*и\s*л[аи]?\s*:", re.I)
_CONVICTION = re.compile(
    r"\bпризнать\s+(?:(?P<subject>.{1,100}?)\s+)?(?P<guilt>(?:не)?виновн\w*)\b"
)
_QUALIFICATION = re.compile(
    r"\s+в\s+совершении\s+преступления\s*,?\s+предусмотренного\s+"
    r"(?P<article>[^!?;]{1,105}?)\s*ук\s*рф\b"
)
_PART_BEFORE_ARTICLE = re.compile(r"(?:п\.?\s*[«\"']?[а-я][»\"']?\s+)?ч\.?\s*([1-6])\s*ст\.?\s*290")
_ARTICLE_BEFORE_PART = re.compile(r"ст\.?\s*290\s*,?\s*ч\.?\s*([1-6])")
_SUBJECT = re.compile(r"[\w .-]{1,100}")
_AMBIGUOUS_DISPOSITION = re.compile(
    r"\b(?:совокупности\s+(?:преступлений|приговоров)|ст\.?\s*(?:69|70)\s*ук\b|"
    r"окончательно\s+назначить|каждому|оправдать)\b"
)
_PRINCIPAL_ASSIGNMENT = re.compile(
    r"\s*(?:\(в\s+ред\.[^()]{1,150}\)\s*)?[,.;]?\s*(?:и\s+)?"
    r"назначить\s+(?:(?:ему|ей)\s+)?(?:основное\s+)?наказание\s+в\s+виде\s+"
    r"(?P<kind>лишения\s+свободы|штрафа)\b"
)
_ANOTHER_ASSIGNMENT = re.compile(
    r"\bназначить\s+[^;.!?]{0,120}?\b(?<!дополнительное )наказание\s+в\s+виде\b"
)

_NUMBER_WORDS = r"(?:\s*\([а-я -]{1,90}\))?"
_YEARS = rf"(?P<years>\d{{1,3}}){_NUMBER_WORDS}\s*(?:год(?:а)?|лет)\b"
_MONTHS = rf"\d{{1,4}}{_NUMBER_WORDS}\s*месяц(?:а|ев)?\b"
_PRISON_TERM = re.compile(
    rf"\s*(?:(?:сроком\s+на|на\s+срок|сроком|срок|на)\s+)?"
    rf"(?:{_YEARS}(?:\s*,?\s*(?:и\s+)?(?P<add_months>{_MONTHS}))?|(?P<months>{_MONTHS}))"
)
_FINE_AMOUNT = re.compile(
    rf"\s*(?:в\s+размере\s+)?"
    rf"(?:(?:\d+\s*[-–]?\s*|[а-я]{{1,40}})кратн\w+\s+сумм\w+\s+взятки[\s,]+"
    rf"(?:то\s+есть|что\s+составляет|что\s+соответствует)\s+)?"
    rf"(?:в\s+сумме\s+)?(?P<rubles>\d{{1,3}}(?:\s\d{{3}}){{1,4}}|\d{{1,13}})"
    rf"{_NUMBER_WORDS}\s*(?:руб(?:лей|ля|ль)?\b\.?|₽)"
)
_AMBIGUOUS_VALUE = re.compile(r"\s*[,;]?\s*(?:(?:или|либо)\b|и\s+(?:штрафа|лишения\s+свободы)\b)")
_EXTRA_DURATION = re.compile(r"\s*[,;]?\s*(?:и\s+)?\d+\s*(?:дн|сут|недел)")
_CONDITIONAL = re.compile(
    r"\bсчитать\s+(?:назначенное\s+наказание\s+)?условн(?:ым|ой|о)\b"
)
_ACTUAL = re.compile(
    r"\b(?:с\s+отбыванием(?:\s+наказания)?|отбывание\s+наказания\s+(?:определить|назначить))"
    r"\s+в\s+(?:исправительной\s+колонии|колонии[- ]поселении|тюрьме|воспитательной\s+колонии)\b"
)


def explicit_principal_outcome(
    kind: object, value: float | None, conditional: bool | None
) -> PrincipalOutcome:
    if kind is None and value is None and conditional is None:
        return UNKNOWN_OUTCOME
    if kind == "imprisonment":
        if value is not None and not 0 < value <= _MAX_PRISON_MONTHS:
            raise ValueError("invalid principal prison term")
        return "imprisonment", value, conditional
    if kind == "fine":
        if value is not None and not 0 < value <= _MAX_FINE_RUB:
            raise ValueError("invalid principal fine amount")
        return "fine", value, None
    raise ValueError("invalid principal punishment")


def extract_principal_outcome(narrative: str, article_part: str) -> PrincipalOutcome:
    headings = list(_DISPOSITION.finditer(narrative))
    if len(headings) != 1:
        return UNKNOWN_OUTCOME
    disposition = " ".join(narrative[headings[0].end():].lower().replace("ё", "е").split())
    if _AMBIGUOUS_DISPOSITION.search(disposition):
        return UNKNOWN_OUTCOME

    convictions = list(_CONVICTION.finditer(disposition))
    if len(convictions) != 1:
        return UNKNOWN_OUTCOME
    conviction = convictions[0]
    before = disposition[:conviction.start()].strip()
    after = conviction["subject"]
    if bool(before) == bool(after) or not _SUBJECT.fullmatch(before or after):
        return UNKNOWN_OUTCOME
    if re.search(r"(?:^|\s)и(?:\s|$)", before or after) or conviction["guilt"] not in {"виновным", "виновной"}:
        return UNKNOWN_OUTCOME

    qualification = _QUALIFICATION.match(disposition, conviction.end())
    if qualification is None:
        return UNKNOWN_OUTCOME
    article = qualification["article"].strip()
    part = _PART_BEFORE_ARTICLE.fullmatch(article) or _ARTICLE_BEFORE_PART.fullmatch(article)
    if part is None or part[1] != article_part:
        return UNKNOWN_OUTCOME

    assignment = _PRINCIPAL_ASSIGNMENT.match(disposition, qualification.end())
    if assignment is None or _ANOTHER_ASSIGNMENT.search(disposition, assignment.end()):
        return UNKNOWN_OUTCOME
    value_text = disposition[assignment.end():]
    if assignment["kind"] == "штрафа":
        amount = _FINE_AMOUNT.match(value_text)
        if amount is not None and not _AMBIGUOUS_VALUE.match(value_text, amount.end()):
            rubles = int(amount["rubles"].replace(" ", ""))
            if 0 < rubles <= _MAX_FINE_RUB:
                return "fine", float(rubles), None
        return UNKNOWN_OUTCOME

    term = _PRISON_TERM.match(value_text)
    if term is None or _AMBIGUOUS_VALUE.match(value_text, term.end()) or _EXTRA_DURATION.match(value_text, term.end()):
        return UNKNOWN_OUTCOME
    years = int(term["years"] or 0)
    month_text = term["add_months"] or term["months"]
    months = int(re.match(r"\d+", month_text)[0]) if month_text else 0
    prison_months = years * 12 + months
    if not 0 < prison_months <= _MAX_PRISON_MONTHS:
        return UNKNOWN_OUTCOME
    conditional = bool(_CONDITIONAL.search(value_text))
    actual = bool(_ACTUAL.search(value_text))
    return "imprisonment", float(prison_months), conditional if conditional != actual else None
