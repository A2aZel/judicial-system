import re


MAX_AMOUNT_RUB = 1_000_000_000_000

_FINDINGS_HEADING = re.compile(r"\bу\s*с\s*т\s*а\s*н\s*о\s*в\s*и\s*л[аи]?\s*:", re.I)
_DISPOSITION_HEADING = re.compile(r"\bп\s*р\s*и\s*г\s*о\s*в\s*о\s*р\s*и\s*л[аи]?\s*:", re.I)
_SENTENCE_BREAK = re.compile(r"(?<=[!?;]) +|(?<=\.) +(?=[А-ЯЁ](?:[а-яё]| ))")
_COURT_FINDING = re.compile(
    r"(?:как установлено судом|судом установлено|суд установил|в судебном заседании установлено)\b"
)
_EVIDENCE_START = re.compile(
    r"\b(?:в\s+судебном\s+заседании|вину\s+(?:не\s+)?признал\w*|"
    r"допрошенн\w*|показани\w*|показал[аи]?|пояснил[аи]?|"
    r"обвинение\s+поддержал\w*|верси\w+\s+следствия)\b"
)
_ATTRIBUTED_OR_HYPOTHETICAL = re.compile(
    r"\b(?:по\s+(?:версии|мнению|словам|показаниям)|согласно\s+(?:обвинени\w*|"
    r"обвинительн\w+\s+(?:заключени\w+|акт\w*)|показани\w*|разъяснени\w*|постановлени\w*)|"
    r"обвиня(?:ется|лся|лась|лись)|отрицал[аи]?|оспаривал[аи]?|утверждал[аи]?|заявил[аи]?|"
    r"просил[аи]?\s+(?:считать|признать)|мог(?:ла|ли)?\s+получить|"
    r"якобы|если)\b"
)
_NEGATED_RECEIPT = re.compile(r"\bне\s+(?:получил[аи]?|получен[аоы]?|установлен[аоы]?)\b")
_RETRACTION = re.compile(
    r"\b(?:исключ(?:ил|ает|ить|ен[аоы]?)|опроверг(?:нут[аоы]?|ает)|"
    r"не\s+(?:доказан[аоы]?|подтвердил[аось]*|наш(?:ел|ла|ло|ли)\s+подтверждения))\b"
)
_PARTIAL_ACQUITTAL = re.compile(
    r"\b(?:суд\s+(?:оправдал[аи]?|оправдывает)|подсудим\w*\s+оправдан[аоы]?|"
    r"оправдать\s+подсудим\w*)\b[^.!?;]{0,125}"
    r"\b(?:в\s+(?:(?:этой|указанной|данной)\s+)?части|по\s+(?:\w+\s+)?эпизоду)\b",
    re.I,
)
_EPISODE = re.compile(r"\b(?:по\s+другому|в\s+другом|по\s+второму|во\s+втором)\s+эпизод\w*\b")

_RECEIPT = (
    r"\b(?:получил[аи]?|получен[аоы]?|получение)\s+"
    r"(?P<between>(?:(?:от|у|через|лично|непосредственно|посредством)\b[^«»\";!?]{0,150}? +)?)"
)
_OTHER_ACTION = re.compile(r"\b(?:предлож\w+|потребовал\w*|просил\w*|вымогал\w*|зарплат\w*|штраф\w*)\b")
_AMOUNT_LABEL = r"(?:в\s+(?:общей\s+)?(?:размере|сумме)|на\s+(?:общую\s+)?сумму)\s+"
_BRIBE = r"(?:взятк[уиа]|в\s+качестве\s+взятки)"
_BRIBE_AMOUNT_LABEL = rf"{_BRIBE}\s+(?:в\s+виде\s+(?:денег|денежных\s+средств)\s+)?{_AMOUNT_LABEL}"
_AMOUNT_AFTER = re.compile(rf"{_RECEIPT}{_BRIBE_AMOUNT_LABEL}")
_AMOUNT_BEFORE = re.compile(
    rf"{_RECEIPT}(?:деньги|денежные\s+средства)\s+{_AMOUNT_LABEL}"
)
_CURRENCY = r"(?:руб(?:лей|ля|ль)?\b\.?|₽)"
_RUBLES = re.compile(
    rf"(?P<whole>[0-9]{{1,3}}(?: [0-9]{{3}}){{1,4}}|[0-9]{{1,13}})(?:,(?P<kopecks>[0-9]{{2}}))?"
    rf"(?: \([а-я -]{{1,90}}\))? *{_CURRENCY}"
)
_MASKED_RUBLES = rf"(?:(?:<[^>]{{1,50}}>|\.{{2,}}|…) *)?{_CURRENCY}"
_MASKED_BRIBE_AMOUNT = re.compile(rf"\b{_BRIBE_AMOUNT_LABEL}{_MASKED_RUBLES}")
_MASKED_MONEY = re.compile(_MASKED_RUBLES)
_AS_BRIBE = re.compile(r"\s*[,–—-]?\s*в\s+качестве\s+взятки\b")
_ALTERNATIVE = re.compile(r"\s*[,;]?\s*(?:или|либо)\b")
_ADDITIONAL = re.compile(
    r"\s*[,;]?\s*(?:(?:и|а)\s+|[+–—-]\s*)?(?:(?:затем|также)\s+)?(?:еще\s+)?(?=[0-9])"
)
_BRIBE_AMOUNT_REFERENCE = re.compile(rf"\b(?:{_BRIBE}\b.*{_AMOUNT_LABEL}|сумм\w*\s+взятки\b)")


def _read_amount(match: re.Match[str]) -> float | None:
    whole = int(match["whole"].replace(" ", ""))
    amount = whole + int(match["kopecks"] or "0") / 100
    return amount if 0 < amount <= MAX_AMOUNT_RUB else None


def _amounts(sentence: str) -> list[float | None]:
    values: list[float | None] = []
    for anchor in _AMOUNT_AFTER.finditer(sentence):
        if _OTHER_ACTION.search(anchor["between"]) or _RUBLES.search(anchor["between"]):
            continue
        money = _RUBLES.match(sentence, anchor.end())
        if money is None or _has_alternative(sentence, money.end()):
            values.append(None)
        else:
            values.append(_read_amount(money))
    for anchor in _AMOUNT_BEFORE.finditer(sentence):
        if _OTHER_ACTION.search(anchor["between"]) or _RUBLES.search(anchor["between"]):
            continue
        money = _RUBLES.match(sentence, anchor.end())
        reported = money or _MASKED_MONEY.match(sentence, anchor.end())
        bribe = _AS_BRIBE.match(sentence, reported.end()) if reported else None
        if bribe:
            values.append(_read_amount(money) if money and not _has_alternative(sentence, bribe.end()) else None)
    return values


def _has_alternative(sentence: str, end: int) -> bool:
    additional = _ADDITIONAL.match(sentence, end)
    return bool(_ALTERNATIVE.match(sentence, end) or additional and _RUBLES.match(sentence, additional.end()))


def extract_bribe_amount(narrative: str) -> float | None:
    dispositions = list(_DISPOSITION_HEADING.finditer(narrative))
    if len(dispositions) != 1:
        return None
    findings_text = narrative[:dispositions[0].start()]
    headings = list(_FINDINGS_HEADING.finditer(findings_text))
    if len(headings) != 1:
        return None
    if _PARTIAL_ACQUITTAL.search(narrative, headings[0].end()):
        return None
    findings = " ".join(findings_text[headings[0].end():].split())
    opening = True
    confirmed_amount = None
    confirmed_sentence = None
    for original in _SENTENCE_BREAK.split(findings):
        sentence = original.lower().replace("ё", "е")
        if _RETRACTION.search(sentence) and _BRIBE_AMOUNT_REFERENCE.search(sentence):
            return None
        if _EVIDENCE_START.search(sentence):
            opening = False
        court = _COURT_FINDING.match(sentence)
        if not opening and court is None:
            continue
        if _ATTRIBUTED_OR_HYPOTHETICAL.search(sentence) or _NEGATED_RECEIPT.search(sentence):
            continue
        if _MASKED_BRIBE_AMOUNT.search(sentence):
            return None
        if court:
            sentence = sentence[court.end():].lstrip(" :,«")
            sentence = sentence.removeprefix("что ")
        amounts = _amounts(sentence)
        if not amounts:
            continue
        if len(amounts) != 1 or amounts[0] is None or _EPISODE.search(sentence):
            return None
        if confirmed_sentence is not None and (sentence != confirmed_sentence or court is None):
            return None
        confirmed_sentence = sentence
        confirmed_amount = amounts[0]
    return confirmed_amount
