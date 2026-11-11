from tools import recover_rostov_150 as mod


def candidate():
    return {
        "first_case_number": "1-171/2018",
        "first_court": "шахтинский городской суд",
        "first_decision_date": "2018-08-29",
    }


def verdict(header: str | None = None) -> str:
    body = " рассмотрел дело. УСТАНОВИЛ: обстоятельства. " + "Доказательства исследованы. " * 60
    return (header or "Дело № 1-171/2018 ПРИГОВОР Именем Российской Федерации 29 августа 2018 года г. Шахты Шахтинский городской суд") + body + "ПРИГОВОРИЛ: назначить наказание."


def test_full_verdict_accepts_numeric_and_textual_header_dates():
    assert mod.sentence_errors(verdict(), candidate()) == []
    numeric = verdict("Дело № 1-171/2018 ПРИГОВОР 29.08.2018 Шахтинский городской суд")
    assert mod.sentence_errors(numeric, candidate()) == []


def test_full_verdict_requires_exact_identity_and_date_before_court():
    wrong_number = verdict().replace("1-171/2018", "1-172/2018")
    assert "sentence_number_mismatch" in mod.sentence_errors(wrong_number, candidate())
    wrong_court = verdict().replace("Шахтинский городской суд", "Морозовский районный суд")
    assert "sentence_court_mismatch" in mod.sentence_errors(wrong_court, candidate())
    late_date = verdict("Дело № 1-171/2018 ПРИГОВОР Шахтинский городской суд 29 августа 2018 года")
    assert "sentence_date_not_before_court" in mod.sentence_errors(late_date, candidate())


def test_full_verdict_requires_motivation_and_operative_parts():
    assert "motivation_missing" in mod.sentence_errors(verdict().replace("УСТАНОВИЛ", "ИЗЛОЖИЛ"), candidate())
    assert "operative_missing" in mod.sentence_errors(verdict().replace("ПРИГОВОРИЛ", "ПОСТАНОВИЛ"), candidate())


def test_procedural_act_is_not_accepted_as_trial_verdict():
    procedural = verdict().replace("ПРИГОВОР Именем", "ОПРЕДЕЛЕНИЕ Именем")
    assert "trial_verdict_header_missing" in mod.sentence_errors(procedural, candidate())


def test_short_quote_is_not_a_full_verdict():
    quote = "Дело № 1-171/2018 ПРИГОВОР 29.08.2018 Шахтинский городской суд УСТАНОВИЛ ПРИГОВОРИЛ"
    assert mod.sentence_errors(quote, candidate()) == ["text_too_short"]


def test_city_word_variant_and_text_date_are_accepted():
    specific = {
        "first_case_number": "1-15/2025",
        "first_court": "ленинский районный суд г ростова на дону",
        "first_decision_date": "2025-02-24",
    }
    text = verdict("уголовное дело № 1-15/2025 ПРИГОВОР Именем Российской Федерации г. Ростов-на-Дону 24 февраля 2025 года Ленинский районный суд города Ростова-на-Дону")
    assert mod.sentence_errors(text, specific) == []
    wrong_day = text.replace("24 февраля 2025", "23 февраля 2025")
    assert "sentence_date_mismatch" in mod.sentence_errors(wrong_day, specific)


def test_sudact_parser_separates_metadata_title_from_full_text():
    page = "<h1>Приговор от 23 февраля</h1><hr class=\"hr-h1\">Полный текст от 24 февраля<br>УСТАНОВИЛ<br>ПРИГОВОРИЛ<h3>Суд:</h3>Метаданные"
    parsed = mod.parse_sudact(page)
    assert parsed["title"] == "Приговор от 23 февраля"
    assert parsed["text"] == "Полный текст от 24 февраля\nУСТАНОВИЛ\nПРИГОВОРИЛ"


def test_new_trial_disposition_is_detected():
    assert mod.new_trial_disposition("ПОСТАНОВИЛ: приговор отменить, дело направить на новое судебное разбирательство.")
