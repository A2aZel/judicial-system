import json

import pytest

from tools import build_training_dataset as mod


@pytest.fixture(autouse=True)
def isolate_regional_pairs(monkeypatch):
    monkeypatch.setattr(mod, "regional_pair_records", lambda *args, **kwargs: [])


def source_record(tmp_path, appeal_id="appeal-1", full_pair=True, outcome="changed"):
    sentence_path = tmp_path / f"{appeal_id}-sentence-source.txt"
    appeal_path = tmp_path / f"{appeal_id}-appeal-source.txt"
    sentence_path.write_text("ПРИГОВОР. Суд ПРИГОВОРИЛ к 2 годам лишения свободы. " + "мотивировка " * 30, encoding="utf-8")
    appeal_path.write_text("ОПРЕДЕЛИЛ: Приговор изменить, " + "доводы " * 30, encoding="utf-8")
    return {
        "source": "synthetic",
        "source_key": appeal_id,
        "verdict_key": "same-verdict",
        "source_year": 2024,
        "article_parts": ["5"],
        "other_articles": ["286"],
        "process": mod.process_flags("особый порядок"),
        "sentence_text": sentence_path.read_text(encoding="utf-8"),
        "appeal_text": appeal_path.read_text(encoding="utf-8"),
        "sentence_path": str(sentence_path),
        "appeal_path": str(appeal_path),
        "first": {"dateFinal": "01.02.2024", "lastHearingDate": "01.02.2024"},
        "lower": {"courtName": "Синтетический суд"},
        "full_pair": full_pair,
        "outcome": outcome,
        "status": "confirmed_pair" if full_pair else "not_started",
    }


def test_deterministic_ids_keep_multiple_appeals_on_one_verdict(tmp_path):
    first = mod.make_case(source_record(tmp_path, "appeal-1"), tmp_path)
    second = mod.make_case(source_record(tmp_path, "appeal-2"), tmp_path)
    repeated = mod.make_case(source_record(tmp_path, "appeal-1"), tmp_path)
    assert first["case_id"] != second["case_id"]
    assert first["verdict_group_id"] == second["verdict_group_id"]
    assert first["case_id"] == repeated["case_id"]
    assert first["verdict_group_id"] == repeated["verdict_group_id"]


def test_no_leakage_and_unknown_is_not_negative(tmp_path):
    case = mod.make_case(source_record(tmp_path, outcome="unknown"), tmp_path)
    retrieval = mod.retrieval_case(case)
    assert retrieval is not None
    assert "appeal" not in retrieval
    assert "отменить" not in retrieval["facts"]["narrative"]
    risk = mod.risk_case(case)
    assert risk["outcome"] == "unknown"
    assert risk["outcome"] != "negative"
    assert "appeal" not in risk["features"]


def test_unconfirmed_candidate_never_enters_risk(tmp_path):
    case = mod.make_case(source_record(tmp_path, full_pair=False), tmp_path)
    assert mod.risk_case(case) is None
    assert case["appeal"]["full_pair"] is False


def test_group_split_keeps_all_appeals_together():
    rows = [
        {"case_id": "a", "verdict_group_id": "group"},
        {"case_id": "b", "verdict_group_id": "group"},
        {"case_id": "c", "verdict_group_id": "other"},
    ]
    mod.assign_splits(rows)
    assert rows[0]["split"] == rows[1]["split"]
    assert {row["split"] for row in rows} <= {"train", "validation", "test"}


def test_article_labels_are_not_invented(tmp_path):
    case = mod.make_case(source_record(tmp_path), tmp_path)
    case["article_parts"] = []
    case["other_articles"] = []
    assert mod.article_case(case) is None


def test_article_part_before_number_is_extracted():
    positive, other = mod.article_parts_from_text("Обвиняемый по ч. 3 ст. 290 УК РФ и ч. 1 ст. 286 УК РФ")
    assert positive == ["3"]
    assert "286 ч.1" in other


def test_structured_process_metadata_wins():
    flags = mod.sudact_process("особый порядок и ст. 73", {"special_order": "False", "art_73": "True"})
    assert flags["special_order"] is False
    assert flags["art_73"] is True


def integrated_record(tmp_path):
    record = source_record(tmp_path)
    record["source"] = "sudact"
    record["metadata"] = {"special_order": "True", "art_73": "False"}
    return record


def test_build_manifest_schema_sources_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "moscow_records", lambda raw=None: [] if raw == mod.EXPANSION_RAW else [integrated_record(tmp_path)])
    monkeypatch.setattr(mod, "sudact_records", lambda: [])
    monkeypatch.setattr(mod, "source_hashes", lambda: {"synthetic": 1})
    manifest = mod.build(tmp_path)
    assert manifest["schema_version"] == 1
    assert manifest["counts"]["cases"] == 1
    assert manifest["counts"]["retrieval_cases"] == 1
    assert manifest["counts"]["risk_examples"] == 1
    assert manifest["counts"]["article_examples"] == 1
    assert manifest["counts"]["by_source"] == {"sudact": 1}
    assert manifest["counts"]["regional_historical_unavailable"] == 4
    assert manifest["leakage_rules"]["unknown_is_not_negative"] is True
    first_created = manifest["created_at"]
    first_manifest_bytes = (tmp_path / "manifest.json").read_bytes()
    resumed = mod.build(tmp_path)
    assert resumed["created_at"] == first_created
    assert (tmp_path / "manifest.json").read_bytes() == first_manifest_bytes


def test_public_manifest_has_no_identifier_lists(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "moscow_records", lambda raw=None: [] if raw == mod.EXPANSION_RAW else [integrated_record(tmp_path)])
    monkeypatch.setattr(mod, "sudact_records", lambda: [])
    monkeypatch.setattr(mod, "source_hashes", lambda: {"synthetic": 1})
    manifest = mod.build(tmp_path)
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "case_id" not in serialized
    assert "verdict_key" not in serialized
    assert "appeal-1" not in serialized


def test_case_schema_contains_separate_punishment_and_appeal(tmp_path):
    case = mod.make_case(source_record(tmp_path), tmp_path)
    for key in ("schema_version", "case_id", "verdict_group_id", "source", "source_year", "article_parts", "other_articles", "process", "facts", "punishment", "appeal", "quality", "split"):
        assert key in case
    assert case["punishment"]["prison_months"] == 24
    assert case["appeal"]["outcome"] == "changed"
    assert case["quality"]["text_coverage"] == 1
    assert case["split"] == "unassigned"


def test_first_instance_qualification_is_separate_from_appeal_metadata(tmp_path):
    record = source_record(tmp_path)
    record["first"]["participants"] = [{"codexArticles": [{"name": "Ст. 290, Ч. 3"}]}]
    record["appeal_participants"] = [{"codexArticles": [{"name": "Ст. 290, Ч. 5"}]}]
    case = mod.make_case(record, tmp_path)
    assert case["article_parts"] == ["3"]
    assert case["appeal"]["outcome_article_parts"] == ["5"]


def moscow_valid_texts():
    sentence = "ПРИГОВОР Тестового районного суда от 01.02.2024 по делу 1-1/2024. Суд установил обстоятельства. Суд ПРИГОВОРИЛ. " + "мотивировка " * 150
    appeal = (
        "АПЕЛЛЯЦИОННОЕ ОПРЕДЕЛЕНИЕ Московского городского суда 10-1/2024 от 01.03.2024. "
        "Рассмотрена жалоба на приговор Тестового районного суда от 01.02.2024 по делу 1-1/2024. "
        "ОПРЕДЕЛИЛ: Приговор оставить без изменения, " + "доводы " * 150
    )
    return sentence, appeal


def moscow_candidate():
    return {
        "appeal_id": "synthetic-moscow",
        "identity": {
            "uid": "SYNTHETIC-UID", "number": "10-1/2024", "year": 2024,
            "courtName": "Московский городской суд", "act_date": "01.03.2024", "documentType": "Уголовное дело",
        },
        "aliases": [],
        "lower": [{"number": "1-1/2024", "courtName": "Тестовый районный суд", "courtCode": "FIRST"}],
        "participants": [],
        "first": {
            "_id": "synthetic-first", "uid": "SYNTHETIC-UID", "number": "1-1/2024", "year": 2024,
            "courtName": "Тестовый районный суд", "dateFinal": "01.02.2024", "lastHearingDate": "01.02.2024",
            "participants": [],
        },
    }


def moscow_document(raw_dir, kind, text):
    binary = raw_dir / "texts" / f"{kind}.bin"
    text_path = raw_dir / "texts" / f"{kind}.txt"
    mod.atomic_write(binary, text.encode("utf-8"))
    mod.atomic_write(text_path, text.encode("utf-8"))
    digest = mod.sha256_file(binary)
    return {
        "path": binary.name, "bytes": binary.stat().st_size, "sha256": digest,
        "text_path": text_path.name, "text_source_sha256": digest, "text_sha256": mod.sha256_file(text_path),
    }


def test_confirmed_pair_requires_current_sha_and_identity(tmp_path):
    sentence, appeal = moscow_valid_texts()
    mod.write_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", [moscow_candidate()])
    documents = {"sentence": moscow_document(tmp_path, "sentence", sentence), "appeal": moscow_document(tmp_path, "appeal", appeal)}
    mod.atomic_json(tmp_path / "pair-progress.json", {"synthetic-moscow": {"status": "confirmed_pair", "outcome": "changed", "documents": documents, "errors": []}})
    records = mod.moscow_records(tmp_path)
    assert records[0]["full_pair"] is True
    (tmp_path / "texts" / "sentence.txt").write_text("изменённый текст", encoding="utf-8")
    records = mod.moscow_records(tmp_path)
    assert records[0]["full_pair"] is False
    assert records[0]["outcome"] == "unknown"
    assert "integrity_mismatch" in records[0]["identity_errors"]
    assert mod.retrieval_case(mod.make_case(records[0], tmp_path)) is None
    assert mod.article_case(mod.make_case(records[0], tmp_path)) is None


def test_current_validation_overrides_stale_failed_checkpoint_status(tmp_path):
    sentence, appeal = moscow_valid_texts()
    mod.write_jsonl(tmp_path / "work/appeal-candidates.jsonl", [moscow_candidate()])
    documents = {"sentence": moscow_document(tmp_path, "sentence", sentence), "appeal": moscow_document(tmp_path, "appeal", appeal)}
    mod.atomic_json(tmp_path / "pair-progress.json", {"synthetic-moscow": {"status": "verification_failed", "documents": documents, "errors": []}})
    record = mod.moscow_records(tmp_path)[0]
    assert record["full_pair"] is True
    assert record["outcome"] == "unchanged"
    assert mod.make_case(record, tmp_path)["quality"]["parser_status"] == "confirmed_pair"


def test_retrieval_excludes_failed_identity(tmp_path):
    case = mod.make_case(source_record(tmp_path), tmp_path)
    case["quality"]["retrieval_errors"] = ["sentence_number_mismatch"]
    assert mod.retrieval_case(case) is None
    assert mod.article_case(case) is None


def test_cross_source_dedup_uses_text_hash_and_canonical_identity(tmp_path):
    first = mod.make_case(source_record(tmp_path, "event-1"), tmp_path)
    second_record = source_record(tmp_path, "event-2")
    second_record["source"] = "another-source"
    second = mod.make_case(second_record, tmp_path)
    assert first["verdict_group_id"] == second["verdict_group_id"]
    rows = [first, second]
    mod.assign_splits(rows)
    assert rows[0]["split"] == rows[1]["split"]


def test_split_mode_is_part_of_resume_state(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "moscow_records", lambda raw=None: [] if raw == mod.EXPANSION_RAW else [integrated_record(tmp_path)])
    monkeypatch.setattr(mod, "sudact_records", lambda: [])
    monkeypatch.setattr(mod, "source_hashes", lambda: {"synthetic": 1})
    mod.build(tmp_path)
    mod.build(tmp_path, split=True)
    rows = mod.read_jsonl(tmp_path / "cases.jsonl")
    assert all(row["split"] != "unassigned" for row in rows)
    mod.build(tmp_path)
    rows = mod.read_jsonl(tmp_path / "cases.jsonl")
    assert all(row["split"] == "unassigned" for row in rows)


def test_resume_rebuilds_missing_or_empty_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "moscow_records", lambda raw=None: [] if raw == mod.EXPANSION_RAW else [integrated_record(tmp_path)])
    monkeypatch.setattr(mod, "sudact_records", lambda: [])
    monkeypatch.setattr(mod, "source_hashes", lambda: {"synthetic": 1})
    mod.build(tmp_path)
    (tmp_path / "cases.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "dataset-card.json").unlink()
    mod.build(tmp_path)
    assert (tmp_path / "dataset-card.json").exists()
    assert len(mod.read_jsonl(tmp_path / "cases.jsonl")) == 1


def test_unknown_process_flags_are_null_not_false():
    flags = mod.process_flags("Обстоятельства описаны в приложении.")
    assert set(flags.values()) == {None}


def test_prison_months_ignores_age():
    assert mod.prison_months("Подсудимому 35 лет. Суд приговорил к 2 годам лишения свободы.") == 24
    assert mod.prison_months("Подсудимому 35 лет. Других данных нет.") is None


def test_sudact_appeal_does_not_invent_article_part():
    row = {"art_290": "True", "parts_290": ""}
    assert mod.sudact_articles(row["parts_290"]) == []



def test_partial_documents_do_not_abort_moscow_dataset(tmp_path):
    sentence, _ = moscow_valid_texts()
    mod.write_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", [moscow_candidate()])
    documents = {"sentence": moscow_document(tmp_path, "sentence", sentence)}
    mod.atomic_json(tmp_path / "pair-progress.json", {"synthetic-moscow": {"status": "download_error", "documents": documents, "errors": ["HTTP 404"]}})
    records = mod.moscow_records(tmp_path)
    assert records[0]["full_pair"] is False
    assert records[0]["outcome"] == "unknown"
    assert records[0]["sentence_text"] == sentence


def test_expansion_inputs_are_hashed_and_use_same_parser(tmp_path, monkeypatch):
    calls = []
    record = integrated_record(tmp_path)

    def fake_moscow_records(raw=None):
        calls.append(mod.MOSCOW_RAW if raw is None else raw)
        return [record] if raw == mod.EXPANSION_RAW else []

    monkeypatch.setattr(mod, "moscow_records", fake_moscow_records)
    monkeypatch.setattr(mod, "sudact_records", lambda: [])
    monkeypatch.setattr(mod, "source_hashes", lambda: {"synthetic": 1})
    manifest = mod.build(tmp_path)
    assert calls == [mod.MOSCOW_RAW, mod.EXPANSION_RAW]
    assert manifest["counts"]["risk_examples"] == 1
    assert manifest["counts"]["by_source"] == {"sudact": 1}


def test_output_hashes_cover_dataset_and_dense_cards(tmp_path):
    mod.atomic_json(tmp_path / "dataset-card.json", {"status": "dataset_only"})
    mod.atomic_json(tmp_path / "model-card.json", {"pointer": "dense_index_generation_v1"})
    hashes = mod.output_hashes(tmp_path)
    assert "dataset-card.json" in hashes
    assert "model-card.json" in hashes
    assert hashes["dataset-card.json"] != hashes["model-card.json"]


def test_expansion_resume_detects_input_hash_change(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "moscow_records", lambda raw=None: [])
    monkeypatch.setattr(mod, "sudact_records", lambda: [])
    monkeypatch.setattr(mod, "source_hashes", lambda: {"synthetic": 1})
    mod.build(tmp_path)
    monkeypatch.setattr(mod, "source_hashes", lambda: {"synthetic": 2})
    assert not mod.valid_resume(tmp_path, mod.read_json(tmp_path / "raw" / "build-state.json", {}), {"synthetic": 2}, False)


def test_verdict_group_is_stable_to_whitespace_and_ocr(tmp_path):
    first = mod.make_case(source_record(tmp_path, "event-a"), tmp_path)
    second_record = source_record(tmp_path, "event-b")
    second_record["sentence_text"] = second_record["sentence_text"].replace(" ", "  ")
    second_record["lower"]["courtName"] = second_record["lower"]["courtName"].replace(" ", "  ")
    second = mod.make_case(second_record, tmp_path)
    assert first["verdict_group_id"] == second["verdict_group_id"]
    rows = [first, second]
    mod.assign_splits(rows)
    assert rows[0]["split"] == rows[1]["split"]


def test_verdict_group_is_stable_to_number_padding(tmp_path):
    first_record = source_record(tmp_path, "event-a")
    first_record["lower"]["number"] = "01-004/2021"
    first = mod.make_case(first_record, tmp_path)
    second_record = source_record(tmp_path, "event-b")
    second_record["lower"]["number"] = "1-4/2021"
    second = mod.make_case(second_record, tmp_path)
    assert first["verdict_group_id"] == second["verdict_group_id"]
    rows = [first, second]
    mod.assign_splits(rows)
    assert rows[0]["split"] == rows[1]["split"]
    assert len(mod.deduplicate_retrieval(rows)) == 1


def test_outcome_recomputed_from_verified_text(tmp_path):
    sentence, appeal = moscow_valid_texts()
    mod.write_jsonl(tmp_path / "work" / "appeal-candidates.jsonl", [moscow_candidate()])
    documents = {"sentence": moscow_document(tmp_path, "sentence", sentence), "appeal": moscow_document(tmp_path, "appeal", appeal)}
    mod.atomic_json(tmp_path / "pair-progress.json", {"synthetic-moscow": {"status": "confirmed_pair", "outcome": "changed", "documents": documents, "errors": []}})
    records = mod.moscow_records(tmp_path)
    assert records[0]["outcome"] == "unchanged"


def test_retrieval_deduplicates_by_verdict_group(tmp_path):
    first = mod.make_case(source_record(tmp_path, "appeal-a"), tmp_path)
    second = mod.make_case(source_record(tmp_path, "appeal-b"), tmp_path)
    rows = mod.deduplicate_retrieval([first, second])
    assert len(rows) == 1
    assert rows[0]["verdict_group_id"] == first["verdict_group_id"]


def test_identity_year_supports_dd_mm_yyyy_and_unknown():
    assert mod.identity_year({"act_date": "01.03.2021"}) == 2021
    assert mod.identity_year({"act_date": None}) is None
    assert mod.identity_year({"act_date": "not-a-date"}) is None


def test_appeal_failure_does_not_invalidate_retrieval_sentence(tmp_path):
    record = source_record(tmp_path, full_pair=False)
    record["identity_errors"] = ["appeal_type_not_confirmed"]
    record["retrieval_errors"] = []
    case = mod.make_case(record, tmp_path)
    assert mod.risk_case(case) is None
    assert mod.retrieval_case(case) is not None


def test_identical_texts_share_group_even_with_different_metadata_aliases(tmp_path):
    first_record = source_record(tmp_path, "event-a")
    first_record["lower"].update({"number": "01-0001/2024", "courtName": "Синтетический суд - Уголовное"})
    first = mod.make_case(first_record, tmp_path)
    second_record = source_record(tmp_path, "event-b")
    second_record["source"] = "another-source"
    second_record["lower"].update({"number": "1-1/24", "year": 2024})
    second_record["first"]["dateFinal"] = "2024-02-01"
    second = mod.make_case(second_record, tmp_path)
    assert mod.canonical_verdict_identity(first_record) == mod.canonical_verdict_identity(second_record)
    second_record["lower"]["courtName"] = "Алиас без совпадения"
    second = mod.make_case(second_record, tmp_path)
    rows = [first, second]
    mod.unify_verdict_text_groups(rows)
    assert rows[0]["verdict_group_id"] == rows[1]["verdict_group_id"]
    mod.assign_splits(rows)
    assert rows[0]["split"] == rows[1]["split"]
    assert len(mod.deduplicate_retrieval(rows)) == 1


def test_resume_rejects_previous_processing_contract():
    assert not mod.valid_resume(mod.Path("unused"), {"build_contract": mod.BUILD_CONTRACT - 1}, {}, False)


def test_regional_pair_adapter_rechecks_artifacts_and_text_only_provenance(tmp_path, monkeypatch):
    monkeypatch.undo()
    sentence, appeal = moscow_valid_texts()
    artifacts = []
    for kind, text in (("sentence", sentence), ("appeal", appeal)):
        path = tmp_path / "corpus" / f"{kind}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        artifacts.append({
            "record_id": "sudact:synthetic", "source": "sudact" if kind == "sentence" else "7kas",
            "text_path": f"corpus/{kind}.txt", "text_sha256": mod.sha256_file(path),
            "raw_path": None, "court": "Тестовый районный суд", "case_number": "1-1/2024",
            "decision_date": "2024-02-01" if kind == "sentence" else "2024-03-01",
        })
    row = {
        "pair_status": "confirmed_pair", "identity_errors": [], "outcome": "unchanged",
        "source": "7kas", "pair_id": "synthetic-pair", "verdict_group_id": "synthetic-group",
        "sentence_artifact": artifacts[0], "appeal_artifact": artifacts[1],
    }
    status = tmp_path / "status.jsonl"
    mod.write_jsonl(status, [row])
    records = mod.regional_pair_records(status, tmp_path)
    assert len(records) == 1
    assert records[0]["source"] == "7kas+sudact"
    assert records[0]["provenance_warnings"] == ["sentence_raw_unavailable"]
    assert len(mod.regional_pair_records(tmp_path / "missing.jsonl", tmp_path, additional_status_paths=[status])) == 1
    remanded = appeal.replace("Приговор оставить без изменения,", "Приговор отменить, дело направить на новое судебное разбирательство,")
    (tmp_path / "corpus/appeal.txt").write_text(remanded, encoding="utf-8")
    row["appeal_artifact"]["text_sha256"] = mod.sha256_file(tmp_path / "corpus/appeal.txt")
    mod.write_jsonl(status, [row])
    assert mod.regional_pair_records(status, tmp_path) == []
    (tmp_path / "corpus/sentence.txt").write_text("damaged", encoding="utf-8")
    assert mod.regional_pair_records(status, tmp_path) == []


def test_saved_appeal_error_does_not_exclude_valid_sentence(tmp_path):
    sentence, appeal = moscow_valid_texts()
    mod.write_jsonl(tmp_path / "work/appeal-candidates.jsonl", [moscow_candidate()])
    documents = {"sentence": moscow_document(tmp_path, "sentence", sentence), "appeal": moscow_document(tmp_path, "appeal", appeal)}
    mod.atomic_json(tmp_path / "pair-progress.json", {"synthetic-moscow": {"status": "confirmed_pair", "documents": documents, "errors": ["appeal_type_not_confirmed"]}})
    record = mod.moscow_records(tmp_path)[0]
    assert record["full_pair"] is True
    assert record["outcome"] == "unchanged"
    assert record["retrieval_errors"] == []
    assert mod.retrieval_case(mod.make_case(record, tmp_path)) is not None


def test_ocr_variants_with_same_complete_identity_share_group(tmp_path):
    first_record = source_record(tmp_path, "event-a")
    first_record["lower"]["number"] = "1-1/2024"
    first = mod.make_case(first_record, tmp_path)
    changed_record = source_record(tmp_path, "event-b")
    changed_record["lower"]["number"] = "1-1/2024"
    changed_record["sentence_text"] = changed_record["sentence_text"].replace("мотивировка", "мотивировкa", 1)
    second = mod.make_case(changed_record, tmp_path)
    assert first["facts"]["narrative"] != second["facts"]["narrative"]
    assert first["verdict_group_id"] == second["verdict_group_id"]
    rows = [first, second]
    mod.assign_splits(rows)
    assert rows[0]["split"] == rows[1]["split"]
    assert len(mod.deduplicate_retrieval(rows)) == 1


def test_legacy_recovery_schema_and_windows_paths_are_normalized(tmp_path, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(mod, "UNIFIED_CASE_INDEX", tmp_path / "unified-case-index.jsonl")
    monkeypatch.setattr(mod, "UNIFIED_METADATA", tmp_path / "metadata.jsonl")
    assert mod.UNIFIED_CASE_INDEX.is_relative_to(tmp_path) and mod.UNIFIED_METADATA.is_relative_to(tmp_path)
    sentence, appeal = moscow_valid_texts()
    evidence = tmp_path / "corpus/evidence.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}", encoding="utf-8")
    artifacts = []
    for kind, text in (("sentence", sentence), ("appeal", appeal)):
        path = tmp_path / "corpus" / f"{kind}.txt"
        path.write_text(text, encoding="utf-8")
        artifacts.append({"path": f"corpus\\{kind}.txt", "sha256": mod.sha256_file(path), "raw_path": None, "source": "sudact.ru" if kind == "sentence" else "bashkortostan", "record_id": None, "document_url": f"https://sudact.ru/regular/doc/{kind}/"})
    identity = {
        "record_id": "appeal-record", "first_court": "Тестовый районный суд",
        "first_case_number": "1-1/2024", "first_decision_date": "2024-02-01",
        "identity_evidence_path": "corpus/evidence.json", "identity_link_evidence_sha256": mod.sha256_file(evidence),
    }
    mod.write_jsonl(tmp_path / mod.UNIFIED_CASE_INDEX, [{"underlying_case_id": "case", "identity": identity}])
    row = {
        "source": "bashkortostan", "source_record_id": "appeal-record", "underlying_case_id": "case",
        "verdict_group_id": "group", "appeal_event_id": "event", "appeal_event": {"court": "Апелляционный суд", "number": "22-1/2024", "date": "01.03.2024", "type": "appeal_against_verdict"},
        "sentence": artifacts[0], "appeal": artifacts[1], "pair_status": "confirmed_pair", "outcome": "changed",
        "identity_errors": [], "recovery_evidence": "corpus/evidence.json", "recovery_evidence_sha256": mod.sha256_file(evidence),
    }
    status = tmp_path / "status.jsonl"
    mod.write_jsonl(status, [row])
    records = mod.regional_pair_records(status, tmp_path)
    assert len(records) == 1
    assert records[0]["source"] == "bashkortostan+sudact"
    assert records[0]["source_key"] == "event"
    assert records[0]["first"]["number"] == "1-1/2024"
    (tmp_path / mod.UNIFIED_CASE_INDEX).unlink()
    mod.write_jsonl(tmp_path / mod.UNIFIED_METADATA, [{"identity": identity}])
    assert len(mod.regional_pair_records(status, tmp_path)) == 1
