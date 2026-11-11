from tools import recover_unified_case_index as mod

import pytest

pytestmark = pytest.mark.skipif(
    not (mod.COLLECTION / "raw").is_dir() or not mod.TARGET.is_file() or not mod.METADATA.is_file() or not mod.PAIR_STATUS.is_file(),
    reason="Local judicial corpus and rebuilt index are not present",
)


def test_raw_backed_new_records_have_verified_counts():
    records = mod.new_records()
    assert len(records) == 748
    assert len({record["underlying_case_id"] for record in records}) == 748
    assert all(record["origin"] == "rebuilt_new_case" for record in records)
    assert all(record["identity"]["case_keys"] for record in records)
    checks = mod.verify_new(records)
    assert checks["unique_normalized_texts"] == 748
    assert checks["text_sha256_verified"] == 748
    assert checks["raw_sha256_verified"] == 748
    assert checks["raw_path_nulls"] == 0
    assert checks["identity_complete_for_recovery"] == 728


def test_ambiguous_candidates_are_not_silently_included():
    recovered = mod.new_records()
    ambiguous = mod.ambiguous_records()
    assert len(ambiguous) == 42
    assert {record["selection_status"] for record in ambiguous} == {"ambiguous_not_included"}
    recovered_ids = {record["underlying_case_id"] for record in recovered}
    assert all(record["case_id"] not in recovered_ids for record in ambiguous)


def test_main_target_contains_only_verified_rebuilt_records():
    assert mod.sha256(mod.TARGET) == "46fab52d181188cdc00704886db576e04927b263be75b53911eff2e30f0b9360"
    assert mod.sha256(mod.METADATA) == "46fab52d181188cdc00704886db576e04927b263be75b53911eff2e30f0b9360"
    status = mod.json.loads((mod.TARGET.parent / "recovery-status.json").read_text(encoding="utf-8"))
    assert status["status"] == "incomplete_rebuild"
    assert status["recovered_new_records"] == 748
    assert status["unproven_existing_records"] == 252
    assert (mod.WORK / "damaged-synthetic-line.jsonl").is_file()
    assert (mod.WORK / "damaged-metadata-synthetic-line.jsonl").is_file()
