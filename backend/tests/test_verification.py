from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.services.verification import (
    VerifyItemRequest,
    apply_item_verification,
    update_record_verification_status,
)


def sample_data():
    return {
        "candidates": [{"field": "glucose", "value": "120", "unit": "mg/dL", "source_excerpt": "Glucose 120 mg/dL"}],
        "clinical_analysis": {
            "status": "complete",
            "medicines": [{"name_as_written": "Med A", "dose_as_written": "1 tablet", "source_excerpt": "Med A 1 tablet"}],
            "tests_and_results": [],
            "documented_instructions": {"precautions": [{"text": "keep appointment", "source_excerpt": "keep appointment"}], "food": [{"text": "after food", "source_excerpt": "after food"}]},
        },
    }


def test_user_can_correct_and_verify_medicine_without_changing_source_excerpt():
    data = sample_data()
    payload = VerifyItemRequest(
        item_type="medicine", index=0, status="user_verified",
        corrected_values={"dose_as_written": "Half tablet"},
        note="Compared with original handwriting",
    )
    item, updated, _clinical, status = apply_item_verification(data, payload)
    assert item["dose_as_written"] == "Half tablet"
    assert item["source_excerpt"] == "Med A 1 tablet"
    assert item["verification_status"] == status == "corrected_and_user_verified"
    assert updated["clinical_analysis"]["medicines"][0]["verification_note"] == "Compared with original handwriting"


def test_marking_candidate_incorrect_is_saved_as_review_not_verified_truth():
    data = sample_data()
    item, updated, _clinical, status = apply_item_verification(
        data, VerifyItemRequest(item_type="candidate", index=0, status="rejected_as_incorrect")
    )
    assert item["verification_status"] == status == "rejected_as_incorrect"
    record = SimpleNamespace(verification_status="pending_review")
    counts = update_record_verification_status(record, updated)
    assert counts["record_status"] == "partially_reviewed"


def test_correction_cannot_rewrite_source_excerpt():
    with pytest.raises(HTTPException) as exc:
        apply_item_verification(
            sample_data(), VerifyItemRequest(item_type="medicine", index=0, corrected_values={"source_excerpt": "fabricated evidence"})
        )
    assert exc.value.status_code == 422


def test_verification_rejects_missing_item_index():
    with pytest.raises(HTTPException) as exc:
        apply_item_verification(sample_data(), VerifyItemRequest(item_type="medicine", index=12))
    assert exc.value.status_code == 404


def test_all_accepted_items_become_reviewed_by_user_not_clinically_verified():
    data = sample_data()
    for payload in [
        VerifyItemRequest(item_type="candidate", index=0),
        VerifyItemRequest(item_type="medicine", index=0),
        VerifyItemRequest(item_type="instruction", section="precautions", index=0),
    ]:
        apply_item_verification(data, payload)
    record = SimpleNamespace(verification_status="pending_review")
    counts = update_record_verification_status(record, data)
    assert counts == {"total_items": 3, "reviewed_items": 3, "record_status": "reviewed_by_user"}


def test_diet_and_exercise_instruction_sections_are_not_supported():
    for section in ("food", "exercise"):
        with pytest.raises(HTTPException) as exc:
            apply_item_verification(
                sample_data(), VerifyItemRequest(item_type="instruction", section=section, index=0)
            )
        assert exc.value.status_code == 422


def test_patient_and_other_document_details_can_be_corrected_and_reviewed():
    data = sample_data()
    data["clinical_analysis"]["patient_details"] = [{"label": "Patient name", "value_as_written": "Monika", "source_excerpt": "Patient name: Monika"}]
    data["clinical_analysis"]["other_details"] = [{"category": "diagnosis", "label": "Impression", "value_as_written": "As written", "date_as_written": "", "source_excerpt": "Impression: As written"}]
    for section, index, corrected in [
        ("patient_details", 0, {"value_as_written": "Mounika"}),
        ("other_details", 0, {"value_as_written": "Not readable"}),
    ]:
        item, updated, _clinical, status = apply_item_verification(
            data, VerifyItemRequest(item_type="detail", section=section, index=index, corrected_values=corrected)
        )
        assert status == "corrected_and_user_verified"
        assert item["verification_status"] == status
        assert item["source_excerpt"]
        assert item["value_as_written"] in {"Mounika", "Not readable"}


def test_document_metadata_and_uncertain_handwriting_are_reviewable():
    data = sample_data()
    data["clinical_analysis"]["document_metadata"] = [{"label": "Doctor", "value_as_written": "Dr Ray", "date_as_written": "", "source_excerpt": "Dr Ray"}]
    data["clinical_analysis"]["uncertain_items"] = [{"item": "[unclear]", "why_uncertain": "handwriting", "source_excerpt": "Rx [unclear]"}]
    metadata, *_ = apply_item_verification(data, VerifyItemRequest(item_type="detail", section="document_metadata", index=0))
    uncertain, *_ = apply_item_verification(data, VerifyItemRequest(item_type="uncertain", index=0, corrected_values={"item": "Rx?"}))
    assert metadata["verification_status"] == "user_verified"
    assert uncertain["verification_status"] == "corrected_and_user_verified"
    assert uncertain["source_excerpt"] == "Rx [unclear]"
