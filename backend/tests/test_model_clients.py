from app.services.model_clients import validate_model_candidates


def test_qwen_candidate_is_kept_only_when_evidence_and_value_are_literal():
    pages = [{"page": 1, "text": "Hemoglobin: 13.2 g/dL\nGlucose: 94 mg/dL", "method": "paddleocr_vl_1_6"}]
    predictions = [
        {"field": "hemoglobin", "value": "13.2", "unit": "g/dL", "source_page": 1, "source_excerpt": "Hemoglobin: 13.2 g/dL"},
        {"field": "glucose", "value": "120", "unit": "mg/dL", "source_page": 1, "source_excerpt": "Glucose: 94 mg/dL"},
        {"field": "diagnosis", "value": "diabetes", "unit": "", "source_page": 1, "source_excerpt": "Not present in this page"},
    ]
    accepted, rejected = validate_model_candidates(pages, predictions)
    assert len(accepted) == 1
    assert rejected == 2
    assert accepted[0]["value"] == "13.2"
    assert accepted[0]["source_line"] == 1
    assert accepted[0]["status"] == "requires_human_review"
    assert accepted[0]["confidence_type"] == "not_calibrated"
    assert accepted[0]["extraction_method"] == "source_validated_model_extractor"


def test_candidate_validator_does_not_invent_page_numbers():
    pages = [{"page": 1, "text": "TSH: 2.4 mIU/L", "method": "pdf_text"}]
    accepted, rejected = validate_model_candidates(pages, [
        {"field": "tsh", "value": "2.4", "unit": "mIU/L", "source_page": 4, "source_excerpt": "TSH: 2.4 mIU/L"}
    ])
    assert accepted == []
    assert rejected == 1


def test_local_adapter_config_requires_existing_peft_adapter(tmp_path, monkeypatch):
    from app.core.config import settings
    from app.services.local_model_client import local_qwen_extractor_configured

    monkeypatch.setattr(settings, "local_qwen_adapter_dir", str(tmp_path))
    assert local_qwen_extractor_configured() is False
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert local_qwen_extractor_configured() is True


def test_local_qwen_response_is_source_validated():
    import json
    from app.services.local_model_client import _parse_and_validate_response

    pages = [{"page": 1, "text": "Hemoglobin: 13.2 g/dL"}]
    response = json.dumps({"candidates": [
        {"field": "hemoglobin", "value": "13.2", "unit": "g/dL",
         "source_page": 1, "source_excerpt": "Hemoglobin: 13.2 g/dL"},
        {"field": "glucose", "value": "120", "unit": "mg/dL",
         "source_page": 1, "source_excerpt": "Hemoglobin: 13.2 g/dL"},
    ]})
    accepted, rejected = _parse_and_validate_response(pages, response)
    assert len(accepted) == 1
    assert rejected == 1
    assert accepted[0]["extraction_method"] == "local_qwen_lora_extractor"


