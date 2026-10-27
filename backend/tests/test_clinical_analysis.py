import json

from app.core.config import settings
from app.services import ai


def test_openrouter_analysis_parses_and_source_checks_document_fields(monkeypatch):
    raw = {
        "document_type": "Prescription",
        "provider": "Dr Example",
        "document_date": "2026-10-01",
        "medicines": [
            {
                "name_as_written": "Exampletab",
                "strength_as_written": "10 mg",
                "dose_as_written": "1 tablet",
                "route_as_written": "oral",
                "frequency_as_written": "once daily",
                "duration_as_written": "5 days",
                "instructions_as_written": "after food",
                "storage_as_written": "",
                "indication_as_written": "",
                "source_excerpt": "Exampletab 10 mg - 1 tablet once daily for 5 days after food",
            },
            {
                "name_as_written": "InventedMed",
                "strength_as_written": "20 mg",
                "dose_as_written": "",
                "route_as_written": "",
                "frequency_as_written": "",
                "duration_as_written": "",
                "instructions_as_written": "",
                "storage_as_written": "",
                "indication_as_written": "",
                "source_excerpt": "InventedMed 20 mg once daily",
            },
        ],
        "tests_and_results": [],
        "documented_instructions": {
            "food": [{"text": "after food", "source_excerpt": "Exampletab 10 mg - 1 tablet once daily for 5 days after food"}],
            "exercise": [{"text": "Exercise daily", "source_excerpt": "exercise every day"}],
            "precautions": [], "temperature": [], "medicine_storage": [], "hospital_follow_up": [],
        },
        "uncertain_items": [],
        "general_education": {"food": "DO NOT DISPLAY THIS DIET PLAN", "exercise": "DO NOT DISPLAY THIS PLAN"},
        "not_specified": ["temperature guidance"],
    }
    monkeypatch.setattr(ai, "openrouter_configured", lambda: True)
    monkeypatch.setattr(ai, "_post_chat", lambda *args, **kwargs: json.dumps(raw))
    result = ai.openrouter_analyze_medical_record(
        text="Exampletab 10 mg - 1 tablet once daily for 5 days after food",
        title="Prescription", record_type="prescription"
    )
    assert result["status"] == "complete"
    assert result["medicines"][0]["source_verified"] is True
    assert result["medicines"][1]["source_verified"] is False
    assert "common_uses_general" not in result["medicines"][0]
    assert "common_uses_general" not in result["medicines"][1]
    assert "food" not in result["documented_instructions"]
    assert "exercise" not in result["documented_instructions"]
    assert "general_education" not in result
    assert "exercise instructions" not in result["not_specified"]


def test_openrouter_analysis_handles_too_little_text():
    result = ai.openrouter_analyze_medical_record(text="", title="Rx", record_type="prescription")
    assert result["status"] == "insufficient_text"
    assert result["medicines"] == []


def test_json_parser_accepts_fenced_json():
    result = ai._parse_json_object('```json\n{"medicines": []}\n```')
    assert result == {"medicines": []}




def test_prescription_lines_are_retained_when_structured_medicine_parse_is_empty(monkeypatch):
    raw = {
        "document_type": "Prescription", "provider": "", "document_date": "",
        "plain_language_summary": "The source contains handwritten prescription lines.",
        "document_metadata": [], "patient_details": [], "other_details": [],
        "medicines": [], "tests_and_results": [],
        "documented_instructions": {"precautions": [], "temperature": [], "medicine_storage": [], "hospital_follow_up": []},
        "uncertain_items": [], "not_specified": [],
    }
    text = "[HANDWRITTEN TEXT]\n1) Tab [unclear] 500 mg BD\n2) Cap. Caldob [unclear]"
    monkeypatch.setattr(ai, "openrouter_configured", lambda: True)
    monkeypatch.setattr(ai, "_post_chat", lambda *args, **kwargs: json.dumps(raw))

    result = ai.openrouter_analyze_medical_record(text=text, title="Prescription", record_type="prescription")
    assert result["medicines"] == []
    assert len(result["prescription_lines"]) == 2
    assert "Tab [unclear] 500 mg BD" in result["prescription_lines"][0]["line_as_written"]
    assert result["prescription_lines"][0]["source_verified"] is True
