from app.services.medical import _local_then_optional_hosted_ocr, _ocr_disagreement_summary, _should_escalate_ocr
from app.services.evidence import build_evidence_report
from types import SimpleNamespace


def test_empty_or_short_text_requests_escalation():
    escalate, reason = _should_escalate_ocr("", {"local_mean_word_confidence": None})
    assert escalate is True
    assert "little or no readable text" in reason


def test_high_quality_local_ocr_does_not_escalate(monkeypatch):
    monkeypatch.setenv("OCR_ESCALATION_MIN_CONFIDENCE", "65")
    text = "Patient report: Hemoglobin 13.2 g/dL. Date 2026-10-15."
    quality = {
        "local_mean_word_confidence": 88.0,
        "local_low_confidence_word_ratio": 0.05,
    }
    escalate, _ = _should_escalate_ocr(text, quality)
    assert escalate is False


def test_low_confidence_or_many_uncertain_words_escalates(monkeypatch):
    monkeypatch.setenv("OCR_ESCALATION_MIN_CONFIDENCE", "65")
    text = "Patient report: Hemoglobin 13.2 g/dL. Date 2026-10-15."
    low_mean = _should_escalate_ocr(text, {
        "local_mean_word_confidence": 42.0,
        "local_low_confidence_word_ratio": 0.2,
    })
    many_low_words = _should_escalate_ocr(text, {
        "local_mean_word_confidence": 75.0,
        "local_low_confidence_word_ratio": 0.4,
    })
    assert low_mean[0] is True
    assert many_low_words[0] is True


def test_disagreement_summary_is_empty_for_matching_transcripts():
    result = _ocr_disagreement_summary("Hemoglobin 13.2 g/dL", "Hemoglobin 13.2 g/dL")
    assert result["disagreement_ratio"] == 0.0
    assert result["disagreement_segments"] == []


def test_disagreement_summary_exposes_differing_tokens_without_selecting_truth():
    result = _ocr_disagreement_summary("Medicine 10 mg daily", "Medicine 70 mg daily")
    assert result["disagreement_ratio"] > 0
    assert result["disagreement_segments"]
    assert {"local", "hosted", "difference_type"} <= set(result["disagreement_segments"][0])


def test_evidence_report_surfaces_ocr_disagreement():
    record = SimpleNamespace(
        id="record-1", title="Prescription", file_sha256="a" * 64,
        extracted_text="Medicine 10 mg daily", record_date=None, created_at=None,
        extracted_data={
            "candidates": [], "extraction_method": "tesseract_ocr", "page_count": 1,
            "ocr_quality": [{
                "page": 1, "review_recommended": True,
                "local_mean_word_confidence": 41.0,
                "cloud_escalation_reason": "low local confidence",
                "disagreement_ratio": 0.2,
                "disagreement_segments": [{"local": "10", "hosted": "70", "difference_type": "replace"}],
            }],
        },
    )
    report = build_evidence_report(record, [record])
    types = {flag["type"] for flag in report["flags"]}
    assert "ocr_quality_review" in types
    assert "ocr_engine_disagreement" in types
    assert len(report["ocr_quality"]) == 1


def test_hosted_ocr_is_not_called_without_consent(monkeypatch):
    monkeypatch.setattr("app.services.medical._ocr_image_with_quality", lambda image: (
        "uncertain transcription text", {
            "local_mean_word_confidence": 35.0,
            "local_low_confidence_word_ratio": 0.5,
            "local_word_count": 3,
            "local_text_characters": 29,
            "confidence_is_calibrated_probability": False,
        }
    ))
    def should_not_call_hosted(*args, **kwargs):
        raise AssertionError("Hosted OCR must not run without explicit consent")
    monkeypatch.setattr("app.services.medical._selected_ocr", should_not_call_hosted)
    text, method, diagnostic = _local_then_optional_hosted_ocr(
        object(), use_cloud_ocr=False, page_number=1, source_type="image"
    )
    assert text == "uncertain transcription text"
    assert method == "tesseract_ocr"
    assert diagnostic["cloud_escalated"] is False
    assert diagnostic["review_recommended"] is True


def test_opted_in_low_quality_page_escalates_and_keeps_disagreement(monkeypatch):
    monkeypatch.setattr("app.services.medical._ocr_image_with_quality", lambda image: (
        "Medicine 10 mg daily", {
            "local_mean_word_confidence": 35.0,
            "local_low_confidence_word_ratio": 0.5,
            "local_word_count": 4,
            "local_text_characters": 21,
            "confidence_is_calibrated_probability": False,
        }
    ))
    monkeypatch.setattr("app.services.medical._selected_ocr", lambda *args, **kwargs: ("Medicine 70 mg daily", "openrouter_vision_ocr"))
    text, method, diagnostic = _local_then_optional_hosted_ocr(
        object(), use_cloud_ocr=True, page_number=2, source_type="scanned_pdf_page"
    )
    assert text == "Medicine 70 mg daily"
    assert method == "openrouter_vision_ocr"
    assert diagnostic["cloud_escalated"] is True
    assert diagnostic["disagreement_ratio"] > 0
    assert diagnostic["disagreement_segments"]
    assert diagnostic["local_transcription"] == "Medicine 10 mg daily"
    assert diagnostic["hosted_transcription"] == "Medicine 70 mg daily"
    assert diagnostic["review_recommended"] is True


def test_hosted_failure_falls_back_to_local_text(monkeypatch):
    monkeypatch.setattr("app.services.medical._ocr_image_with_quality", lambda image: (
        "local transcription", {
            "local_mean_word_confidence": 30.0,
            "local_low_confidence_word_ratio": 0.8,
            "local_word_count": 2,
            "local_text_characters": 20,
            "confidence_is_calibrated_probability": False,
        }
    ))
    def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")
    monkeypatch.setattr("app.services.medical._selected_ocr", fail)
    text, method, diagnostic = _local_then_optional_hosted_ocr(
        object(), use_cloud_ocr=True, page_number=1, source_type="image"
    )
    assert text == "local transcription"
    assert method == "tesseract_ocr"
    assert diagnostic["cloud_fallback_used"] is True
    assert "provider unavailable" in diagnostic["cloud_error"]


def test_vision_ocr_uses_openrouter_without_huggingface(monkeypatch):
    from app.services import medical, ai

    monkeypatch.setattr(ai, "openrouter_configured", lambda: True)
    monkeypatch.setattr(ai, "openrouter_vision_transcribe", lambda image: "Prescription text from OpenRouter")

    text, method = medical._selected_ocr(object(), use_cloud_ocr=True, filename="page-1")
    assert text == "Prescription text from OpenRouter"
    assert method == "openrouter_vision_ocr"


def test_no_hosted_provider_gives_actionable_error(monkeypatch):
    from app.services import medical, ai

    monkeypatch.setattr(ai, "openrouter_configured", lambda: False)
    try:
        medical._selected_ocr(object(), use_cloud_ocr=True, filename="page-1")
    except RuntimeError as exc:
        assert "OPENROUTER_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected actionable provider error")


def test_short_handwritten_medical_note_escalates_even_with_good_tesseract_confidence():
    text = "Rx Amox 500 mg BD"
    quality = {
        "local_mean_word_confidence": 91.0,
        "local_low_confidence_word_ratio": 0.0,
    }
    escalate, reason = _should_escalate_ocr(text, quality, source_type="image")
    assert escalate is True
    assert "short image-page transcription" in reason or "short medical-note transcription" in reason


def test_searchable_pdf_merge_preserves_handwritten_visual_text():
    from app.services.medical import _merge_text_layer_and_ocr

    merged = _merge_text_layer_and_ocr(
        "Printed lab report\nHemoglobin 13.2 g/dL",
        "Printed lab report\nHemoglobin 13.2 g/dL\nhandwritten: repeat test in 2 weeks",
    )
    assert "Hemoglobin 13.2 g/dL" in merged
    assert "handwritten: repeat test in 2 weeks" in merged
    assert "Additional visual OCR text" in merged


def test_sparse_image_page_escalates_even_with_high_tesseract_confidence():
    text = "Patient has headache for 3 days; Rx noted below"
    quality = {
        "local_mean_word_confidence": 92.0,
        "local_low_confidence_word_ratio": 0.0,
    }
    escalate, reason = _should_escalate_ocr(text, quality, source_type="scanned_pdf_page")
    assert escalate is True
    assert "short image-page transcription" in reason


def test_manual_force_vision_retries_selected_page_even_if_local_quality_is_good(monkeypatch):
    monkeypatch.setattr("app.services.medical._ocr_image_with_quality", lambda image: (
        "Patient took medicine and returned for follow-up next week.", {
            "local_mean_word_confidence": 95.0,
            "local_low_confidence_word_ratio": 0.0,
            "local_word_count": 9,
            "local_text_characters": 59,
            "confidence_is_calibrated_probability": False,
        }
    ))
    monkeypatch.setattr("app.services.medical._selected_ocr", lambda *args, **kwargs: (
        "Patient took medicine; handwritten dosage [unclear]; return next week.", "openrouter_vision_ocr"
    ))
    text, method, diagnostic = _local_then_optional_hosted_ocr(
        object(), use_cloud_ocr=True, page_number=3, source_type="scanned_pdf_page", force_vision=True
    )
    assert method == "openrouter_vision_ocr"
    assert "handwritten dosage" in text
    assert diagnostic["cloud_escalated"] is True
    assert diagnostic["cloud_escalation_reason"] == "manual page re-read requested"


def test_high_confidence_page_still_uses_vision_when_enabled(monkeypatch):
    monkeypatch.setattr("app.services.medical._ocr_image_with_quality", lambda image: (
        "Patient report: Hemoglobin 13.2 g/dL. Date 2026-10-15.", {
            "local_mean_word_confidence": 94.0,
            "local_low_confidence_word_ratio": 0.0,
            "local_word_count": 8,
            "local_text_characters": 49,
            "confidence_is_calibrated_probability": False,
        }
    ))
    calls = []
    def vision(*args, **kwargs):
        calls.append(kwargs)
        return "Handwritten note: review dosage with prescriber.", "openrouter_vision_ocr"
    monkeypatch.setattr("app.services.medical._selected_ocr", vision)
    text, method, diagnostic = _local_then_optional_hosted_ocr(
        object(), use_cloud_ocr=True, page_number=1, source_type="image"
    )
    assert calls, "Vision transcription must run for opted-in pages even when local OCR confidence is high"
    assert text == "Handwritten note: review dosage with prescriber."
    assert method == "openrouter_vision_ocr"
    assert diagnostic["cloud_escalated"] is True


def test_vision_transcriber_sends_full_page_and_enlarged_handwriting_views(monkeypatch):
    from PIL import Image, ImageDraw
    from app.services import ai

    captured = {}
    def fake_post(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return "[HANDWRITTEN TEXT]\nTab [unclear] 500 mg [unclear]"

    monkeypatch.setattr(ai, "_post_chat", fake_post)
    image = Image.new("RGB", (800, 1200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 500), "Tab handwritten medicine 500 mg", fill="navy")

    result = ai.openrouter_vision_transcribe(image)
    content = captured["messages"][1]["content"]
    image_parts = [part for part in content if part.get("type") == "image_url"]
    text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]

    assert result.startswith("[HANDWRITTEN TEXT]")
    assert len(image_parts) == 3, "Send full page plus two enlarged, overlapping page crops"
    assert any("handwritten" in value.casefold() for value in text_parts)
    assert "qwen/qwen2.5-vl-72b-instruct" == captured["kwargs"]["model"]
    assert captured["kwargs"]["max_tokens"] >= 5000


def test_successful_vision_diagnostic_records_strategy_version(monkeypatch):
    monkeypatch.setattr("app.services.medical._ocr_image_with_quality", lambda image: (
        "printed header only", {
            "local_mean_word_confidence": 90.0,
            "local_low_confidence_word_ratio": 0.0,
            "local_word_count": 3,
            "local_text_characters": 20,
            "confidence_is_calibrated_probability": False,
        }
    ))
    monkeypatch.setattr("app.services.medical._selected_ocr", lambda *args, **kwargs: (
        "[HANDWRITTEN TEXT] Tab [unclear] 500 mg", "openrouter_vision_ocr"
    ))
    _, _, diagnostic = _local_then_optional_hosted_ocr(
        object(), use_cloud_ocr=True, page_number=1, source_type="image"
    )
    assert diagnostic["vision_strategy_version"] == 2
    assert diagnostic["vision_strategy"] == "full_page_plus_enlarged_handwriting_regions"
