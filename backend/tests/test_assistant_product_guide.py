from app.services.assistant_product_guide import _product_help_fallback

def test_category_fallback_lists_every_supported_upload_category():
    answer = _product_help_fallback("What are all the record categories?")
    for category in ("Lab report", "Prescription", "Vaccination record", "Diagnosis / clinical note", "Medication record", "Allergy record", "Hospital visit / discharge", "Imaging report", "Other document"):
        assert category in answer

def test_secure_sharing_fallback_explains_consent_expiry_and_revocation():
    answer = _product_help_fallback("How does secure sharing work?").casefold()
    assert "consent" in answer
    assert "revoke" in answer
    assert "24 hours" in answer

def test_profile_and_reminders_have_usable_directions():
    assert "Health profile" in _product_help_fallback("How does my profile page work?")
    assert "Reminders & follow-ups" in _product_help_fallback("How do reminders work?")


def test_openrouter_product_prompt_contains_core_product_features(monkeypatch):
    from app.services import ai
    captured = {}
    def fake_post(messages, *, model, max_tokens, temperature=0.1):
        captured["messages"] = messages
        captured["model"] = model
        return "answer"
    monkeypatch.setattr(ai, "_post_chat", fake_post)
    assert ai.openrouter_product_answer("Explain all HEALTHSYNC-AI features") == "answer"
    system_prompt = captured["messages"][0]["content"].casefold()
    for term in ("record categories", "timeline", "reminders", "health profile", "secure sharing", "sha-256", "do not invent"):
        assert term in system_prompt
