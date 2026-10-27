from app.services.assistant_policy import (
    is_healthsync_product_question as _is_healthsync_product_question,
    is_record_question as _is_record_question,
    requests_new_treatment_advice as _requests_new_treatment_advice,
)


def test_assistant_accepts_record_and_prescription_questions():
    assert _is_record_question("What medicine and dose are written on my latest prescription?")
    assert _is_record_question("Summarize my latest lab report")


def test_assistant_refuses_unrelated_questions():
    assert not _is_record_question("Write me a fantasy story about dragons")
    assert not _is_record_question("How do I fix a React component?")


def test_assistant_flags_new_treatment_advice():
    assert _requests_new_treatment_advice("Should I stop my medicine?")
    assert _requests_new_treatment_advice("Can I take a new medication?")
    assert not _requests_new_treatment_advice("What dose was written on the prescription?")


def test_assistant_accepts_healthsync_product_questions():
    assert _is_healthsync_product_question("Tell me about HEALTHSYNC-AI")
    assert _is_healthsync_product_question("What are all the record categories?")
    assert _is_healthsync_product_question("List all record categories")
    assert _is_healthsync_product_question("How does the health timeline work?")
    assert _is_healthsync_product_question("How do reminders and follow-ups work?")
    assert _is_healthsync_product_question("What is stored in my health profile page?")
    assert _is_healthsync_product_question("How does secure sharing work and how do I revoke access?")


def test_assistant_keeps_personal_record_questions_out_of_product_faq():
    assert not _is_healthsync_product_question("What are my reminders due tomorrow?")
    assert not _is_healthsync_product_question("What medicine and dose are written on my latest prescription?")



def test_product_help_classifier_covers_uploads_and_site_navigation():
    assert _is_healthsync_product_question("How do I upload a medical record?")
    assert _is_healthsync_product_question("Tell me about the health line and profile")
    assert _is_healthsync_product_question("How can I use the AI assistant on this website?")
