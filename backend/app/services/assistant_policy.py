"""Server-side routing and safety rules for HEALTHSYNC-AI's assistant."""
import re

_RECORD_SCOPE_TERMS = {
    "medical", "health record", "my record", "records", "report", "lab", "laboratory", "blood test",
    "result", "prescription", "prescribed", "medicine", "medication", "tablet", "capsule", "drug",
    "dose", "dosage", "strength", "frequency", "pharmacy", "allergy", "allergies", "vaccination",
    "vaccine", "immunization", "hospital", "discharge", "diagnosis", "diagnosed", "clinical", "scan",
    "imaging", "follow-up", "follow up", "doctor", "clinician", "test report", "my health",
}

_PRODUCT_FEATURE_TERMS = {
    "website", "web site", "web app", "site", "app", "platform", "healthsync", "feature", "features", "dashboard",
    "record category", "record categories", "medical records", "health records", "record", "records", "category", "categories", "upload", "document analysis", "analysis page", "documents",
    "health timeline", "timeline", "health line", "reminder", "reminders", "health profile", "profile", "secure sharing",
    "sharing", "share link", "privacy", "security", "sha-256", "integrity", "settings", "account", "signup",
    "sign up", "login", "log in", "vaccination page", "vaccinations", "ai assistant", "health assistant",
    "how does it work", "how do i use", "what can it do", "what can you do", "what is this", "tell me about",
}
_PRODUCT_INTENT_PATTERNS = (
    r"\bhow\s+(?:does|do|can|to)\b",
    r"\bwhat\s+(?:is|are|does|can|do)\b",
    r"\b(?:explain|describe|tell me about|teach me about|help me use|how to use)\b",
    r"\b(?:where|which page|which section)\b",
    r"\b(?:features?|capabilities|purpose|purpose of|overview|guide|instructions|list)\b",
    r"\b(?:work|works|working|use|using|navigate|find|access|create|add|edit|manage|share|revoke)\b",
)


def is_healthsync_product_question(question: str) -> bool:
    """Recognize requests about the HEALTHSYNC-AI website and its features.

    Do not treat requests for the user's personal due dates/records as product help;
    those need account-scoped data handling instead.
    """
    q = re.sub(r"\s+", " ", str(question or "").casefold()).strip()
    if not q:
        return False
    personal_data_request = re.search(
        r"\b(?:my|mine|show me|list my|which of my|what are my|when is my|when are my|do i have|am i due|are any of my)\b",
        q,
    ) and re.search(r"\b(?:records?|reminders?|vaccinations?|vaccines?|appointments?|profile|medicines?|medications?|results?)\b", q)
    if personal_data_request and "page" not in q and not re.search(r"\b(?:how|where|which page|which section|feature|work|works|use|create|manage|edit|settings)\b", q):
        return False
    has_feature = any(term in q for term in _PRODUCT_FEATURE_TERMS)
    has_product_intent = any(re.search(pattern, q) for pattern in _PRODUCT_INTENT_PATTERNS)
    mentions_product_name = "healthsync" in q or "this website" in q or "this app" in q or "this platform" in q
    # Broad app-level questions should work even if no clinical keyword is present.
    broad_app_question = mentions_product_name and ("?" in q or has_product_intent or len(q.split()) <= 12)
    return broad_app_question or (has_feature and has_product_intent)


def is_record_question(question: str) -> bool:
    q = str(question or "").casefold()
    return any(term in q for term in _RECORD_SCOPE_TERMS)


def requests_new_treatment_advice(question: str) -> bool:
    q = question.casefold()
    patterns = (
        r"\bshould i (?:take|stop|start|skip|increase|decrease|change|double)\b",
        r"\bcan i (?:take|stop|start|combine|mix)\b",
        r"\b(?:recommend|prescribe) (?:me )?(?:a|some|any)? ?(?:medicine|medication|drug|tablet|treatment)\b",
        r"\bwhat (?:medicine|medication|drug|treatment) should i take\b",
        r"\b(?:diagnose me|what disease do i have|is this cancer|is this diabetes)\b",
        r"\bchange my (?:dose|dosage|medication)\b",
    )
    return any(re.search(pattern, q) for pattern in patterns)
