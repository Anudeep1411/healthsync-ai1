"""Patient-specific medical-record retrieval with OpenRouter-only grounded answers."""
import re
import json
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models import User, MedicalRecord, Reminder
from app.services.ai import openrouter_configured, openrouter_record_answer, openrouter_product_answer
from app.services.local_model_client import local_qwen_extractor_configured
from app.services.assistant_product_guide import _product_help_fallback
from app.services.assistant_policy import (
    is_healthsync_product_question as _is_healthsync_product_question,
    is_record_question as _is_record_question,
    requests_new_treatment_advice as _requests_new_treatment_advice,
)

router = APIRouter(tags=["AI Health Assistant"])



class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1600)
    record_ids: list[str] = Field(default_factory=list, max_length=10)
    use_external_ai: bool = True


def _tokens(value: str) -> list[str]:
    ignore = {
        "what", "when", "where", "which", "about", "from", "with", "have", "does", "this", "that", "then", "there",
        "tell", "show", "find", "please", "my", "the", "for", "and", "are", "was", "were", "can", "you", "me", "any",
        "all", "how", "record", "records", "medical", "health", "latest", "recent", "history", "summary", "summarize",
        "summarise", "overview", "uploaded", "document", "documents", "into", "from", "your", "their", "our", "about",
    }
    result = []
    for word in re.findall(r"[a-zA-Z0-9]{3,}", value.casefold()):
        if word in ignore:
            continue
        # Light normalization catches common plurals without a heavyweight NLP dependency.
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and len(word) > 4 and not word.endswith("ss"):
            word = word[:-1]
        result.append(word)
    return result


def _short_excerpt(text: str, terms: list[str], limit: int = 650) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return "No OCR text was extracted. Open the original document and review it manually."
    if not terms:
        return text[:limit] + ("…" if len(text) > limit else "")
    low = text.casefold()
    best = max((low.find(term) for term in terms), default=-1)
    if best < 0:
        return text[:limit] + ("…" if len(text) > limit else "")
    start = max(0, best - 170)
    return ("…" if start else "") + text[start:start + limit] + ("…" if start + limit < len(text) else "")



@router.get("/config")
def config_status(user: User = Depends(get_current_user)):
    return {
        "openrouter_configured": openrouter_configured(),
        "chat_model_id": settings.openrouter_chat_model_id if openrouter_configured() else None,
        "summary_model_id": settings.openrouter_summary_model_id if openrouter_configured() else None,
        "vision_model_id": settings.openrouter_vision_model_id if openrouter_configured() else None,
        "external_model_configured": openrouter_configured(),
        "model_id": settings.openrouter_chat_model_id if openrouter_configured() else None,
        "assistant_provider": "openrouter" if openrouter_configured() else "local_fallback",
        "vision_model_configured": openrouter_configured(),
        "openrouter_configured": openrouter_configured(),
        "local_qwen_configured": local_qwen_extractor_configured(),
        "local_qwen_base_model_id": settings.local_qwen_base_model_id,
    }


@router.post("/chat")
def chat(p: ChatRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    # Product/help questions are a separate, non-patient-data path. This lets the
    # assistant explain the website instead of incorrectly refusing all non-record queries.
    if _is_healthsync_product_question(p.question):
        answer = None
        provider_error = None
        if p.use_external_ai and openrouter_configured():
            try:
                answer = openrouter_product_answer(p.question)
                mode = "openrouter_product_help"
            except RuntimeError as exc:
                provider_error = str(exc)[:240]
                mode = "product_help_fallback_after_ai_error"
        else:
            mode = "product_help_fallback"
        if not answer:
            answer = _product_help_fallback(p.question)
            if provider_error:
                answer += "\n\nI couldn't reach the hosted assistant just now, so this answer uses the built-in HEALTHSYNC-AI feature guide."
        return {
            "answer": answer, "mode": mode, "sources": [],
            # No saved medical-record context is sent on this product FAQ route.
            "external_data_sent": mode == "openrouter_product_help",
            "medical_record_context_sent": False,
            "provider": "openrouter" if mode == "openrouter_product_help" else "built_in_guide",
            "disclaimer": "Product guidance reflects the configured prototype. Some functions depend on deployment settings; this assistant does not diagnose or prescribe."
        }

    if not _is_record_question(p.question):
        return {
            "answer": "I can help explain HEALTHSYNC-AI features and how to use the website, or answer questions grounded in your uploaded medical records. Try asking about record categories, the health timeline, reminders, your profile, AI analysis, privacy, or secure sharing.",
            "mode": "scope_refusal", "sources": [], "external_data_sent": False,
            "disclaimer": "This assistant does not provide diagnosis or treatment recommendations.",
        }
    if _requests_new_treatment_advice(p.question):
        return {
            "answer": "I can help locate and report exactly what your saved prescription says, but I cannot recommend a medicine or tell you to start, stop, combine, or change a dose. Please confirm medication decisions with your prescriber or pharmacist. If you want, ask: ‘What medicine and dose are written on my latest prescription?’",
            "mode": "clinical_advice_refusal", "sources": [], "external_data_sent": False,
            "disclaimer": "Prescription transcription can be inaccurate. Verify the original document with a healthcare professional.",
        }
    terms = _tokens(p.question)
    query = db.query(MedicalRecord).filter(MedicalRecord.owner_user_id == user.id)
    if p.record_ids:
        ids = []
        import uuid
        for raw in p.record_ids:
            try:
                ids.append(uuid.UUID(raw))
            except ValueError:
                raise HTTPException(400, "One of the selected record IDs is invalid.")
        query = query.filter(MedicalRecord.id.in_(ids))
    all_records = query.order_by(MedicalRecord.created_at.desc()).limit(100).all()
    ranked = []
    for record in all_records:
        record_data = record.extracted_data or {}
        structured_context = json.dumps(record_data.get("clinical_analysis", {}), ensure_ascii=False) if record_data.get("clinical_analysis") else ""
        candidate_context = json.dumps(record_data.get("candidates", []), ensure_ascii=False)
        searchable = f"{record.title} {record.record_type} {record.original_filename} {record.extracted_text or ''} {structured_context} {candidate_context}".casefold().replace("_", " ").replace("-", " ")
        # Expand common record categories so natural user wording can find records by type.
        category_aliases = {
            "lab_report": "laboratory lab report test result blood count glucose hemoglobin",
            "prescription": "prescription medicine medication drug prescribed dose strength route frequency",
            "medication": "prescription medicine medication drug prescribed dose strength route frequency",
            "vaccination": "vaccine vaccination immunization dose booster",
            "allergy": "allergy allergen reaction sensitivity",
            "imaging": "imaging scan xray ultrasound radiology",
        }
        searchable += " " + category_aliases.get(str(record.record_type or "").casefold(), "")
        score = sum(searchable.count(t) for t in terms)
        if score or not terms:
            ranked.append((score, record))
    ranked.sort(key=lambda pair: (pair[0], pair[1].created_at.isoformat() if pair[1].created_at else ""), reverse=True)
    chosen = [r for score, r in ranked if score > 0][:3] if terms else all_records[:3]
    if not chosen:
        return {
            "answer": "I could not find a matching uploaded medical record for that question. Upload the relevant report or ask about wording that appears in a saved record. I will not fill gaps with general medical knowledge.",
            "mode": "no_matching_medical_record", "sources": [], "external_data_sent": False,
            "disclaimer": "This assistant reports saved records; it does not diagnose or recommend treatment."
        }
    sources = []
    context_parts = []
    for record in chosen:
        snippet = _short_excerpt(record.extracted_text or "", terms)
        record_data = record.extracted_data or {}
        stored_summary = record_data.get("summary", "")
        clinical_analysis = record_data.get("clinical_analysis") or {}
        item = {"record_id": str(record.id), "title": record.title, "record_type": record.record_type,
                "record_date": record.record_date.isoformat() if record.record_date else None,
                "extraction_method": record_data.get("extraction_method"),
                "snippet": snippet, "summary": stored_summary,
                "clinical_analysis_status": clinical_analysis.get("status")}
        sources.append(item)
        structured_json = json.dumps(clinical_analysis, ensure_ascii=False)[:2500] if clinical_analysis else "No structured AI analysis is saved for this record."
        candidates_json = json.dumps(record_data.get("candidates", []), ensure_ascii=False)[:1200]
        source_context = _short_excerpt(record.extracted_text or "", terms, limit=3000)
        context_parts.append(
            f"Record: {record.title} | Type: {record.record_type} | Date: {item['record_date'] or 'not recorded'}\n"
            f"OCR source excerpt (primary evidence; may contain transcription errors): {source_context}\n"
            f"Structured medical extraction JSON (secondary; check its source flags): {structured_json}\n"
            f"Rule-based candidates (unverified unless status says otherwise): {candidates_json}\n"
            f"Stored AI summary (secondary and may be inaccurate): {str(stored_summary)[:1600]}"
        )
    context = "\n\n".join(context_parts)[:14000] or "No relevant records were found for this account."
    answer = None
    mode = "local_record_retrieval"
    provider_errors = []
    if p.use_external_ai and openrouter_configured():
        try:
            answer = openrouter_record_answer(p.question, context)
            mode = "openrouter_record_grounded"
        except RuntimeError as exc:
            provider_errors.append(f"OpenRouter unavailable: {str(exc)[:220]}")

    if answer is None:
        summarization_requested = any(word in p.question.casefold() for word in ("summarize", "summary", "summarise", "overview", "suggest", "recommend"))
        answer = f"I found {len(sources)} matching uploaded medical record(s).\n\n" + "\n\n".join(
            f"• {s['title']} ({s['record_date'] or 'date not recorded'}): {s['summary'] if summarization_requested and s['summary'] else s['snippet']}"
            for s in sources
        )
        if provider_errors:
            answer += "\n\nHosted AI could not be reached, so this is a source-only fallback rather than a generated interpretation. " + "; ".join(provider_errors)
        answer += "\n\nNo diagnosis or treatment recommendation is being made. Verify values against each original document."
        mode = "local_record_retrieval" if not provider_errors else "local_fallback_after_ai_error"
    return {"answer": answer, "mode": mode, "sources": sources,
            "external_data_sent": mode == "openrouter_record_grounded",
            "provider": "openrouter" if mode == "openrouter_record_grounded" else "local",
            "disclaimer": "Answers are grounded only in selected uploaded records. OCR/model output can be wrong; verify the original. This is not diagnosis or treatment advice."}


@router.get("/insights")
def follow_up_insights(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Surface document-mentioned follow-up language without inferring clinical urgency."""
    cue_pattern = re.compile(r"(?i)(follow[ -]?up|repeat (?:the )?(?:test|lab|blood|scan|ultrasound|x-?ray)|recheck|review with (?:your )?(?:doctor|physician|clinician)|return to (?:the )?clinic|next visit|review after|come back|consult (?:a )?(?:doctor|specialist))")
    records = db.query(MedicalRecord).filter(MedicalRecord.owner_user_id == user.id).order_by(MedicalRecord.record_date.desc().nullslast(), MedicalRecord.created_at.desc()).limit(200).all()
    reminders = db.query(Reminder).filter(Reminder.owner_user_id == user.id, Reminder.is_completed.is_(False)).all()
    cues = []
    for record in records:
        for line in (record.extracted_text or "").splitlines():
            clean = " ".join(line.split()).strip()
            if not clean:
                continue
            match = cue_pattern.search(clean)
            if not match:
                continue
            keywords = set(_tokens(clean))
            has_match = any(keywords & set(_tokens(f"{rem.title} {rem.description or ''}")) for rem in reminders)
            cues.append({"record_id": str(record.id), "record_title": record.title,
                         "record_date": record.record_date.isoformat() if record.record_date else None,
                         "excerpt": clean[:400], "matched_phrase": match.group(0),
                         "matching_pending_reminder_found": has_match,
                         "status": "Review this source wording" if has_match else "No similar pending reminder title/text found"})
            if len(cues) >= 30:
                break
        if len(cues) >= 30:
            break
    return {"cues": cues, "pending_reminders": [{"id": str(r.id), "title": r.title, "due_at": r.due_at.isoformat()} for r in reminders],
            "note": "These are phrase matches in OCR text only, not clinical recommendations or proof that a follow-up is overdue. Confirm with a healthcare professional."}
