import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.db.session import get_db
from app.models import User, MedicalRecord, AuditEvent
from app.services.evidence import build_evidence_report
from app.services.medical import sha256_file
from app.services.ai import openrouter_configured, openrouter_analyze_medical_record
from app.core.config import settings
from app.services.verification import VerifyItemRequest, apply_item_verification, update_record_verification_status
from pathlib import Path

router = APIRouter(tags=["Evidence Engine"])


@router.post("/analyze/{record_id}")
def analyze(record_id: str, use_openrouter: bool = Query(False), force_vision_page: int | None = Query(None, ge=1, le=20), db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    try:
        rid = uuid.UUID(record_id)
    except ValueError:
        raise HTTPException(400, "Invalid record ID.")
    record = db.query(MedicalRecord).filter(MedicalRecord.id == rid, MedicalRecord.owner_user_id == user.id).first()
    if not record:
        raise HTTPException(404, "Record not found.")
    others = db.query(MedicalRecord).filter(MedicalRecord.owner_user_id == user.id).all()
    if force_vision_page is not None and not use_openrouter:
        raise HTTPException(400, "Enable AI-assisted processing before requesting another handwriting reading.")
    if force_vision_page is not None and force_vision_page > int((record.extracted_data or {}).get("page_count", 1) or 1):
        raise HTTPException(404, "That page is not present in the extracted document.")

    # OpenRouter is the only hosted model provider in the application path.
    # If it fails or is unconfigured, preserve local evidence and OCR for review.
    extracted_data = dict(record.extracted_data or {})
    if not use_openrouter:
        clinical_analysis = {
            "status": "consent_required",
            "message": "The saved source text is available. Allow AI-assisted processing from the upload workflow to organize the document into medicine, result, date, provider and written-instruction fields.",
            "medicines": [], "tests_and_results": [],
            "documented_instructions": {"precautions": [], "temperature": [], "medicine_storage": [], "hospital_follow_up": []},
            "not_specified": ["AI extraction not run because cloud analysis was not approved"],
        }
    elif openrouter_configured() and (record.extracted_text or "").strip():
        try:
            # If a previous vision read failed or the user explicitly asks for a page re-read, retry
            # the affected page before asking the text model to structure the results.
            old_data = dict(record.extracted_data or {})
            old_diagnostics = old_data.get("ocr_quality", [])
            needs_vision_retry = force_vision_page is not None or any(
                bool(d.get("review_recommended") or d.get("cloud_escalation_needed"))
                and (
                    not bool(d.get("cloud_escalated"))
                    or int(d.get("vision_strategy_version") or 0) < 2
                )
                for d in old_diagnostics if isinstance(d, dict)
            )
            if needs_vision_retry and Path(record.storage_key).is_file():
                from app.services.medical import extract_document, extract_candidates
                selected_pages = {force_vision_page} if force_vision_page is not None else None
                refreshed = extract_document(record.storage_key, use_cloud_ocr=True, force_vision_pages=selected_pages)
                old_text = record.extracted_text or ""
                old_data["analysis_ocr_retry_attempted"] = True
                refreshed_pages = refreshed.get("pages", [])
                refreshed_diagnostics = refreshed.get("ocr_quality", old_diagnostics)
                old_data["analysis_ocr_retry_used"] = any(d.get("cloud_escalated") for d in refreshed_diagnostics)
                prior_pages = old_data.get("page_texts", [])
                if selected_pages and isinstance(prior_pages, list) and prior_pages:
                    # A page retry changes only the requested page. Keep the previous hosted
                    # transcription for every other page so one retry cannot degrade the record.
                    page_map = {int(p.get("page", 0)): dict(p) for p in prior_pages if isinstance(p, dict)}
                    diagnostic_map = {int(d.get("page", 0)): dict(d) for d in old_diagnostics if isinstance(d, dict)}
                    for new_page in refreshed_pages:
                        page_no = int(new_page.get("page", 0) or 0)
                        if page_no in selected_pages:
                            page_map[page_no] = dict(new_page)
                    for new_diag in refreshed_diagnostics:
                        page_no = int(new_diag.get("page", 0) or 0)
                        if page_no in selected_pages:
                            diagnostic_map[page_no] = dict(new_diag)
                    merged_pages = [page_map[k] for k in sorted(page_map)]
                    old_data["page_texts"] = merged_pages
                    old_data["ocr_quality"] = [diagnostic_map[k] for k in sorted(diagnostic_map)]
                    new_text = "\n\n".join(str(p.get("text") or "") for p in merged_pages if p.get("text")).strip()
                    pages_for_candidates = merged_pages
                else:
                    # Older records have no per-page cache; preserve the full refresh result.
                    new_text = refreshed.get("text") or ""
                    old_data["page_texts"] = refreshed_pages
                    old_data["ocr_quality"] = refreshed_diagnostics
                    pages_for_candidates = refreshed_pages
                if new_text.strip() and (new_text.strip() != old_text.strip() or old_data["analysis_ocr_retry_used"]):
                    record.extracted_text = new_text
                    old_data["extraction_method"] = refreshed.get("extraction_method", old_data.get("extraction_method"))
                    old_data["page_count"] = refreshed.get("page_count", old_data.get("page_count"))
                    old_data["candidates"] = extract_candidates(pages_for_candidates)
                extracted_data = old_data
                record.extracted_data = extracted_data
            analysis_args = {
                "text": record.extracted_text or "",
                "title": record.title,
                "record_type": record.record_type or "other",
                "date_text": record.record_date.isoformat() if record.record_date else None,
            }
            clinical_analysis = openrouter_analyze_medical_record(**analysis_args)
            clinical_analysis["analysis_error"] = None
        except Exception as exc:
            clinical_analysis = {
                "status": "unavailable",
                "message": "AI structured extraction could not finish. Existing OCR text and local evidence are retained; retry Analyze or compare with the original.",
                "analysis_error": str(exc)[:400],
                "medicines": [], "tests_and_results": [],
                "documented_instructions": {"precautions": [], "temperature": [], "medicine_storage": [], "hospital_follow_up": []},
                "not_specified": ["medicine details", "precautions", "temperature guidance", "medicine storage conditions", "hospital follow-up interval"],
            }
    elif not openrouter_configured():
        clinical_analysis = {
            "status": "not_configured",
            "message": "AI-assisted analysis is unavailable right now. The original and extracted text remain saved for review.",
            "medicines": [], "tests_and_results": [],
            "documented_instructions": {"precautions": [], "temperature": [], "medicine_storage": [], "hospital_follow_up": []},
            "not_specified": ["medicine details", "precautions", "temperature guidance", "medicine storage conditions", "hospital follow-up interval"],
        }
    else:
        clinical_analysis = {
            "status": "insufficient_text",
            "message": "No readable OCR text is available for AI extraction. Retry OCR or review the original manually.",
            "medicines": [], "tests_and_results": [],
            "documented_instructions": {"precautions": [], "temperature": [], "medicine_storage": [], "hospital_follow_up": []},
            "not_specified": ["medicine details", "precautions", "temperature guidance", "medicine storage conditions", "hospital follow-up interval"],
        }
    extracted_data["clinical_analysis"] = clinical_analysis
    # Keep the record-card summary useful after AI analysis instead of leaving only the
    # initial printed-header extractive summary at the top of the Medical Records screen.
    clinical_summary = str(clinical_analysis.get("plain_language_summary") or "").strip()
    if clinical_summary:
        summary_parts = ["Document summary", clinical_summary]
        medicine_rows = clinical_analysis.get("medicines", [])
        if isinstance(medicine_rows, list) and medicine_rows:
            summary_parts.append("Medicine / prescription lines (transcription only)")
            for medicine in medicine_rows[:20]:
                if not isinstance(medicine, dict):
                    continue
                name = str(medicine.get("name_as_written") or "[unclear]").strip()
                details = []
                for field in ("strength_as_written", "dose_as_written", "route_as_written", "frequency_as_written", "duration_as_written", "instructions_as_written"):
                    value = str(medicine.get(field) or "").strip()
                    if value:
                        details.append(value)
                summary_parts.append("• " + name + (" — " + "; ".join(details) if details else ""))
        raw_lines = clinical_analysis.get("prescription_lines", [])
        if isinstance(raw_lines, list) and raw_lines:
            summary_parts.append("Prescription lines detected for human review (raw transcription)")
            for row in raw_lines[:15]:
                if isinstance(row, dict) and row.get("line_as_written"):
                    summary_parts.append("• " + str(row["line_as_written"]).replace("\n", " / ")[:350])
        uncertain_rows = clinical_analysis.get("uncertain_items", [])
        if isinstance(uncertain_rows, list) and uncertain_rows:
            summary_parts.append("Needs handwriting review")
            for item in uncertain_rows[:10]:
                if isinstance(item, dict):
                    summary_parts.append("• " + str(item.get("item") or "[unclear]")[:220])
        summary_parts.append("Compare all text with the original; review status does not certify clinical correctness.")
        extracted_data["summary"] = "\n".join(summary_parts)[:6000]
        extracted_data["summary_method"] = "openrouter_structured_analysis"
    record.extracted_data = extracted_data
    db.add(record)
    db.flush()

    report = build_evidence_report(record, others)
    exists = Path(record.storage_key).is_file()
    current_hash = sha256_file(record.storage_key) if exists else None
    report["integrity"] = {
        "file_exists": exists,
        "hash_matches": bool(current_hash and current_hash == record.file_sha256),
        "stored_sha256": record.file_sha256,
        "current_sha256": current_hash,
    }
    if not report["integrity"]["hash_matches"]:
        report["flags"].append({"type": "integrity_mismatch", "severity": "high_review", "message": "The stored file is missing or its hash differs from the original upload. Do not rely on extracted evidence until reviewed."})
    db.add(AuditEvent(owner_user_id=user.id, record_id=record.id, action="evidence_analyzed", details={"candidate_count": len(report["candidate_values"]), "flag_count": len(report["flags"]), "clinical_analysis_status": clinical_analysis.get("status"), "medicine_candidate_count": len(clinical_analysis.get("medicines", []))}))
    db.commit()
    return report


@router.post("/verify/{record_id}")
def verify_item(record_id: str, payload: VerifyItemRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Save the user's review/correction for one extracted fact without changing source evidence."""
    try:
        rid = uuid.UUID(record_id)
    except ValueError:
        raise HTTPException(400, "Invalid record ID.")
    record = db.query(MedicalRecord).filter(MedicalRecord.id == rid, MedicalRecord.owner_user_id == user.id).first()
    if not record:
        raise HTTPException(404, "Record not found.")
    data = dict(record.extracted_data or {})
    item, data, clinical, saved_status = apply_item_verification(data, payload)
    if payload.item_type != "candidate":
        data["clinical_analysis"] = clinical
    record.extracted_data = data
    counts = update_record_verification_status(record, data)
    db.add(AuditEvent(owner_user_id=user.id, record_id=record.id, action="record_item_verified", details={
        "item_type": payload.item_type, "index": payload.index, "section": payload.section,
        "status": saved_status, "corrected_fields": list(payload.corrected_values.keys()),
    }))
    db.commit()
    return {
        "record_id": str(record.id), "item_type": payload.item_type, "index": payload.index,
        "section": payload.section, "item": item, "verification_status": saved_status,
        "verification_summary": counts,
        "message": "Review saved. This records your comparison with the source; it is not clinical approval."
    }

@router.get("/timeline/{field_name}")
def timeline(field_name: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    records = db.query(MedicalRecord).filter(MedicalRecord.owner_user_id == user.id).order_by(MedicalRecord.record_date.asc().nullslast(), MedicalRecord.created_at.asc()).all()
    points = []
    for record in records:
        for candidate in (record.extracted_data or {}).get("candidates", []):
            if candidate.get("field", "").lower() == field_name.lower():
                points.append({
                    "record_id": str(record.id), "record_title": record.title,
                    "date": record.record_date.isoformat() if record.record_date else (record.created_at.isoformat() if record.created_at else None),
                    "value": candidate.get("value"), "unit": candidate.get("unit", ""),
                    "source_page": candidate.get("source_page"),
                    "source_excerpt": candidate.get("source_excerpt"),
                    "file_sha256": record.file_sha256,
                })
    return {"field": field_name, "points": points, "note": "Historical display only. Values and units require human review; this endpoint does not interpret clinical significance."}
