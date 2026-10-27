import uuid
from pathlib import Path
from datetime import date
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.api.deps import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models import User, MedicalRecord, AuditEvent
from app.services.medical import sha256_file, extract_document, extract_candidates, extractive_summary

router = APIRouter(tags=["Medical Records"])
allowed = {".pdf", ".doc", ".docx", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def out(r):
    return {
        "id": str(r.id), "title": r.title, "record_type": r.record_type,
        "provider_name": r.provider_name, "doctor_name": r.doctor_name,
        "record_date": r.record_date.isoformat() if r.record_date else None,
        "original_filename": r.original_filename, "file_sha256": r.file_sha256,
        "verification_status": r.verification_status,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "extracted_data": r.extracted_data or {}, "extracted_text": r.extracted_text,
    }


@router.post("")
async def upload(
    file: UploadFile = File(...), title: str = Form(...),
    record_type: str = Form("other"), provider_name: str | None = Form(None),
    doctor_name: str | None = Form(None), record_date: date | None = Form(None),
    use_cloud_ocr: bool = Form(True),
    db: Session = Depends(get_db), user: User = Depends(get_current_user),
):
    filename = Path(file.filename or "upload").name
    extension = Path(filename).suffix.lower()
    if extension not in allowed:
        raise HTTPException(415, "Supported files: PDF, DOC, DOCX, PNG, JPG, JPEG, WEBP, BMP, GIF, and TIFF.")
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if not content:
        raise HTTPException(400, "File is empty.")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {settings.max_upload_mb} MB upload limit.")
    if extension == ".docx" and not content.startswith(b"PK\x03\x04"):
        raise HTTPException(415, "Invalid DOCX signature.")
    if extension == ".doc" and not content.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        raise HTTPException(415, "Invalid legacy DOC signature.")
    if extension == ".pdf" and not content.startswith(b"%PDF"):
        raise HTTPException(415, "Invalid PDF signature.")
    if extension in {".jpg", ".jpeg"} and not content.startswith(b"\xff\xd8"):
        raise HTTPException(415, "Invalid JPEG signature.")
    if extension == ".png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(415, "Invalid PNG signature.")
    if extension == ".webp" and (len(content) < 12 or content[:4] != b"RIFF" or content[8:12] != b"WEBP"):
        raise HTTPException(415, "Invalid WEBP signature.")
    if extension in {".tif", ".tiff"} and content[:4] not in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}:
        raise HTTPException(415, "Invalid TIFF signature.")

    folder = Path(settings.upload_dir) / str(user.id)
    folder.mkdir(parents=True, exist_ok=True)
    storage_path = folder / f"{uuid.uuid4().hex}{extension}"
    storage_path.write_bytes(content)
    try:
        extracted = extract_document(str(storage_path), use_cloud_ocr=use_cloud_ocr)
    except Exception as exc:
        storage_path.unlink(missing_ok=True)
        message = str(exc).casefold()
        if "not configured" in message or "no provider token" in message:
            status = 503
        elif use_cloud_ocr or "model worker" in message or "paddleocr-vl" in message:
            status = 502
        else:
            status = 422
        raise HTTPException(status, f"Document extraction failed: {exc}")

    # Keep source-linked regex candidates local. The UI discloses hosted processing before upload;
    # OpenRouter vision reads rendered/scanned pages during upload, while structured extraction
    # is run later only when the user selects Analyze document.
    # A configured local extraction adapter may add candidates, but OpenRouter remains the
    # only hosted AI provider in the application path.
    candidate_values = extract_candidates(extracted["pages"])
    structured_extraction_method = "regex_baseline"
    extraction_warnings = []
    model_candidate_rejections = 0
    ocr_diagnostics = extracted.get("ocr_quality", [])
    escalated_page_numbers = {
        int(item.get("page", 0))
        for item in ocr_diagnostics
        if item.get("cloud_escalated") and item.get("page") is not None
    }
    escalation_needed = any(item.get("cloud_escalation_needed") for item in ocr_diagnostics)

    try:
        from app.services.local_model_client import local_qwen_extractor_configured, local_qwen_extract_candidates
        local_model_enabled = local_qwen_extractor_configured()
    except Exception:
        local_model_enabled = False

    if local_model_enabled:
        try:
            local_candidates, model_candidate_rejections = local_qwen_extract_candidates(extracted["pages"])
            if local_candidates:
                existing_keys = {
                    (str(item.get("field", "")).casefold(), str(item.get("value", "")).casefold(),
                     str(item.get("unit", "")).casefold(), int(item.get("source_page") or 0))
                    for item in candidate_values
                }
                for item in local_candidates:
                    key = (str(item.get("field", "")).casefold(), str(item.get("value", "")).casefold(),
                           str(item.get("unit", "")).casefold(), int(item.get("source_page") or 0))
                    if key not in existing_keys:
                        candidate_values.append(item)
                        existing_keys.add(key)
                structured_extraction_method = "local_qwen_lora_extractor"
            else:
                extraction_warnings.append("Local Qwen returned no evidence-validated fields; deterministic candidates were retained.")
        except Exception as exc:
            extraction_warnings.append(f"Local Qwen extractor unavailable; deterministic candidates retained ({str(exc)[:240]}).")
    elif escalated_page_numbers:
        # OpenRouter vision has already been used for pages whose local OCR was weak.
        # Keep deterministic candidate extraction here; the full source-grounded details
        # are requested by the Evidence endpoint after upload, avoiding a redundant model call.
        extraction_warnings.append("Vision-assisted reading was used on these pages; review its transcription against the original.")
    elif use_cloud_ocr and escalation_needed:
        extraction_warnings.append("Vision-assisted reading was unavailable; local text and deterministic candidates were retained for review.")
    # Keep upload lightweight: the first saved summary is deterministic. A source-grounded
    # plain-language summary is produced by the explicit Evidence analysis request after upload.
    summary_method = "extractive"
    stored_summary = extractive_summary(extracted["text"], candidate_values)
    record = MedicalRecord(
        owner_user_id=user.id, title=title.strip() or filename,
        record_type=record_type, provider_name=provider_name, doctor_name=doctor_name,
        record_date=record_date, storage_key=str(storage_path), original_filename=filename,
        file_sha256=sha256_file(str(storage_path)), extracted_text=extracted["text"],
        extracted_data={
            "candidates": candidate_values,
            "extraction_method": extracted["extraction_method"],
            "page_count": extracted["page_count"],
            "page_texts": [{"page": int(p.get("page", i + 1)), "text": str(p.get("text") or "")[:30000], "method": str(p.get("method") or "unknown")} for i, p in enumerate(extracted.get("pages", []))],
            "ocr_quality": extracted.get("ocr_quality", []),
            "summary": stored_summary,
            "summary_method": summary_method,
            "structured_extraction_method": structured_extraction_method,
            "model_candidate_rejections": model_candidate_rejections,
            "extraction_warnings": extraction_warnings,
        }, verification_status="pending_review",
    )
    try:
        db.add(record)
        db.flush()
        db.add(AuditEvent(owner_user_id=user.id, record_id=record.id, action="record_uploaded", details={"filename": filename, "sha256": record.file_sha256}))
        db.commit()
        db.refresh(record)
    except Exception:
        db.rollback()
        storage_path.unlink(missing_ok=True)
        raise HTTPException(500, "Could not save the uploaded record.")
    return out(record)


@router.get("")
def listing(q: str | None = None, record_type: str | None = None, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    query = db.query(MedicalRecord).filter(MedicalRecord.owner_user_id == user.id)
    if q and q.strip():
        term = f"%{q.strip()}%"
        query = query.filter(or_(MedicalRecord.title.ilike(term), MedicalRecord.original_filename.ilike(term), MedicalRecord.provider_name.ilike(term), MedicalRecord.extracted_text.ilike(term)))
    if record_type and record_type != "all":
        query = query.filter(MedicalRecord.record_type == record_type)
    return [out(r) for r in query.order_by(MedicalRecord.created_at.desc()).all()]


@router.get("/{record_id}")
def get_record(record_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    try:
        rid = uuid.UUID(record_id)
    except ValueError:
        raise HTTPException(400, "Invalid record ID.")
    record = db.query(MedicalRecord).filter(MedicalRecord.id == rid, MedicalRecord.owner_user_id == user.id).first()
    if not record:
        raise HTTPException(404, "Record not found.")
    return out(record)


@router.delete("/{record_id}", status_code=204)
def delete(record_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    try:
        rid = uuid.UUID(record_id)
    except ValueError:
        raise HTTPException(400, "Invalid record ID.")
    record = db.query(MedicalRecord).filter(MedicalRecord.id == rid, MedicalRecord.owner_user_id == user.id).first()
    if not record:
        raise HTTPException(404, "Record not found.")
    Path(record.storage_key).unlink(missing_ok=True)
    db.add(AuditEvent(owner_user_id=user.id, record_id=record.id, action="record_deleted", details={"filename": record.original_filename}))
    db.delete(record)
    db.commit()
    return None


@router.post("/{record_id}/verify-integrity")
def integrity(record_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    try:
        rid = uuid.UUID(record_id)
    except ValueError:
        raise HTTPException(400, "Invalid record ID.")
    record = db.query(MedicalRecord).filter(MedicalRecord.id == rid, MedicalRecord.owner_user_id == user.id).first()
    if not record:
        raise HTTPException(404, "Record not found.")
    exists = Path(record.storage_key).is_file()
    current = sha256_file(record.storage_key) if exists else None
    return {"record_id": str(record.id), "file_exists": exists, "hash_matches": bool(current and current == record.file_sha256), "stored_sha256": record.file_sha256, "current_sha256": current}


class RecordUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    record_type: str | None = Field(default=None, max_length=40)
    provider_name: str | None = Field(default=None, max_length=200)
    doctor_name: str | None = Field(default=None, max_length=200)
    record_date: date | None = None


@router.patch("/{record_id}")
def update_record(record_id: str, p: RecordUpdate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    try:
        rid = uuid.UUID(record_id)
    except ValueError:
        raise HTTPException(400, "Invalid record ID.")
    record = db.query(MedicalRecord).filter(MedicalRecord.id == rid, MedicalRecord.owner_user_id == user.id).first()
    if not record:
        raise HTTPException(404, "Record not found.")
    changes = p.model_dump(exclude_unset=True)
    if "title" in changes and changes["title"] is not None:
        changes["title"] = changes["title"].strip()
        if not changes["title"]:
            raise HTTPException(422, "Title cannot be blank.")
    for key, value in changes.items():
        setattr(record, key, value)
    db.add(AuditEvent(owner_user_id=user.id, record_id=record.id, action="record_metadata_updated", details={"fields": sorted(changes.keys())}))
    db.commit()
    db.refresh(record)
    return out(record)


@router.get("/download/{record_id}")
def download_record(record_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    try:
        rid = uuid.UUID(record_id)
    except ValueError:
        raise HTTPException(400, "Invalid record ID.")
    record = db.query(MedicalRecord).filter(MedicalRecord.id == rid, MedicalRecord.owner_user_id == user.id).first()
    if not record:
        raise HTTPException(404, "Record not found.")
    storage_path = Path(record.storage_key)
    if not storage_path.is_file():
        raise HTTPException(404, "The stored original file could not be found.")
    media_types = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".tif": "image/tiff", ".tiff": "image/tiff"}
    return FileResponse(storage_path, media_type=media_types.get(storage_path.suffix.lower(), "application/octet-stream"), filename=record.original_filename)
