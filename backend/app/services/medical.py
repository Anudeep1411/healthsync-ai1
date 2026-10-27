"""OCR and source-grounded candidate extraction for HEALTHSYNC-AI.

OCR and extraction are heuristic. They preserve page and excerpt provenance and
must never be treated as a diagnosis or a substitute for human review.
"""
import base64
import difflib
import hashlib
import io
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import fitz
from app.core.config import settings


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ocr_image_with_quality(image):
    """Run local Tesseract OCR and return text plus word-confidence diagnostics.

    Tesseract confidence is a routing signal, not a calibrated probability of correctness.
    """
    import pytesseract
    from pytesseract import Output
    from PIL import ImageOps, ImageFilter

    image = ImageOps.exif_transpose(image.convert("RGB"))
    if max(image.size) < 1800:
        scale = min(2.0, 1800 / max(image.size))
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    gray = ImageOps.autocontrast(ImageOps.grayscale(image)).filter(ImageFilter.MedianFilter(size=3))
    tessdata_dir = settings.tessdata_dir or os.getenv("TESSDATA_DIR")
    config = '--oem 3 --psm 6'
    if tessdata_dir:
        config += f' --tessdata-dir "{tessdata_dir}"'

    text = pytesseract.image_to_string(gray, lang="eng", config=config).strip()
    data = pytesseract.image_to_data(gray, lang="eng", config=config, output_type=Output.DICT)
    if len(text) < 15:
        config_sparse = '--oem 3 --psm 11'
        if tessdata_dir:
            config_sparse += f' --tessdata-dir "{tessdata_dir}"'
        alt = pytesseract.image_to_string(gray, lang="eng", config=config_sparse).strip()
        if len(alt) > len(text):
            text = alt
            data = pytesseract.image_to_data(gray, lang="eng", config=config_sparse, output_type=Output.DICT)

    confidences = []
    recognized_words = []
    for token, raw_conf in zip(data.get("text", []), data.get("conf", [])):
        token = str(token or "").strip()
        if not token:
            continue
        recognized_words.append(token)
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            continue
        if confidence >= 0:
            confidences.append(confidence)

    mean_confidence = round(sum(confidences) / len(confidences), 2) if confidences else None
    low_ratio = round(sum(value < 50 for value in confidences) / len(confidences), 3) if confidences else None
    quality = {
        "local_mean_word_confidence": mean_confidence,
        "local_low_confidence_word_ratio": low_ratio,
        "local_word_count": len(recognized_words),
        "local_text_characters": len(text),
        "confidence_is_calibrated_probability": False,
    }
    return text, quality


def _ocr_image(image):
    """Backward-compatible text-only wrapper for local OCR."""
    return _ocr_image_with_quality(image)[0]


def _should_escalate_ocr(text: str, quality: dict, source_type: str | None = None) -> tuple[bool, str]:
    """Decide whether weak or ambiguous OCR warrants an OpenRouter vision retry."""
    if not (text or "").strip() or len((text or "").strip()) < 15:
        return True, "local OCR returned little or no readable text"
    try:
        threshold = float(os.getenv("OCR_ESCALATION_MIN_CONFIDENCE", "65"))
    except (TypeError, ValueError):
        threshold = 65.0
    threshold = max(0.0, min(100.0, threshold))
    mean_confidence = quality.get("local_mean_word_confidence")
    low_ratio = quality.get("local_low_confidence_word_ratio")
    if mean_confidence is None:
        return True, "local OCR confidence could not be estimated"
    if mean_confidence < threshold:
        return True, f"local mean word confidence {mean_confidence:.1f} is below threshold {threshold:.1f}"
    if low_ratio is not None and low_ratio >= 0.35:
        return True, f"{low_ratio:.0%} of recognized words have low Tesseract confidence"

    # Tesseract can be confidently wrong on cursive prescriptions. This lightweight
    # quality signal can route suspicious output to a handwriting-capable vision model.
    words = re.findall(r"[A-Za-z][A-Za-z0-9./+-]*", (text or ""))
    if len(words) >= 5:
        shorthand = {"mg", "ml", "mcg", "iu", "od", "bd", "bid", "tid", "qid", "hs", "prn", "iv", "im", "sc", "po", "bp", "hr", "wt", "ht", "tab", "cap", "no", "rx", "dx", "hx", "sos", "stat", "am", "pm"}
        suspicious = [w for w in words if len(w) >= 5 and not re.search(r"[aeiouy]", w.casefold()) and w.casefold() not in shorthand]
        if len(suspicious) / max(len(words), 1) >= 0.28:
            return True, "local OCR contains many ambiguous tokens; a handwriting-capable vision pass may help"

    image_sources = {"image", "scanned_pdf_page", "searchable_pdf_with_image_ocr", "docx_embedded_image", "pdf_text_layer_unusable"}
    if source_type in image_sources:
        # Tesseract may assign high confidence to an incorrect reading of cursive.
        # Sparse image-page text is therefore treated as potentially incomplete even
        # when the numeric confidence looks good; vision retries are still opt-in.
        character_count = len((text or "").strip())
        if len(words) < 45 and character_count < 420:
            return True, "short image-page transcription may omit handwritten notes or prescription details"
        if len(words) >= 6:
            medical_shorthand = {"rx", "od", "bd", "bid", "tid", "qid", "hs", "sos", "stat", "tab", "cap", "mg", "ml", "mcg", "iu", "po", "iv", "im", "sc", "after", "before", "food", "daily"}
            shorthand_count = sum(1 for word in words if word.casefold() in medical_shorthand)
            if shorthand_count >= 2 and len(words) <= 32:
                return True, "short medical-note transcription may contain handwriting or prescription shorthand"

    return False, "local OCR quality passed the routing threshold"


def _ocr_disagreement_summary(local_text: str, hosted_text: str, max_segments: int = 5) -> dict:
    """Summarize word-level differences without claiming either transcript is correct."""
    local_tokens = re.findall(r"\w+|[^\w\s]", (local_text or "").lower(), flags=re.UNICODE)
    hosted_tokens = re.findall(r"\w+|[^\w\s]", (hosted_text or "").lower(), flags=re.UNICODE)
    if not local_tokens and not hosted_tokens:
        return {"disagreement_ratio": 0.0, "disagreement_segments": []}
    matcher = difflib.SequenceMatcher(a=local_tokens, b=hosted_tokens, autojunk=False)
    ratio = round(1.0 - matcher.ratio(), 4)
    segments = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        segments.append({
            "local": " ".join(local_tokens[i1:i2])[:180],
            "hosted": " ".join(hosted_tokens[j1:j2])[:180],
            "difference_type": tag,
        })
        if len(segments) >= max_segments:
            break
    return {"disagreement_ratio": ratio, "disagreement_segments": segments}


def _local_then_optional_hosted_ocr(image, *, use_cloud_ocr: bool, page_number: int, source_type: str, force_vision: bool = False) -> tuple[str, str, dict]:
    """Transcribe a visual page, using vision assistance by default for handwriting coverage.

    Local OCR is retained as a fallback and as a comparison transcript. When cloud OCR is
    enabled by the backend workflow, each image/scanned page is sent to the configured
    vision model so legible cursive, marginal notes, stamps and annotations are less likely
    to be silently missed. No model can guarantee every handwritten dosage is readable.
    """
    local_error = None
    try:
        local_text, quality = _ocr_image_with_quality(image)
    except Exception as exc:
        local_text = ""
        quality = {
            "local_mean_word_confidence": None,
            "local_low_confidence_word_ratio": None,
            "local_word_count": 0,
            "local_text_characters": 0,
            "confidence_is_calibrated_probability": False,
        }
        local_error = f"Local text recognition was unavailable ({type(exc).__name__})."

    should_escalate, reason = _should_escalate_ocr(local_text, quality, source_type)
    if use_cloud_ocr:
        # Vision model is the primary assisted path for images and rendered document pages.
        # Do not rely on a high Tesseract confidence score to decide that cursive is correct.
        should_escalate = True
        reason = "page-level handwriting and annotation pass enabled"
    if force_vision:
        should_escalate = True
        reason = "manual page re-read requested"

    diagnostic = {
        "page": page_number,
        "source_type": source_type,
        **quality,
        "local_ocr_error": local_error,
        "cloud_escalation_consented": bool(use_cloud_ocr),
        "cloud_escalation_needed": bool(should_escalate),
        "cloud_escalated": False,
        "cloud_escalation_reason": reason,
        "disagreement_ratio": None,
        "disagreement_segments": [],
        "review_recommended": True,
        "diagnostic_note": "Recognition output may contain handwriting errors. Check medicine names, strengths, doses, units and dates against the original.",
    }
    if not use_cloud_ocr:
        return local_text, ("tesseract_ocr" if local_text else "no_text"), diagnostic

    try:
        hosted_text, hosted_method = _selected_ocr(image, use_cloud_ocr=True, filename=f"page-{page_number}")
        hosted_text = (hosted_text or "").strip()
        diagnostic["cloud_escalated"] = True
        diagnostic["cloud_method"] = hosted_method
        # Persist the reading strategy version so existing records can be upgraded once
        # to full-page + enlarged handwriting crops when they are analyzed again.
        diagnostic["vision_strategy_version"] = 2
        diagnostic["vision_strategy"] = "full_page_plus_enlarged_handwriting_regions"
        if hosted_text:
            diagnostic.update(_ocr_disagreement_summary(local_text, hosted_text))
            diagnostic["review_recommended"] = True
            # Both transcripts are retained for page-level comparison and auditability.
            diagnostic["local_transcription"] = (local_text or "")[:6000]
            diagnostic["hosted_transcription"] = hosted_text[:6000]
            return hosted_text, hosted_method, diagnostic
        diagnostic["cloud_error"] = "Enhanced page reading returned no text; the first reading was retained."
        return local_text, ("tesseract_ocr" if local_text else "no_text"), diagnostic
    except Exception as exc:
        # A hosted failure must not delete/replace the original or make a readable local scan disappear.
        diagnostic["cloud_error"] = str(exc)[:240]
        diagnostic["cloud_fallback_used"] = True
        diagnostic["review_recommended"] = True
        return local_text, ("tesseract_ocr" if local_text else "no_text"), diagnostic


def _selected_ocr(image, *, use_cloud_ocr: bool, filename: str) -> tuple[str, str]:
    """Use the configured vision model when the user consented to assisted OCR.

    If the vision service fails, the caller retains local OCR and records a review diagnostic.
    """
    if not use_cloud_ocr:
        return _ocr_image(image), "tesseract_ocr"

    from app.services.ai import openrouter_configured, openrouter_vision_transcribe
    if not openrouter_configured():
        raise RuntimeError("OpenRouter is not configured. Set OPENROUTER_API_KEY in the project root .env.")
    text = (openrouter_vision_transcribe(image) or "").strip()
    if not text:
        raise RuntimeError("OpenRouter vision OCR returned empty text.")
    return text, "openrouter_vision_ocr"


def _text_layer_has_misencoded_glyphs(text: str) -> bool:
    """Return True when a PDF text layer likely contains broken font-to-Unicode mapping.

    Some hospital/lab PDFs render correctly but extract Latin glyphs as Private Use Area
    characters (for example U+F052 instead of the letter R). Keeping that text alongside a
    good visual OCR result makes downstream AI analysis harder, so use rendered-page text
    instead when a meaningful fraction of the layer is unmapped.
    """
    chars = [char for char in (text or "") if not char.isspace()]
    if not chars:
        return False
    bad = sum(
        0xE000 <= ord(char) <= 0xF8FF
        or 0xF0000 <= ord(char) <= 0xFFFFD
        or 0x100000 <= ord(char) <= 0x10FFFD
        or char == "\ufffd"
        for char in chars
    )
    return (bad / len(chars)) >= 0.02


def _append_machine_readable_codes(text: str, image) -> str:
    """Append decoded QR/barcode payloads when the optional decoder is available.

    This is useful for vaccination certificates and lab labels. A payload is kept as source
    text only; it is not treated as authenticated or clinically validated data.
    """
    try:
        from pyzbar.pyzbar import decode
        decoded = decode(image)
    except Exception:
        # Barcode decoding is optional; never let a decoder/library error discard OCR text.
        return text or ""
    payloads = []
    seen = set()
    for item in decoded:
        raw = getattr(item, "data", b"")
        try:
            payload = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
        except Exception:
            continue
        if payload and payload not in seen:
            payloads.append(payload)
            seen.add(payload)
    if not payloads:
        return text or ""
    section = "[Decoded QR/barcode payload - compare with the original]\n" + "\n".join(payloads)
    return ((text or "").rstrip() + "\n\n" + section).strip()


def _merge_text_layer_and_ocr(text_layer: str, visual_ocr: str) -> str:
    """Preserve searchable PDF text and append OCR lines not already represented.

    Numeric differences are retained so a possible dosage/result mismatch is visible.
    """
    base = (text_layer or "").strip()
    visual = (visual_ocr or "").strip()
    if not base:
        return visual
    if not visual:
        return base

    def norm(line: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", line.casefold())

    base_lines = [line.strip() for line in base.splitlines() if line.strip()]
    base_norm = [norm(line) for line in base_lines]
    extras = []
    for line in visual.splitlines():
        clean = line.strip()
        normalized = norm(clean)
        if not normalized:
            continue
        duplicate = False
        for existing_norm in base_norm:
            if normalized == existing_norm:
                duplicate = True
                break
            # Only suppress near-identical lines when digits also match.
            if re.sub(r"\d", "#", normalized) == re.sub(r"\d", "#", existing_norm):
                from difflib import SequenceMatcher
                if SequenceMatcher(None, normalized, existing_norm).ratio() >= 0.96:
                    duplicate = True
                    break
        if not duplicate and clean not in extras:
            extras.append(clean)
    if extras:
        return base + "\n\n[Additional visual OCR text; compare with the original]\n" + "\n".join(extras)
    return base

def extract_document(path: str, use_cloud_ocr: bool = False, force_vision_pages: set[int] | None = None) -> dict:
    """Extract page by page with local OCR and consented vision-assisted transcription.

    During an opted-in upload, every rendered image/scanned page is examined by the configured
    vision model as well as local OCR so handwriting, marginal notes, stamps and annotations
    are not skipped just because local OCR appears confident. Searchable PDF text is preserved.
    A manual reread can target one selected page. Failures preserve the available transcription.
    """
    from PIL import Image, ImageOps
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    pages: list[dict] = []
    ocr_quality: list[dict] = []
    force_vision_pages = {int(page) for page in (force_vision_pages or set()) if int(page) > 0}

    # Word files may contain floating/anchored images that python-docx cannot expose as inline shapes.
    # When LibreOffice is available, render DOC/DOCX to PDF and use the same page-wise selectable-text
    # plus visual-OCR pipeline as PDFs. Keep direct parsers below as a fallback for minimal installs.
    if suffix in {".doc", ".docx"}:
        import shutil, subprocess, tempfile
        office = shutil.which("libreoffice") or shutil.which("soffice")
        if office:
            with tempfile.TemporaryDirectory(prefix="healthsync-office-") as tmpdir:
                converted = subprocess.run(
                    [office, "--headless", "--convert-to", "pdf", "--outdir", tmpdir, path],
                    capture_output=True, text=True, timeout=60, check=False,
                )
                pdf_path = Path(tmpdir) / f"{file_path.stem}.pdf"
                if converted.returncode == 0 and pdf_path.is_file():
                    rendered = extract_document(str(pdf_path), use_cloud_ocr=use_cloud_ocr, force_vision_pages=force_vision_pages)
                    rendered["document_conversion"] = "libreoffice_to_pdf"
                    return rendered

    if suffix == ".docx":
        try:
            from docx import Document
            document = Document(path)
            parts = []
            for paragraph in document.paragraphs:
                value = " ".join((paragraph.text or "").split())
                if value:
                    parts.append(value)
            # Headers and footers can contain dates, hospital names, page notes or contact details.
            for section in document.sections:
                for paragraph in list(section.header.paragraphs) + list(section.footer.paragraphs):
                    value = " ".join((paragraph.text or "").split())
                    if value and value not in parts:
                        parts.append(value)
            # Preserve table content too; many lab reports use tables.
            for table in document.tables:
                for row in table.rows:
                    cells = [" ".join((cell.text or "").split()) for cell in row.cells]
                    cells = [cell for cell in cells if cell]
                    if cells:
                        parts.append(f"{cells[0]}: {cells[1]}" if len(cells) == 2 else " | ".join(cells))
            text = "\n".join(parts).strip()[:30000]
            pages.append({"page": 1, "text": text, "method": "docx_text"})
            ocr_quality.append({
                "page": 1, "source_type": "word_document_text", "local_word_count": len(text.split()),
                "local_text_characters": len(text), "cloud_escalation_consented": bool(use_cloud_ocr),
                "cloud_escalated": False, "review_recommended": True,
                "diagnostic_note": "Word text and tables extracted; embedded images are OCR-processed separately. Review important fields against the original."
            })
            # DOCX files may contain inserted scans/photos of notes or prescriptions.
            from PIL import Image as PILImage
            seen_images = set()
            for shape in document.inline_shapes:
                try:
                    blip = shape._inline.graphic.graphicData.pic.blipFill.blip
                    relationship_id = blip.embed
                    if relationship_id in seen_images:
                        continue
                    seen_images.add(relationship_id)
                    image_part = document.part.related_parts[relationship_id]
                    with PILImage.open(io.BytesIO(image_part.blob)) as embedded:
                        embedded = ImageOps.exif_transpose(embedded)
                        page_number = len(pages) + 1
                        page_vision_enabled = use_cloud_ocr and (not force_vision_pages or page_number in force_vision_pages)
                        image_text, image_method, diagnostic = _local_then_optional_hosted_ocr(
                            embedded, use_cloud_ocr=page_vision_enabled,
                            page_number=page_number, source_type="docx_embedded_image",
                            force_vision=page_number in force_vision_pages,
                        )
                        image_text = _append_machine_readable_codes(image_text, embedded)
                    pages.append({"page": len(pages) + 1, "text": image_text[:30000], "method": image_method})
                    diagnostic["source_type"] = "docx_embedded_image"
                    ocr_quality.append(diagnostic)
                except Exception as image_exc:
                    ocr_quality.append({
                        "page": len(pages) + 1, "source_type": "docx_embedded_image",
                        "cloud_escalated": False, "review_recommended": True,
                        "cloud_error": f"Embedded image OCR skipped: {str(image_exc)[:180]}"
                    })
        except Exception as exc:
            raise RuntimeError("Could not read this DOCX file. Re-save it as a valid .docx document and retry.") from exc
    elif suffix == ".doc":
        # Legacy binary .doc: extract selectable text and, when available, convert to PDF
        # so embedded/handwritten images are also inspected page by page.
        import shutil, subprocess, tempfile
        antiword_text = ""
        antiword = shutil.which("antiword")
        if antiword:
            converted = subprocess.run([antiword, path], capture_output=True, text=True, timeout=30, check=False)
            antiword_text = (converted.stdout or "").strip()[:30000] if converted.returncode == 0 else ""
        libreoffice = shutil.which("libreoffice") or shutil.which("soffice")
        converted_result = None
        if libreoffice:
            with tempfile.TemporaryDirectory(prefix="healthsync-doc-") as tmpdir:
                converted = subprocess.run(
                    [libreoffice, "--headless", "--convert-to", "pdf", "--outdir", tmpdir, path],
                    capture_output=True, text=True, timeout=60, check=False,
                )
                pdf_path = Path(tmpdir) / f"{file_path.stem}.pdf"
                if converted.returncode == 0 and pdf_path.is_file():
                    converted_result = extract_document(str(pdf_path), use_cloud_ocr=use_cloud_ocr, force_vision_pages=force_vision_pages)
        if converted_result:
            pages.extend(converted_result.get("pages", []))
            ocr_quality.extend(converted_result.get("ocr_quality", []))
            if antiword_text:
                # Preserve antiword-only text absent from the PDF conversion.
                first_page = pages[0] if pages else None
                if first_page:
                    first_page["text"] = _merge_text_layer_and_ocr(antiword_text, first_page.get("text", ""))[:30000]
                    first_page["method"] = "antiword_text+" + str(first_page.get("method", "pdf_ocr"))
                else:
                    pages.append({"page": 1, "text": antiword_text, "method": "antiword_text"})
        elif antiword_text:
            pages.append({"page": 1, "text": antiword_text, "method": "antiword_text"})
            ocr_quality.append({
                "page": 1, "source_type": "legacy_word_document_text", "local_word_count": len(antiword_text.split()),
                "local_text_characters": len(antiword_text), "cloud_escalation_consented": bool(use_cloud_ocr),
                "cloud_escalated": False, "review_recommended": True,
                "diagnostic_note": "Selectable legacy Word text was extracted. Install LibreOffice in the backend container to also OCR embedded/scanned handwriting in .doc files. Review the original document."
            })
        else:
            raise RuntimeError("Legacy .doc extraction needs antiword or LibreOffice. Save as .docx or PDF, or install one of those tools and retry.")
    elif suffix == ".pdf":
        with fitz.open(path) as pdf:
            if not 1 <= pdf.page_count <= 20:
                raise RuntimeError("PDF must contain between 1 and 20 pages for this prototype.")
            for index, page in enumerate(pdf, start=1):
                raw_text_layer = page.get_text("text").strip()
                text_layer_unusable = _text_layer_has_misencoded_glyphs(raw_text_layer)
                # If a custom PDF font has a broken Unicode map, do not pollute the readable
                # transcription with private-use characters; rely on the rendered page instead.
                text_layer = "" if text_layer_unusable else raw_text_layer
                pix = page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), alpha=False)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                try:
                    page_vision_enabled = use_cloud_ocr and (not force_vision_pages or index in force_vision_pages)
                    source_type = (
                        "pdf_text_layer_unusable" if text_layer_unusable else
                        "searchable_pdf_with_image_ocr" if len(text_layer) >= 20 else
                        "scanned_pdf_page"
                    )
                    visual_text, visual_method, diagnostic = _local_then_optional_hosted_ocr(
                        image, use_cloud_ocr=page_vision_enabled, page_number=index,
                        source_type=source_type,
                        force_vision=index in force_vision_pages,
                    )
                    visual_text = _append_machine_readable_codes(visual_text, image)
                finally:
                    image.close()
                if len(text_layer) >= 20:
                    page_text = _merge_text_layer_and_ocr(text_layer, visual_text)
                    method = "pdf_text+" + visual_method
                    diagnostic["source_type"] = "searchable_pdf_with_image_ocr"
                    diagnostic["text_layer_characters"] = len(text_layer)
                    diagnostic["visual_ocr_characters"] = len(visual_text or "")
                    diagnostic["diagnostic_note"] = "Selectable PDF text was preserved and rendered-page OCR was also run to capture handwritten annotations. Compare any OCR differences with the original."
                else:
                    page_text = visual_text
                    method = visual_method
                    if text_layer_unusable:
                        diagnostic["source_type"] = "pdf_text_layer_unusable"
                        diagnostic["text_layer_ignored"] = True
                        diagnostic["text_layer_characters"] = len(raw_text_layer)
                        diagnostic["diagnostic_note"] = "The PDF text layer used a broken font-to-Unicode map, so it was ignored and the rendered page was transcribed instead."
                pages.append({"page": index, "text": (page_text or "")[:30000], "method": method})
                ocr_quality.append(diagnostic)
    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                if image.width * image.height > 25_000_000:
                    raise RuntimeError("Image dimensions exceed the 25-megapixel processing limit.")
                page_vision_enabled = use_cloud_ocr and (not force_vision_pages or 1 in force_vision_pages)
                page_text, method, diagnostic = _local_then_optional_hosted_ocr(
                    image, use_cloud_ocr=page_vision_enabled, page_number=1, source_type="image",
                    force_vision=1 in force_vision_pages,
                )
                page_text = _append_machine_readable_codes(page_text, image)
            pages.append({"page": 1, "text": page_text[:30000], "method": method})
            ocr_quality.append(diagnostic)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Image OCR failed. Check the bundled Tesseract model and file format.") from exc
    else:
        raise ValueError("Unsupported document type. Upload PDF, DOC, DOCX, PNG, JPG, JPEG, WEBP, or TIFF.")

    text = "\n\n".join(p["text"] for p in pages if p["text"]).strip()[:100000]
    methods = sorted({p["method"] for p in pages})
    overall_method = methods[0] if len(methods) == 1 else ("mixed" if methods else "no_text")
    return {
        "text": text,
        "pages": pages,
        "page_count": len(pages),
        "extraction_method": overall_method,
        "ocr_quality": ocr_quality,
    }


# Conservative patterns: field/value evidence only. No diagnosis or clinical classification.
_PATTERNS = [
    ("hemoglobin", r"(?i)\b(?:haemoglobin|hemoglobin|hb)\b\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(g/dl|g/l)?"),
    ("glucose", r"(?i)\b(?:blood glucose|fasting glucose|fasting sugar|glucose)\b\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(mg/dl|mmol/l)?"),
    ("hba1c", r"(?i)\b(?:hba1c|glycated haemoglobin|glycated hemoglobin)\b\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(%)?"),
    ("cholesterol", r"(?i)\b(?:total cholesterol|cholesterol)\b\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(mg/dl|mmol/l)?"),
    ("creatinine", r"(?i)\bcreatinine\b\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(mg/dl|umol/l|µmol/l)?"),
    ("vitamin_d", r"(?i)\bvitamin\s*d(?:\s*25-?oh)?\b\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(ng/ml|nmol/l)?"),
    ("tsh", r"(?i)\b(?:tsh|thyroid stimulating hormone)\b\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(miu/l|uiu/ml)?"),
    ("platelets", r"(?i)\bplatelets?\b\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:10\^?3/?ul|x10\^?3/?ul|thousand/µl)?"),
    ("wbc", r"(?i)\b(?:wbc|white blood cells?)\b\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:10\^?3/?ul|x10\^?3/?ul)?"),
]


def extract_candidates(pages: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    for page in pages:
        for line_number, line in enumerate((page.get("text") or "").splitlines(), start=1):
            clean_line = " ".join(line.split())
            if not clean_line:
                continue
            for field, pattern in _PATTERNS:
                for match in re.finditer(pattern, clean_line):
                    value = match.group(1)
                    unit = (match.group(2) or "").strip()
                    # Avoid identical duplicate matches for the same field/line.
                    if any(c["field"] == field and c["source_page"] == page["page"] and c["source_excerpt"] == clean_line for c in candidates):
                        continue
                    candidates.append({
                        "evidence_id": f"p{page['page']}-l{line_number}-{field}-{len(candidates)+1}",
                        "field": field,
                        "value": value,
                        "unit": unit,
                        "source_page": page["page"],
                        "source_line": line_number,
                        "source_excerpt": clean_line[:500],
                        # Heuristic traceability indicator, not a calibrated probability.
                        "confidence": 0.68 if page.get("method") == "pdf_text" else 0.52,
                        "extraction_method": page.get("method", "unknown"),
                        "status": "requires_human_review",
                    })
    return candidates


def extract_text(path: str) -> str:
    """Backward-compatible convenience wrapper."""
    return extract_document(path)["text"]


def candidates(text: str) -> list[dict]:
    """Compatibility wrapper for text-only candidate extraction."""
    return extract_candidates([{"page": 1, "text": text, "method": "pdf_text"}])


def extractive_summary(text: str, candidates: list[dict] | None = None, max_lines: int = 8) -> str:
    """Create a conservative, stored extractive summary without inventing clinical meaning."""
    lines = [" ".join(line.split()).strip() for line in (text or "").splitlines()]
    lines = [line for line in lines if len(line) >= 4]
    if not lines:
        return "No readable text was extracted. Open the original file and review it manually."
    terms = re.compile(r"(?i)(patient|name|date|report|result|impression|diagnosis|prescription|medicine|medication|dose|vaccine|vaccination|allergy|hemoglobin|glucose|hba1c|cholesterol|creatinine|platelets|wbc|tsh|vitamin|mg/dl|mmol/l|mg|mcg|ml)\b")
    preferred = [line for line in lines if terms.search(line)]
    selected = []
    for line in (preferred + lines):
        if line not in selected:
            selected.append(line)
        if len(selected) >= max_lines:
            break
    summary = "\n".join(f"• {line[:280]}" for line in selected)
    if candidates:
        summary += "\n\nValues detected for review: " + "; ".join(
            f"{c.get('field')}: {c.get('value')} {c.get('unit') or ''} (page {c.get('source_page')})" for c in candidates[:12]
        )
    summary += "\n\nExtractive summary only: compare every value, unit and date with the original document. This is not a diagnosis."
    return summary[:5000]
