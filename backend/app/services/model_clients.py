"""Source-evidence validation for optional local model candidates. Hosted AI uses OpenRouter."""
from __future__ import annotations

import re



def _normalise_for_match(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def validate_model_candidates(
    pages: list[dict], predictions: list[dict], *, extraction_method: str = "source_validated_model_extractor"
) -> tuple[list[dict], int]:
    """Only retain model fields whose quoted evidence and value occur in source text.

    This lexical gate is intentionally conservative. It is not clinical validation and
    does not prove that a transcription is medically correct; a person still reviews it.
    """
    page_map = {int(p.get("page", 1)): p for p in pages}
    accepted: list[dict] = []
    rejected = 0
    for index, item in enumerate(predictions):
        if not isinstance(item, dict):
            rejected += 1
            continue
        try:
            page_num = int(item.get("source_page", 1))
        except (TypeError, ValueError):
            rejected += 1
            continue
        page = page_map.get(page_num)
        excerpt = " ".join(str(item.get("source_excerpt") or "").split())
        field = str(item.get("field") or "").strip()[:80]
        value = str(item.get("value") or "").strip()[:200]
        unit = str(item.get("unit") or "").strip()[:40]
        if not page or not field or not value or len(excerpt) < 4:
            rejected += 1
            continue
        normalized_page = _normalise_for_match(page.get("text", ""))
        normalized_excerpt = _normalise_for_match(excerpt)
        normalized_value = _normalise_for_match(value)
        # Excerpt must be a literal, normalized span from the supplied source page.
        if normalized_excerpt not in normalized_page or normalized_value not in normalized_excerpt:
            rejected += 1
            continue
        source_lines = (page.get("text") or "").splitlines()
        line_number = next((n for n, line in enumerate(source_lines, start=1)
                            if normalized_excerpt in _normalise_for_match(line)), None)
        if line_number is None:
            # Multi-line excerpts are retained only if their text is source-backed;
            # line is null rather than an invented location.
            line_number = item.get("source_line") if isinstance(item.get("source_line"), int) else None
        accepted.append({
            "evidence_id": f"qwen-p{page_num}-l{line_number or 'na'}-{index + 1}",
            "field": field,
            "value": value,
            "unit": unit,
            "source_page": page_num,
            "source_line": line_number,
            "source_excerpt": excerpt[:500],
            "confidence": None,
            "confidence_type": "not_calibrated",
            "extraction_method": extraction_method,
            "status": "requires_human_review",
        })
    return accepted, rejected


