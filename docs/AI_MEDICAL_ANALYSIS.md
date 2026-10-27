# HEALTHSYNC-AI document analysis

## User workflow

1. Upload a supported PDF, Word document, or image. Local text extraction and Tesseract OCR run first.
2. AI-assisted page reading runs by default after upload, as stated in the upload notice. Local OCR and OpenRouter vision both read rendered/scanned pages; structured extraction is started separately when the user clicks Analyze document. Searchable PDF text is preserved alongside visual OCR, so pen annotations are not silently dropped.
3. The Evidence Engine saves OCR text, extracted candidates, page diagnostics, source excerpts, and file integrity metadata.
4. When OpenRouter detailed analysis is enabled, the backend sends the extracted text and basic record metadata to the configured OpenRouter model. It source-checks returned excerpts against OCR text before showing extracted fields.
5. Users review medicines, doses, routes, frequencies, test results, dates, provider names, explicitly written instructions, and uncertain items against the original. Verification status is saved after each review/correction.
6. The AI assistant uses OpenRouter only when enabled, and responses are grounded in the user's own retrieved records.

## Information shown

The interface shows medicine details as written, tests/results, provider/date when detected, explicit instructions present in the source, source excerpts, uncertainty flags, and the full extracted text. It does not generate diet plans, exercise plans, general education, diagnoses, or treatment recommendations. If the original document itself contains a food or other care instruction, that sentence may be transcribed as source text and is not a generated plan.

## Provider settings

Set `OPENROUTER_API_KEY` in the project-root `.env`. Configure `OPENROUTER_VISION_MODEL_ID` for visual transcription and `OPENROUTER_SUMMARY_MODEL_ID` for structured extraction. The application has no hosted Hugging Face path; OpenRouter is the only hosted AI provider. Keep provider secrets in the backend environment only.

## Limits

- Handwriting recognition is not guaranteed. The vision model may misread cursive, abbreviations, names, or digits; uncertain text must be manually verified against the image.
- OCR confidence is a routing heuristic, not a calibrated correctness probability.
- A matched source excerpt proves only that the words appear in OCR text, not that OCR interpreted the image correctly.
- OpenRouter vision/analysis may transmit medical page images or OCR text outside the local machine. The upload screen discloses this by default; upload only records you are authorized to process.
- The result is an organizational aid, not diagnosis, prescribing, or clinician verification.
