# HEALTHSYNC-AI sample input support

The document pipeline is designed for these input patterns:

- Handwritten prescriptions photographed on a phone.
- Prescriptions with printed letterhead plus handwritten medicine names, schedules, annotations, and signatures.
- Hospital OPD/visit slips with printed fields and handwritten complaint, finding, investigation, and treatment notes.
- Multi-page PDF lab reports with tables, units, reference intervals, comments, and page-level results.
- Vaccination certificates with printed beneficiary/vaccine details and QR/barcode areas.
- Phone screenshots of digital vaccination cards or wallet views.
- DOC/DOCX files, including embedded/scanned images when LibreOffice can render them.

## Reading strategy

1. Extract selectable PDF/Word text when its character mapping is usable.
2. Render visual pages and run local OCR for an offline baseline.
3. After the user uploads the document with the AI-processing notice displayed, use the configured vision model on the full page plus two enlarged, overlapping handwriting crops, including pages where local OCR appears confident. The crops make cursive medicine names, strengths, dose marks, marginal notes, stamps, tables, and annotations easier to inspect than a single downscaled full-page view.
4. Decode QR/barcode payloads when the optional decoder is available. Keep the payload as source text and do not treat it as authenticated health data.
5. Keep page numbers, alternative OCR readings, and review diagnostics so users can compare uncertain details with the original.
6. Send the complete extracted text to structured analysis in chunks for long documents, then merge the source-grounded details.
7. If structured extraction does not split a prescription into medicine fields, preserve likely prescription lines as raw source text for the user to inspect rather than dropping the lines or inventing medicine names. Existing records are re-read once with the crop-based strategy when analyzed if they were created with an older OCR strategy.

## Special PDF font handling

Some lab PDFs look correct when rendered but expose Private Use Area Unicode characters through their selectable text layer due to a broken font mapping. The extractor detects this pattern and discards the corrupted layer for that page, using the rendered page transcription instead. Ordinary readable PDF text remains preserved and is merged with visual OCR to capture handwritten annotations.

## Important limits

- Handwriting recognition is probabilistic. Medicine names, dose numbers, decimals, units, timing, and dates must be compared with the source.
- Structured details are transcription and organization—not a decision that a drug or dose is medically appropriate.
- QR/barcode payloads are recorded as content only. The application does not use them to certify a document's authenticity.
- AI processing sends document images/text to the configured hosted service only when the user agrees to AI-assisted reading. Do not process another person's medical information without permission.
- Unsupported, unreadable, corrupt, or encrypted files should remain reviewable as originals with a clear inline processing status rather than fabricated extracted values.
