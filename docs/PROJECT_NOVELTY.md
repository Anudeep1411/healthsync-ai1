# HEALTHSYNC-AI: project contribution and capstone framing

## Problem addressed
Personal medical records may be scattered across lab PDFs, prescriptions, scanned report images, and vaccination documents. HEALTHSYNC-AI gives one account a private place to collect these files and inspect extracted evidence with source context.

## Core contribution: source-grounded Evidence Engine
Rather than presenting a model-generated summary as an unsupported fact, the prototype stores each extracted candidate with a traceable source anchor:

- uploaded file SHA-256 digest;
- extraction method (`pdf_text` or `tesseract_ocr`);
- page and line position when available;
- exact source-line excerpt;
- candidate field, value, and unit;
- heuristic confidence indicator and explicit human-review status.

The Evidence Engine flags multiple values for the same field in one document, shows repeated fields across a patient's own records, and checks whether the uploaded file still matches its stored hash. It makes the extraction trail inspectable, which is the central explainability/provenance feature of this prototype.

## Architecture
1. React/Vite frontend gathers signup, verification, record, vaccine, and reminder actions.
2. FastAPI validates requests and enforces owner-scoped access with JWT bearer authentication.
3. PostgreSQL stores account metadata, pending signup state, records, audit events, vaccinations, and reminders.
4. Original uploads are stored in a persistent volume, keyed by generated filenames rather than user-supplied paths.
5. PyMuPDF reads searchable PDFs. The bundled Tesseract English model handles scanned PDFs and images.
6. Regex-based field candidate extraction keeps values attached to page/line/excerpt provenance.
7. The backend retains source-to-source comparison and OCR diagnostics for traceability; the user-facing analysis screen focuses on readable extracted facts and verification controls, not low-level OCR flags.

## What is and is not novel about this prototype
This bundle implements a practical integration of private document organization, hash-based integrity checking, OCR-assisted extraction, and field-level source provenance. It should not be described as blockchain, immutable storage, cryptographic signatures, clinical validation, or an empirically validated diagnostic AI. SHA-256 detects a mismatch only if the stored digest is trusted and protected. The confidence numbers are heuristic, not statistically calibrated.

## Suggested capstone evaluation
- OCR text quality: compare OCR against manually transcribed ground truth using character/word error rate.
- Candidate extraction: precision, recall, and F1 for supported fields against annotated documents.
- Provenance quality: percentage of candidates with a correct source page and matching excerpt.
- Integrity detection: verify unchanged files pass and deliberately altered files fail hash comparison.
- Workflow completion: signup/OTP/login, upload, evidence review, search, vaccination entry, reminder completion.
- Usability: task completion time and participant feedback, using synthetic or appropriately consented de-identified records.

## Safety boundaries
Use synthetic or appropriately de-identified data for testing. OCR mistakes are possible. Candidate values must be reviewed against original reports by a person. Do not use the application to make medical decisions.


## Extended prototype contribution: reviewable record-to-action pipeline

The expanded build connects four steps that are often demonstrated separately: **capture → extraction → provenance-aware review → patient-confirmed action**. A medical/vaccination image is stored as an original artifact; OCR text and an extractive summary are saved with the record; candidate values can be traced back to source excerpts; and a vaccine due date becomes a reminder only after a person confirms or enters it. Deletion/editing is supported so users can correct the structured record rather than treating model output as immutable truth.

A second distinction is upload-disclosed, page-level visual reading: local Tesseract supplies a fallback/comparison transcript, and after the upload action, the configured vision model processes each rendered/scanned page even when local confidence appears high. This is a product/architecture contribution rather than a claim that the underlying OCR or language models are novel. The system is designed to keep source text available, expose uncertainty, and keep a human in control of extracted fields and dates.

## Recommended evaluation

Build a de-identified or synthetic evaluation set with paired source images and corrected text, multiple document types (typed lab reports, prescriptions, low-light phone images, and handwriting), and ground-truth extracted fields. Report character/word error rate for transcription, precision/recall/F1 for supported fields, date accuracy, citation/source-span correctness, unsupported-claim rate, and how often users correct OCR suggestions. Separate results by typed vs handwritten documents; do not present overall average OCR accuracy as proof that handwritten prescriptions are reliable.


## OpenRouter-based, prescription-focused assistant update

The prototype separates three external-AI actions: vision transcription for difficult handwriting (image input), AI-written summaries (OCR text input), and record-grounded assistant answers (retrieved record excerpts plus the question). OpenRouter is the configured hosted provider used by the app flow; use is disclosed on upload, and credentials remain backend-only. The assistant adds a server-side topic scope check and refuses direct requests for new diagnoses, medication recommendations, or dose changes. This improves application scope and data-flow clarity, but is not model fine-tuning and does not prove clinical accuracy.

## Novelty assessment

The broad concept of AI-assisted personal health records, OCR, reminders, summarization, and secure sharing is not by itself a strong research novelty claim; comparable systems and papers exist. A more defensible project contribution is an evaluated **Provenance-Aware Prescription-to-Action workflow**: preserve the uploaded source and hash, store a transcription with uncertain spans marked, extract candidate medicine fields with page/line/excerpt provenance, make a human verify/correct them, and only then enable patient-managed reminders or structured history. Novelty should be demonstrated with measurable results—handwriting versus printed word-error rate, medication-name/strength/date field F1, provenance correctness, unsupported-summary claims, human correction rate, and reminder-task completion—against a Tesseract-only baseline. Until those experiments are run, describe this as a system-design contribution or prototype contribution, not a proven novel AI model.


## Implemented novelty module: upload-disclosed page-level handwriting transcription

The current code revision adds a **source-aware visual reading pipeline** to keep handwritten content from being ignored merely because local OCR reports a high confidence score:

1. Searchable PDF text is preserved. Each rendered/scanned page is also analyzed visually after the upload action, with the data-processing notice displayed beforehand.
2. Local Tesseract supplies a comparison/fallback transcript and word-confidence diagnostics. The configured vision model provides the enhanced reading for handwriting, marginal annotations, stamps, corrections and tables.
3. The backend stores per-page readings and bounded disagreement diagnostics; a vision failure retains local text and the original file.
4. The text-analysis step organizes the whole document in chunks when it is long, reducing the risk of silently dropping details from later pages. Source excerpts must match the OCR text before structured details are shown as source-matched.
5. Users can confirm, reject, or correct each extracted item. The saved status is user transcription review, not clinician approval or a determination that a prescription is medically appropriate.

Hosted processing may transmit medical images/text externally and can incur usage costs; the upload notice discloses this data flow. Tesseract confidence is a routing/diagnostic signal, not a calibrated probability of correctness. Even vision models can misread cursive, medical abbreviations, medication names, and numbers; verification against the original remains necessary.

### How to demonstrate this contribution

On the same held-out document set, compare (A) local Tesseract only, (B) hosted vision OCR on every eligible page when enabled, and (C) the adaptive local-first strategy. Report CER/WER for transcription; field precision/recall/F1; page-level escalation rate; number of hosted calls; local-vs-hosted disagreement rate; and the percentage of disagreement flags that reviewers judge genuinely ambiguous. Stratify by searchable PDF, printed scan, phone photo, and handwriting. Use patient/document-separated splits and synthetic or authorized de-identified documents. Report measured outcomes only; this is an implemented system-design contribution, not a claim of proven clinical accuracy or a novel base model.


## Local fine-tuned model integration (current code revision)

A trained Qwen3 LoRA/QLoRA adapter can now be loaded by `backend/app/services/local_model_client.py`. If `LOCAL_QWEN_ADAPTER_DIR` points to an adapter containing `adapter_config.json`, uploads run the local adapter on OCR text for evidence-linked structured candidate extraction before considering hosted structured extraction. Candidate values are still passed through the source-excerpt validator; the adapter does not diagnose and it is not automatically accurate merely because it is fine-tuned. The adapter weights are not included in this repository. Set `INSTALL_LOCAL_MODEL=true` when building Docker to install optional local-inference dependencies, and mount the trained adapter directory as described in `ml_training/README.md`.

The app can still fall back to local Tesseract OCR, deterministic candidates, extractive summaries, file hashing, evidence reports, and local record retrieval when hosted providers are unavailable. OpenRouter is the only hosted model in the application path; hosted processing can transmit record content and is disclosed in the app. To substantiate a model contribution, benchmark the actual trained adapter against the regex baseline and hosted base model using held-out, document-separated data, source-span accuracy, field precision/recall/F1, JSON validity, rejection rate, latency, and memory use.
