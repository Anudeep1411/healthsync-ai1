# HEALTHSYNC-AI model architecture and training plan

## Runtime architecture

1. **Text extraction:** PyMuPDF reads searchable PDF text; Tesseract OCR runs locally on scanned PDFs and images. DOCX text, tables and inline embedded images are handled separately.
2. **Handwriting OCR:** after the user uploads a document (with the data-processing notice displayed), local Tesseract and OpenRouter vision both process each rendered/scanned page. The vision transcription is preferred for reading while the local result and disagreement details are retained in record metadata; searchable PDF text remains preserved alongside visual OCR.
3. **Structured extraction:** OpenRouter organizes the extracted text into medicine, test/result, provider/date, explicit written-instruction and uncertainty fields. Each source excerpt is checked against the OCR transcript before being labelled source-matched. Deterministic candidates remain available if hosted extraction is unavailable.
4. **Review:** extracted fields remain provisional until a user checks them against the original. Source-text matching does not prove OCR correctness or clinical truth.
5. **Assistant:** OpenRouter is the only hosted provider used by the application path. Only excerpts retrieved from the current user's records are eligible as context when hosted AI is enabled.

Set `OPENROUTER_API_KEY` in the root `.env`. Provider and model settings remain backend-only. The default handwriting path uses `qwen/qwen2.5-vl-72b-instruct` through OpenRouter. Local Qwen/LoRA inference is disabled by default, so `HF_TOKEN` is not required for the standard hosted workflow.

## Optional local fine-tuning scaffold

Base model: <https://huggingface.co/Qwen/Qwen3-4B>

The repository includes a QLoRA training script (`ml_training/train_qwen_extractor.py`) using Transformers, TRL and PEFT. It expects a CUDA GPU (such as a Colab T4/A100) and reviewed JSONL training/validation datasets. The included example rows are synthetic demonstrations and are not sufficient for meaningful training or research claims. A trained adapter is not bundled.

## Adaptive OCR routing and limitations

Local Tesseract runs first as a fallback and comparison transcript. When the user uploads a document, OpenRouter vision is called for every rendered/scanned page, regardless of Tesseract confidence, because high confidence does not guarantee readable cursive. The backend retains per-page diagnostics for traceability; the normal user-facing flow shows extracted details and review controls rather than technical OCR flags. Confidence is a routing heuristic, not a calibrated probability. Even a vision model can misread cursive, medical abbreviations, medication names, or numbers; verification against the original remains necessary.

## Hosted-data disclosure

During upload, rendered/scanned page images are sent to OpenRouter for vision transcription, as disclosed in the upload UI. After upload, the user lands on Document analysis; OCR text is sent for source-grounded structured extraction and summaries only when the user selects Analyze document. The assistant does retrieval over the current user's saved records at query time; it does not fine-tune a shared model on those records.
