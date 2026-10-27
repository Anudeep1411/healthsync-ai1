# Handwriting and prescription extraction fix

This update targets handwritten prescription images and mixed printed/handwritten medical pages.

## Changes

- OpenRouter vision transcription receives a full-page view plus two enlarged, overlapping body crops with contrast/sharpness enhancement.
- The transcription instructions explicitly require pen-written medicine lines, dose/schedule marks, annotations, findings, and follow-up notes to be transcribed rather than only the printed header.
- Existing records created by an earlier OCR strategy are re-read once with the new strategy when analyzed. Users can still re-read a page manually from the Medical Records view.
- The structured extractor is instructed to preserve medicine lines even if part of the name is unclear, and to mark unknown spans instead of guessing.
- Raw prescription-like source lines remain visible in the analysis when the model does not convert them into separate medication fields.
- After analysis, the record card's stored summary is refreshed with the plain-language summary, extracted medicine lines, and uncertainty notes.
- No diet plans or medical recommendations are generated. Only instructions actually present in the uploaded source are transcribed.

## Validation

- Backend tests: 46 passed.
- Python compilation: passed.
- Docker Compose YAML parse: passed.
- Frontend production build: not run; frontend dependencies were not installed in the workspace.
- Live OpenRouter handwriting accuracy: not verified without the project's API key. The configured `qwen/qwen2.5-vl-72b-instruct` is used through OpenRouter as the vision-capable Qwen reader; actual quality depends on model response and the source image.

## User workflow

1. Rebuild the Docker app with `docker compose up --build -d`.
2. Open the existing record and press **Analyze**. Records with old OCR diagnostics are automatically re-read once with the new image-crop strategy.
3. Expand **Handwriting review by page** and inspect **Compare the two readings** if needed.
4. Review each parsed medicine and any fallback prescription source lines against the original before marking a transcription as matching.

This workflow checks transcription only. It does not decide whether a medicine or dosage is medically correct.
