# AI provider and assistant behavior

All hosted inference is routed through OpenRouter. The configured vision model is `qwen/qwen2.5-vl-72b-instruct` for image-based reading of handwriting and scanned pages; the configured chat/summary models are used for product help, structured analysis, summaries and record-grounded answers. Local Qwen inference is disabled by default, and no local LoRA adapter is required for the judge demo.

## HEALTHSYNC-AI product guide in the assistant

The assistant has two distinct routes:

1. **Product/how-to questions:** a canonical guide describes the landing page, dashboard, all nine record categories, document analysis, health timeline, vaccinations, reminders, health profile, AI assistant, settings/security, SHA-256 integrity, and consent-based secure sharing. It does not retrieve or send saved patient-record contents to OpenRouter. If OpenRouter is unavailable, built-in answers provide a reliable fallback for categories and core features.
2. **Personal record questions:** the backend retrieves only the authenticated user’s owner-scoped medical records and passes selected relevant context to the configured OpenRouter model. The response must remain grounded in source snippets and must not diagnose or recommend treatment.

Keep `OPENROUTER_API_KEY` in the private root `.env`. Do not put credentials in React/Vite variables, source control, screenshots, or a shared ZIP. The `.env.example` contains blank secret placeholders.
