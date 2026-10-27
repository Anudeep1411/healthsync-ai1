# HEALTHSYNC-AI — Complete Full-Stack Capstone Project

**One-command startup:** `docker compose up --build`

This archive includes a dedicated animated public landing page, React/Vite authentication and dashboard, FastAPI backend, PostgreSQL configuration, source files, tests, OCR code, and the English Tesseract `eng.traineddata` OCR model. It does **not** require a `.sh` script or a local Python/Node installation to run the application; Docker Desktop is the only runtime prerequisite.

## Run it in a fresh environment

1. Install and start Docker Desktop.
2. Extract this ZIP and open a terminal in the `healthsync-ai-main` folder (the folder containing `docker-compose.yml`).
3. Run:

   ```bash
   docker compose up --build
   ```

4. Wait until the logs report that PostgreSQL is ready and the backend/frontend services have started.
5. Open the app: **http://localhost:5173**
6. The public landing page appears first; choose **Get started** to sign up or **Sign in** for an existing account.
7. Open backend API docs: **http://localhost:8000/docs**

The command builds the backend and frontend images, starts PostgreSQL, waits for health checks, and launches the application. First build requires internet access to fetch base images and package dependencies. The English OCR model is bundled in the ZIP and is copied into the backend image, so it is not downloaded at runtime.

To stop services, press `Ctrl+C`. To keep services running in the background, use `docker compose up --build -d`; view logs using `docker compose logs -f backend`.

## Signup and OTP

- Create an account in the app.
- By default, `OTP_DELIVERY=console` is enabled for local development. The app will explicitly say that email delivery is disabled, and the OTP prints in the backend log. View it with `docker compose logs -f backend`.
- To send real email, create a `.env` file in the project root (same folder as `docker-compose.yml`), set `OTP_DELIVERY=smtp`, and fill `SMTP_USER`, `SMTP_APP_PASSWORD`, and `SMTP_FROM` with your email account and app password. For Gmail, enable 2-Step Verification and create an App Password; do not use your regular account password. Restart with `docker compose up --build -d` after changing `.env`. Then try signup again and check Inbox/Spam. If SMTP fails, signup returns an error instead of falsely claiming the email was sent.
- Enter the six-digit code.
- Only after successful OTP verification is the row inserted into `users` and a JWT returned.
- The default `OTP_DELIVERY=console` is development-only. SMTP settings can be configured in the root `.env` if you need actual email delivery. Never use console delivery in production.

## Features included

- OTP-verified registration with expiry, incorrect-attempt limit, resend cooldown, password hashing, JWT authentication, and protected endpoints.
- Medical document upload for PDF, PNG, JPG/JPEG, WEBP, and TIFF; per-user access isolation; metadata edit/delete; original download; SHA-256 integrity verification; and persistent upload storage.
- Document reading for PDF, DOC/DOCX, and common image formats. After upload with an inline AI-processing notice, every rendered/scanned page is read using local OCR plus a vision-assisted transcription pass that includes the full page and two enlarged, overlapping handwriting crops. This improves coverage of handwritten prescriptions, marginal notes, stamps, and annotations. If a medicine entry is not confidently structured, likely prescription lines are still preserved as raw source text for review. Searchable PDF text is preserved and merged with visual reading. The app saves the available text if enhanced reading fails. No transcription system guarantees perfect doctor-handwriting recognition.
- Vaccination-card scan/import that stores the original card, suggests a vaccine name/date/dose where detectable, requires user confirmation, supports manual entry/edit/delete, and can create a reminder from a next-due date the user confirms.
- AI-assisted document organization creates a plain-language, source-grounded summary, patient/visit details, medicine/prescription fields, tests/results, history, impressions, allergies, procedures, vaccine and appointment details, and a list of uncertain text. Users can correct each field and record whether it matches the original. This is transcription review, not clinical approval.
- Settings page with server-side password change and clear document-review/privacy guidance; model names and provider switches are not exposed in the user-facing interface.
- Evidence Engine with explainable value candidates, source page and line, source excerpt, OCR/text extraction method, heuristic confidence indicator, file integrity signal, multi-value flags, and cross-record comparisons/timelines. This is a source traceability engine, not a diagnosis engine.
- Vaccination records, reminders, search over medical records, audit events, and dashboard UI.
- Health and database health endpoints.

## Novelty / capstone framing

The central differentiator is the **source-grounded Evidence Engine**: extracted candidate facts are linked to page/line/excerpt provenance and the uploaded file's SHA-256 digest; users can review multi-value flags and compare repeated fields across their own records. OCR and extraction confidence values are heuristic rather than medically calibrated. The system highlights items for review and does not declare a diagnosis, label a value as clinically normal/abnormal, or recommend treatment.

## Database and file persistence

PostgreSQL data and uploaded files are stored in Docker volumes. **For this hackathon demo, `DEMO_RESET_DATABASE_ON_START=true` is enabled by default in `docker-compose.yml`. Every backend container start (including `docker compose restart backend`) drops and recreates the PostgreSQL `public` schema and clears uploaded files, so all accounts and records are removed and you must register again.** This is intentionally destructive and exists only to make judge demonstrations repeatable. Do not use it for real patient records.

To preserve demo data between restarts, set `DEMO_RESET_DATABASE_ON_START=false` in the project-root `.env` and recreate the backend. To manually erase everything including named volumes, `docker compose down -v` is also destructive.

## Ports and configuration

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- OpenAPI docs: `http://localhost:8000/docs`
- PostgreSQL host port: `5433` (inside Compose network it uses `postgres:5432`)

Optional: copy `.env.example` to `.env` in the project root. It supports PostgreSQL, JWT, SMTP, AI-provider credentials, `DEMO_RESET_DATABASE_ON_START`, `HF_TOKEN`, and upload size settings. The OpenRouter key is used only by the backend for vision-assisted reading, structured extraction, summaries, product help, and record-grounded responses; never expose it in React or commit it to Git. The configured vision default is `qwen/qwen2.5-vl-72b-instruct` through OpenRouter. `HF_TOKEN` is not required for the default hosted workflow; local Qwen inference and its adapter are disabled by default. The frontend does not show provider names, model IDs, or token setup. For non-demo use, change all development secrets and set `DEMO_RESET_DATABASE_ON_START=false`.

## OpenRouter setup

1. Copy `.env.example` to `.env` in the project root.
2. Set `OPENROUTER_API_KEY=...` for vision-assisted reading, structured extraction, and record-grounded summaries. Keep the key server-side and never put it in React code.
3. Leave `HF_TOKEN` blank for the default hosted OpenRouter workflow. Local Qwen inference is optional and disabled.
4. Run `docker compose up --build -d backend`.

After upload, every rendered/scanned page is sent to the configured vision route to improve handwritten text coverage. The user then lands on Document analysis and chooses when to run structured extraction for a plain-language summary and source-linked fields. If enhanced reading fails, existing/local text is retained and the UI shows an inline status. The user-facing UI intentionally hides model names and provider controls.

**Data flow:** page images and extracted text may be processed by the configured external AI service after upload; this is disclosed directly on the upload screen. Keep results as drafts and compare them with the original. Upload only records you are authorized to process. Never upload real patient information to the reset-on-start demo environment.

## Validation and limitations

This bundle is a development prototype, not a production-certified medical application. OCR can misread values. Candidate matching covers a limited set of common lab labels and requires review. File hashing detects changes relative to the recorded digest but does not itself make an external store immutable. Before production use, add a security review, malware scanning, CSRF strategy if cookie auth is adopted, formal retention/backups, rate limiting, email anti-abuse, stronger operational secrets, audit policies, accessibility review, and applicable privacy/regulatory assessment.

## Project structure

```text
healthsync-ai-main/
├── docker-compose.yml
├── README.md
├── LICENSES/
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── models/tessdata/eng.traineddata
│   └── app/
│       ├── main.py
│       ├── api/deps.py
│       ├── api/v1/router.py
│       ├── api/v1/endpoints/ (auth, records, evidence, vaccinations, reminders, assistant)
│       ├── core/config.py
│       ├── db/session.py
│       ├── models/models.py
│       └── services/ (OTP, OCR, evidence engine)
├── ml_training/
│   ├── train_qwen_extractor.py
│   ├── evaluate_qwen_extractor.py
│   └── data/*.example.jsonl (synthetic schema examples only)
└── frontend/
    ├── Dockerfile
    ├── package.json
    ├── vite.config.js
    └── src/ (React UI, API client, styles)
```

## Frontend experience

The React/Vite frontend uses Tailwind CSS utility classes and a responsive light theme. The public landing page includes a mobile navigation menu, product overview, feature cards, workflow explanation, privacy and consent principles, and a dedicated section covering lab reports, prescriptions, diagnoses, hospital visits, allergies, medications, vaccination records, reminders, AI-assisted retrieval/summaries, and consent-based sharing. Feature descriptions communicate intended workflows; see the prototype limitations below before treating any feature as production-ready.

### Tailwind CSS not appearing in Docker

The frontend Docker image must include both `postcss.config.js` and `tailwind.config.js` so Vite/PostCSS can compile the `@tailwind` directives and utility classes. They are included in the frontend Dockerfile. After replacing an older project version, rebuild the frontend image without cache:

```bash
docker compose build --no-cache frontend
docker compose up -d
docker compose logs -f frontend
```

If running the frontend directly on your computer instead of Docker, run `cd frontend && npm install && npm run dev`.

## Uploads, handwriting OCR and AI configuration

Local OCR can work without an API key. After upload (with an inline AI-processing notice), the backend sends each rendered/scanned page for enhanced handwriting transcription regardless of local confidence. The uploaded text is saved first; the user then selects Analyze document to create a source-grounded summary and reviewable fields. Messy handwriting, abbreviations and low-quality scans can still be misread; compare with the original. Keep the backend key server-side and restart after configuration changes with `docker compose up --build -d`.

This app does not claim to provide clinical decisions. OCR and model output must be reviewed against the original, and schedules are only reminders for dates a user enters/confirms. AI output is not diagnosis or treatment.
## Optional local Qwen fine-tuning scaffold

The default Docker app does not launch local model workers and does not require `MODEL_WORKER_TOKEN`. Vision-assisted handwriting reading, structured extraction, summaries, product help, and record-grounded responses use OpenRouter through backend credentials. The optional Qwen3-4B QLoRA script is a training scaffold only; no fine-tuned adapter is bundled. `HF_TOKEN` is not needed for the standard hosted workflow.

Candidate values are accepted only when the quoted evidence and value appear in source text. This does not prove OCR or clinical correctness; all candidates require human review.


## Novelty: adaptive, source-aware OCR

HEALTHSYNC-AI combines source-linked candidate extraction with local OCR and a vision-assisted second reading of each rendered/scanned page during upload and analysis. The system keeps source text available so a person can check the original before relying on extracted values. See [`docs/PROJECT_NOVELTY.md`](docs/PROJECT_NOVELTY.md) for the contribution framing and evaluation protocol.


### Optional local Qwen experiment (disabled by default)

The optional local extraction path can load a trained Qwen LoRA adapter in the backend, but the default judge demo uses OpenRouter for hosted AI tasks. Local Tesseract remains a fallback and comparison transcript. After training and copying the adapter to `ml_training/output/healthsync-qwen3-extractor`, run:

```bash
INSTALL_LOCAL_MODEL=true docker compose up --build -d
```

Set `LOCAL_QWEN_ADAPTER_HOST_DIR` if your adapter lives elsewhere. Verify that a trained adapter directory and its dependencies are available before expecting local model inference. `HF_TOKEN` can authenticate gated/private model downloads from the backend. The adapter is not bundled and must be trained first. The normal judge demo uses the vision-assisted reading workflow after upload, with an inline data-processing notice.

## Reminder notifications and email

The frontend notification bell shows reminders that are due today or overdue. Opening the bell marks the displayed items as seen for this browser/account; the unread badge then clears. Vaccination next-due dates create linked reminder entries, so they appear in the same notification list.

To send reminder emails, configure `SMTP_HOST`, `SMTP_PORT=465`, `SMTP_USER`, `SMTP_APP_PASSWORD`, and optionally `SMTP_FROM` in the project root `.env`. The backend checks due reminders once per minute and sends one email per reminder due date, logging successful sends in `audit_events`. Email is disabled when SMTP credentials are missing. The API container must remain running for scheduled email delivery. For production, use a durable worker/scheduler and migrate the audit-based send ledger to a dedicated notification table.

### OpenRouter vision retry

After upload, local OCR and the configured vision model both read each rendered/scanned page to improve capture of handwritten notes. If the second reading is unavailable, existing text is preserved and the UI shows an inline status. Configure `OPENROUTER_API_KEY` in the project-root `.env`, then rebuild/restart the backend. Keep API keys server-side; never put them in frontend variables. Model usage may incur costs.

## Structured prescription and care extraction

During upload, local OCR and the configured vision model read rendered/scanned pages. The app then opens Document analysis and waits until the user selects Analyze document to organize source-backed fields into a plain-language summary, document metadata, patient/visit details, additional document facts, medicines and instructions, tests/results, and uncertain items. Every displayed fact retains source text when available and can be marked as matching or not matching the original. Users may correct the transcription and save their review. This is source transcription review, not clinical approval. No diet, meal, exercise, or lifestyle plan is generated. Results are saved to the record and can be used as context by the AI Health Assistant.

Configure `OPENROUTER_API_KEY` in the project-root `.env`. The frontend does not reveal provider names, model names, or token settings. AI-assisted reading is enabled by default after the upload action; the screen clearly discloses that content may be processed by the configured AI service. Not-specified fields are not filled with invented medication doses or follow-up schedules. See [`docs/AI_MEDICAL_ANALYSIS.md`](docs/AI_MEDICAL_ANALYSIS.md) for workflow and limitations.

### Verify extracted information

The Evidence Engine includes per-item verification controls for medicine fields, test results, care instructions, and rule-based candidates. Users can confirm a value against the original, correct its transcription while retaining the source excerpt, or mark it incorrect. The saved state is user review—not clinician approval. See [`docs/VERIFYING_EXTRACTED_INFORMATION.md`](docs/VERIFYING_EXTRACTED_INFORMATION.md).


### Document review workflow updates

- Upload accepts PDF, DOCX, DOC, PNG, JPG/JPEG, WEBP, BMP, GIF, and TIFF. Scanned images and PDF pages use OCR; DOCX paragraphs and tables are extracted as text. Legacy `.doc` uses antiword in Docker, with LibreOffice as a fallback; DOCX or PDF is recommended if running outside the provided container.
- After upload, the user lands on Document analysis and can choose when to analyze. Medical records offers the stored summary and complete extracted text; the analysis page shows organized fields and per-item review controls. Only destructive delete actions use confirmation dialogs.
- AI-assisted reading is enabled by default after the user chooses Upload; an inline notice explains that document images/text may be processed by the configured vision service. Provider details remain hidden in the interface. After upload, the user lands on Document analysis and chooses when to run structured extraction.
- The app does not generate diet, meal, exercise or lifestyle plans. It can still transcribe a source-written instruction if that text appears in the uploaded medical document, so the original content is not silently discarded.

## Sample input support

See [Sample Input Support](docs/SAMPLE_INPUT_SUPPORT.md) for the document types and reading behavior tested for prescriptions, mixed handwritten notes, lab reports, vaccination certificates, QR/barcodes, and digital-card screenshots.



## Current hosted AI pipeline (OpenRouter)

- **Qwen2.5-VL through OpenRouter** reads rendered/scanned pages and handwritten notes. Configure `OPENROUTER_API_KEY` in the project-root `.env`.
- **OpenRouter chat/summary models** handle structured extraction, summaries, product help, and record-grounded answers.
- **Evidence validation** rejects candidate values that cannot be anchored to supplied source excerpts; users still review each item against the original.
- Local Qwen3-4B + LoRA is an optional separate experiment and is disabled by default. `HF_TOKEN` is not required for the hosted workflow.

The HEALTHSYNC-AI vector logo is available at `frontend/public/healthsync-logo.svg` and is used in the landing page and shared navigation branding.
