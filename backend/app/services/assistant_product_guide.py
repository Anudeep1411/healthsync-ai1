"""Built-in product guidance for HEALTHSYNC-AI, safe to use without AI dependencies."""

def _product_help_fallback(question: str) -> str:
    """Reliable offline FAQ fallback; deliberately contains no patient-specific data."""
    q = question.casefold()
    categories = (
        "1. Lab report — blood tests and laboratory results.",
        "2. Prescription — a prescription document and instructions as written.",
        "3. Vaccination record — immunization certificates and recorded doses.",
        "4. Diagnosis / clinical note — recorded diagnoses and clinician notes.",
        "5. Medication record — medication history.",
        "6. Allergy record — documented allergies or reactions.",
        "7. Hospital visit / discharge — visit summaries and discharge documents.",
        "8. Imaging report — X-ray, ultrasound, scan, and radiology reports.",
        "9. Other document — other relevant health documents.",
    )
    all_categories = "\n".join(categories)
    if any(k in q for k in ("everything", "all features", "all about", "overview", "website of healthsync", "tell me about healthsync", "what can healthsync do", "what does healthsync do")):
        return (
            "HEALTHSYNC-AI is a personal health-record and immunization-management website. Its goal is to bring scattered health documents into one organized place and make them easier to find and review.\n\n"
            "Main website areas:\n"
            "• Dashboard / Overview — record counts, recent documents, vaccination counts, open reminders, and shortcuts.\n"
            "• Medical records — upload and organize documents, edit titles/categories, inspect extracted source text, download originals, and manage records.\n"
            "• Document analysis — run OCR/hosted vision reading and view a plain-language summary plus extracted candidate details; compare them with the original.\n"
            "• Health timeline — browse saved events in date order.\n"
            "• Vaccinations — manage vaccine name, dose, received date, due date, provider, and notes.\n"
            "• Reminders & follow-ups — create date/time tasks and mark them completed.\n"
            "• Health profile — optionally save demographics, allergies, conditions, medications, surgeries, family history, emergency contacts, doctor/hospital, insurance, pharmacy, and notes.\n"
            "• AI health assistant — ask product/how-to questions or questions grounded in your saved records. It does not diagnose or prescribe.\n"
            "• Secure sharing — choose a record scope, recipient, purpose, consent, and an expiry of 24 hours, 3 days, or 7 days; revoke active access. Email sending requires SMTP configuration.\n"
            "• Settings & security — account/security controls. SHA-256 fingerprints can help detect file changes but are not encryption.\n\n"
            "Record categories:\n" + all_categories + "\n\n"
            "Security and limits: access is intended to be scoped to the authenticated user; secure shares are limited, expiring and revocable. Use HTTPS in deployment. Hosted AI may process selected document content. OCR and AI extraction can be wrong, especially handwriting, so verify medicine names, doses, dates, and results against the original. This is an organizer, not a diagnostic or prescribing system."
        )
    if any(k in q for k in ("categor", "types of record", "type of record", "what can i upload", "document type")):
        return "HEALTHSYNC-AI supports these record categories in the upload form:\n" + all_categories + "\n\nUpload PDF, DOC, DOCX, or supported image files. Extracted details are drafts; compare them with the original document."
    if any(k in q for k in ("secure shar", "sharing", "share link", "revoke", "privacy", "security")):
        return (
            "Secure sharing lets you share a selected medical record (or all current records, if chosen) with a named healthcare recipient. "
            "Enter the recipient name and email, state the purpose, explicitly confirm consent, and select an expiry of 24 hours, 3 days, or 7 days. "
            "Active access can be revoked from the Access management area. The backend is designed to scope the link to selected records, store a hash of the share token, and audit access. "
            "Email delivery depends on SMTP configuration and the public app URL. Use HTTPS in deployment. SHA-256 helps detect changed file bytes; it is not encryption and does not guarantee absolute security."
        )
    if "profile" in q:
        return (
            "Open Health profile from the side navigation. It stores optional user-entered details such as date of birth, sex at birth, height, weight, blood group, allergies, conditions, current medications, surgeries, family history, emergency contacts, primary doctor/hospital, insurance, preferred pharmacy, and notes. "
            "Save changes to the authenticated account, and leave unknown fields blank rather than guessing. Only enter information you are comfortable storing."
        )
    if any(k in q for k in ("reminder", "follow-up", "follow up", "appointment")):
        return (
            "Open Reminders & follow-ups to add a title, due date/time, and description for a personal appointment, vaccination, test repeat, or follow-up task. "
            "You can review pending reminders and mark them completed. Reminders organize dates you enter or confirm; they do not determine clinical urgency or create a medical schedule automatically. "
            "Email/push delivery depends on the services configured for the deployment."
        )
    if any(k in q for k in ("timeline", "health line", "chronolog", "history")):
        return (
            "Open Health timeline to review saved health items and events in date order. It helps you navigate history across records; the order and completeness depend on the dates extracted or entered. "
            "Open a record to compare its extracted details with the original, because OCR can misread handwriting or values."
        )
    if any(k in q for k in ("vaccin", "immuniz")):
        return (
            "Open Vaccinations to add, view, edit, or remove vaccination entries. Available fields include vaccine name, dose, date received, next due date, provider, and notes. "
            "You can scan a certificate where that workflow is enabled, but always check the extracted name, dose, and dates against the certificate. The app does not independently decide a clinical vaccination schedule."
        )
    if any(k in q for k in ("integrity", "sha-256", "hash", "tamper")):
        return (
            "HEALTHSYNC-AI calculates a SHA-256 digest for a stored file so its bytes can be compared with a trusted reference later. "
            "A matching hash supports that the file has not changed relative to that reference; it does not prove the content is medically correct, encrypt the file, or by itself make the database tamper-proof."
        )
    if any(k in q for k in ("ai assistant", "assistant", "ai search", "summar", "ocr", "handwrit", "analyz", "analysis")):
        return (
            "The Document analysis page lets you choose a saved document and click Analyze document when ready. OCR and the configured hosted vision model can read printed text, scanned pages, and handwriting, then produce a readable summary and candidate fields. "
            "AI assistance can answer product questions or questions grounded in saved records. It must not diagnose or prescribe. Review medicine names, strengths, doses, decimal points, units, and dates against the original; unclear handwriting should remain uncertain rather than guessed."
        )
    return (
        "HEALTHSYNC-AI is a personal health-record and immunization-management website. It brings medical documents into one place, extracts text from PDF/Word/image uploads, organizes reviewable details, and helps you browse a chronological health timeline. "
        "Its main areas are Dashboard, Medical records, Document analysis, Health timeline, Vaccinations, Reminders & follow-ups, AI health assistant, Health profile, Secure sharing, and Settings & security.\n\n"
        "The upload categories are:\n" + all_categories + "\n\n"
        "Use reminders for dates you choose or confirm, the profile for optional personal health history, and Secure sharing to grant a named recipient time-limited access with consent and revocation. SHA-256 file hashes can help detect changes. OCR/AI results are drafts for human review, not diagnosis, prescription advice, or proof of medical correctness."
    )

