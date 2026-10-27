> This directory contains optional model-training experiments. They are separate from the normal application runtime. **The app uses OpenRouter vision for handwriting/page reading. When a trained Qwen3-4B LoRA adapter is mounted and local inference dependencies are installed, Qwen can run locally to extract structured fields from OCR text. `HF_TOKEN` is backend-only and can authenticate Hugging Face model downloads; it does not itself enable inference or replace `OPENROUTER_API_KEY`.**

# HEALTHSYNC-AI: Optional Qwen fine-tuning

## Application runtime: OpenRouter

Set only the provider settings needed for the app in the project-root `.env` file (next to `docker-compose.yml`):

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_CHAT_MODEL_ID=openai/gpt-4o-mini
OPENROUTER_SUMMARY_MODEL_ID=openai/gpt-4o-mini
OPENROUTER_VISION_MODEL_ID=qwen/qwen2.5-vl-72b-instruct
```

The application uses local OCR first and OpenRouter vision for pages requiring handwriting assistance. Local Qwen inference is disabled in the current demo. The hosted vision model and chat models run through OpenRouter. Choosing **Analyze** uses OpenRouter to organize the OCR/vision text into source-grounded medicine, result, date, provider, and uncertainty fields. A user can also request a re-read of one selected page. If the API key is missing or OpenRouter fails, local extracted text remains saved and the UI displays an inline status. Qwen2.5-VL handles hosted vision reading through OpenRouter. Qwen3-4B + LoRA is a separate optional training experiment and is not loaded by the current runtime. `HF_TOKEN` is optional for public model downloads and useful for gated/private artifacts.

Keep the OpenRouter key private and only in the backend environment; never place it in React/Vite code. Documents sent to OpenRouter leave your local backend, so enable that behavior only when appropriate for the record and your privacy requirements.

Rebuild/restart after updating `.env`:

```bash
docker compose up --build -d
docker compose logs -f backend
```

The rest of this folder documents a separate optional training exercise. It is not required to run HEALTHSYNC-AI and does not automatically change the model used through OpenRouter.

## 2. Fine-tune Qwen3-4B with QLoRA

Base model: <https://huggingface.co/Qwen/Qwen3-4B>

Training scaffold: `train_qwen_extractor.py` uses Transformers, TRL, PEFT and 4-bit QLoRA. Use a CUDA GPU (for example a Google Colab T4/A100); the training script intentionally refuses to start without CUDA. The MacBook Air M1 is fine for code and data preparation, but is not the recommended machine to fine-tune Qwen3-4B.

The JSONL files shipped in `data/*.example.jsonl` are **synthetic format examples only** and are too small to train a meaningful model. Build a reviewed, permitted, preferably de-identified dataset; avoid sending real patient records to third-party tools without appropriate authorization. Split by patient/document source, and keep an untouched test set. Each JSONL row follows the chat `messages` format. The assistant answer should contain a JSON object like:

```json
{"candidates":[{"field":"hemoglobin","value":"13.2","unit":"g/dL","source_page":1,"source_line":1,"source_excerpt":"Hemoglobin: 13.2 g/dL"}]}
```

Every source excerpt must literally appear in the input OCR text and must contain the value. Include unclear pages and examples where the correct answer is an empty candidate list. The script requires at least 50 training and 10 validation rows unless you explicitly pass `--allow-small-dataset` for a pipeline smoke test only. Those are technical minimums, not evidence of research quality.

### Run in Google Colab

1. Create a new Colab notebook and choose **Runtime → Change runtime type → T4 GPU** (or a stronger CUDA GPU).
2. Upload the HEALTHSYNC-AI ZIP, unzip it, and `cd` to its project folder in the notebook. Alternatively, clone your private repository.
3. Place your reviewed `train.jsonl` and `validation.jsonl` in `ml_training/data/`. Do not use real identifiable patient data in a notebook unless you have the required authorization and privacy controls.
4. Run:

```bash
%cd /content/healthsync-ai-fresh
!pip install -r ml_training/requirements.txt
!python ml_training/train_qwen_extractor.py \
  --model Qwen/Qwen3-4B \
  --train ml_training/data/train.jsonl \
  --validation ml_training/data/validation.jsonl \
  --output ml_training/output/healthsync-qwen3-extractor
```

Download `ml_training/output/healthsync-qwen3-extractor/` from Colab after training, or upload the adapter to Hugging Face Hub using a private repository if needed for model storage. The output is a **LoRA adapter**, not a standalone full model. Uploading it does not change application runtime behavior: the app continues to use OpenRouter for hosted vision and structured extraction unless you explicitly configure and enable the optional local adapter described below.

## 3. Evaluate before claiming results

```bash
python ml_training/evaluate_qwen_extractor.py \
  --model Qwen/Qwen3-4B \
  --adapter ml_training/output/healthsync-qwen3-extractor \
  --data ml_training/data/validation.jsonl
```

Report exact field/value precision, recall and F1, JSON-validity rate, source-support rate, OCR errors and hallucination/rejection rates on a held-out set. These are engineering metrics, not clinical validation. Training completion alone does not establish accuracy.

## Safety and data flow

- OCR and model fields are suggestions, not diagnosis or treatment advice.
- Backend code validates quoted excerpts and values against OCR source text; this does not prove that the OCR text itself is correct.
- Every stored candidate is marked `requires_human_review`.
- The hosted model does not count as fine-tuned until you have trained an adapter and deployed that adapter for inference.


## 4. Use the trained adapter locally in the app (no hosted inference for extraction)

After training and copying the adapter folder onto the backend host, set these values in the root `.env` file:

```dotenv
LOCAL_QWEN_BASE_MODEL_ID=Qwen/Qwen3-4B
LOCAL_QWEN_ADAPTER_DIR=./ml_training/output/healthsync-qwen3-extractor
LOCAL_QWEN_MAX_NEW_TOKENS=700
```

Place the adapter on the host at `ml_training/output/healthsync-qwen3-extractor` (or set `LOCAL_QWEN_ADAPTER_HOST_DIR` to another host directory). For Docker, build with local inference dependencies installed:

```bash
INSTALL_LOCAL_MODEL=true docker compose up --build -d
```

The Docker build installs `backend/requirements-local-model.txt` (`torch`, `transformers`, `peft`, `accelerate`, and `safetensors`), mounts the adapter read-only, and sets `LOCAL_QWEN_ADAPTER_DIR=/app/local_models/healthsync-qwen3-extractor`. For a non-Docker deployment, install that same requirements file in the backend environment and point `LOCAL_QWEN_ADAPTER_DIR` at the adapter directory. Restart the backend. When a valid `adapter_config.json` exists at `LOCAL_QWEN_ADAPTER_DIR`, uploads use the local fine-tuned adapter for structured extraction by default. OCR remains local Tesseract; the app applies the same literal-source-evidence validator to the adapter output. If the adapter is not configured or fails to load, the app retains deterministic regex candidates and reports a warning.

The optional local adapter is a separate experimental structured-extraction path. OpenRouter remains the hosted provider for vision reading and analysis. The adapter is not bundled in the source ZIP; until you train or place a real adapter in the configured directory, Qwen inference is not active. Local model inference can be slower or require more RAM/VRAM; benchmark on your actual deployment before enabling it for all users.
