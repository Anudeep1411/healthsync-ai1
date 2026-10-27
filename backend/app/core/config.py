from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    app_name: str = "HEALTHSYNC-AI API"
    environment: str = "development"
    api_prefix: str = "/api"
    database_url: str = "postgresql+psycopg://healthsync_user:healthsync_dev_password@localhost:5433/healthsync"
    jwt_secret: str = "local-development-secret-change-this-before-deployment-123456"
    jwt_expire_minutes: int = 60
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    upload_dir: str = "./storage/medical_records"
    public_app_url: str = ""
    tessdata_dir: str = str(Path(__file__).resolve().parents[2] / "models" / "tessdata")
    max_upload_mb: int = 15
    # Optional local fine-tuned extractor settings; blank adapter means local inference is disabled.
    hf_token: str = ""  # Backend-only Hugging Face token for model downloads
    local_qwen_base_model_id: str = ""
    local_qwen_adapter_dir: str = ""
    local_qwen_max_new_tokens: int = 700
    openrouter_api_key: str = ""
    openrouter_endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_chat_model_id: str = "openai/gpt-4o-mini"
    openrouter_summary_model_id: str = "openai/gpt-4o-mini"
    openrouter_vision_model_id: str = "qwen/qwen2.5-vl-72b-instruct"
    otp_ttl_minutes: int = 10
    otp_max_attempts: int = 5
    otp_resend_seconds: int = 60
    otp_delivery: str = "console"
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 465
    smtp_user: str = ""
    smtp_app_password: str = ""
    smtp_from: str = ""
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")
    @property
    def cors_origin_list(self):
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]
settings = Settings()
