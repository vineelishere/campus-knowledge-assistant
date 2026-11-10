from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env lives in the project root, but commands are run from backend/. Look in both places.
# load_dotenv also exports the values to os.environ, which Langfuse and huggingface_hub read.
_BACKEND = Path(__file__).resolve().parents[1]
_ROOT = _BACKEND.parent
for _p in (_ROOT / ".env", _BACKEND / ".env"):
    if _p.exists():
        load_dotenv(_p, override=False)
        break


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    # "development" or "production". In production the app refuses to start with the
    # default JWT secret, so tokens cannot be forged with a known key.
    environment: str = "development"

    database_url: str = "postgresql://campus:campus@localhost:5432/campus"
    jwt_secret: str = "dev-only-change-me"
    jwt_expire_minutes: int = 480

    # Failed /auth/login attempts allowed per email within the window, per API process.
    login_max_attempts: int = 10
    login_window_seconds: int = 300

    # --- Hugging Face (free serverless inference) ---
    hf_token: str = ""
    hf_provider: str = ""  # blank = let huggingface_hub choose ("auto")
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    # Any OpenAI-compatible endpoint; the shipped .env.example points at Gemini:
    #   https://generativelanguage.googleapis.com/v1beta/openai/ with LLM_MODEL=gemini-3.8-flash
    # Ollama, Groq and OpenRouter work the same way. If LLM_BASE_URL is set it is used instead of
    # Hugging Face for the answer model.
    llm_base_url: str = ""
    llm_api_key: str = ""
    # Optional second model, tried when the main one stays overloaded or rate limited.
    llm_fallback_model: str = ""
    # Output token budget. Reasoning models (e.g. Gemini) count their thinking against it.
    llm_max_tokens: int = 2048
    # Optional: "low" / "medium" / "high" (or "none" where supported). Empty = provider default.
    llm_reasoning_effort: str = ""

    # embeddings: "local" (sentence-transformers, no API key) or "hf_api" (hosted, needs HF_TOKEN)
    embed_backend: str = "local"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384

    # reranker: "local" (small cross-encoder, ~90 MB, CPU) or "hf_api"
    rerank_backend: str = "local"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    candidates_per_retriever: int = 30
    rerank_top_n: int = 6
    min_rerank_score: float = 0.0

    input_price_per_mtok: float = 0.0
    output_price_per_mtok: float = 0.0

    allowed_origins: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
