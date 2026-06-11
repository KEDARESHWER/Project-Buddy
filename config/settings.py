"""
config/settings.py
──────────────────
Centralised settings loaded from environment variables (or a .env file).
Using Pydantic v2's BaseSettings means every value is type-checked at
startup, and missing required keys raise a clear error before any agent
tries to use them.

Usage:
    from config.settings import settings
    print(settings.llm_model)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class JarvisSettings(BaseSettings):
    """
    All configuration for Project Jarvis, sourced from .env or the
    process environment.  Defaults are provided where a missing value
    would not break the system; secrets (API keys) have no default so
    the app fails loudly if they are absent.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",           # Silently ignore unrecognised env vars
        case_sensitive=False,
    )

    # ── LLM ─────────────────────────────────────────────────
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    jarvis_llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        alias="JARVIS_LLM_MODEL",
    )
    jarvis_llm_temperature: float = Field(default=0.2, alias="JARVIS_LLM_TEMPERATURE")
    jarvis_llm_max_tokens: int = Field(default=4096, alias="JARVIS_LLM_MAX_TOKENS")

    # ── ChromaDB ─────────────────────────────────────────────
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma"),
        alias="CHROMA_PERSIST_DIR",
    )
    chroma_collection_user_context: str = Field(
        default="user_context",
        alias="CHROMA_COLLECTION_USER_CONTEXT",
    )
    chroma_collection_tasks: str = Field(
        default="tasks",
        alias="CHROMA_COLLECTION_TASKS",
    )
    chroma_collection_conversations: str = Field(
        default="conversations",
        alias="CHROMA_COLLECTION_CONVERSATIONS",
    )

    # ── Gmail ────────────────────────────────────────────────
    gmail_credentials_path: Path = Field(
        default=Path("./config/gmail_credentials.json"),
        alias="GMAIL_CREDENTIALS_PATH",
    )
    gmail_token_path: Path = Field(
        default=Path("./config/gmail_token.json"),
        alias="GMAIL_TOKEN_PATH",
    )
    gmail_poll_interval_seconds: int = Field(
        default=60,
        alias="GMAIL_POLL_INTERVAL_SECONDS",
    )

    # ── Twilio / WhatsApp ────────────────────────────────────
    twilio_account_sid: str = Field(default="", alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field(default="", alias="TWILIO_AUTH_TOKEN")
    twilio_whatsapp_from: str = Field(
        default="whatsapp:+14155238886",
        alias="TWILIO_WHATSAPP_FROM",
    )
    twilio_whatsapp_to: str = Field(default="", alias="TWILIO_WHATSAPP_TO")

    # ── Webhook Server ───────────────────────────────────────
    webhook_host: str = Field(default="0.0.0.0", alias="WEBHOOK_HOST")
    webhook_port: int = Field(default=8000, alias="WEBHOOK_PORT")
    webhook_secret: str = Field(default="change-me", alias="WEBHOOK_SECRET")

    # ── Docker Sandbox ───────────────────────────────────────
    docker_sandbox_image: str = Field(
        default="python:3.11-slim",
        alias="DOCKER_SANDBOX_IMAGE",
    )
    docker_sandbox_timeout_seconds: int = Field(
        default=30,
        alias="DOCKER_SANDBOX_TIMEOUT_SECONDS",
    )
    docker_sandbox_mem_limit: str = Field(
        default="256m",
        alias="DOCKER_SANDBOX_MEM_LIMIT",
    )
    docker_sandbox_cpu_quota: int = Field(
        default=50000,
        alias="DOCKER_SANDBOX_CPU_QUOTA",
    )

    # ── Redis / Celery ───────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # ── Logging ──────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )
    log_file: Path = Field(default=Path("./logs/jarvis.log"), alias="LOG_FILE")

    # ── Derived helpers ──────────────────────────────────────
    @field_validator("chroma_persist_dir", "log_file", mode="before")
    @classmethod
    def _ensure_parent_dirs(cls, v: str | Path) -> Path:
        """Auto-create parent directories so Jarvis never crashes on a missing path."""
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# Module-level singleton — import this everywhere.
settings = JarvisSettings()
