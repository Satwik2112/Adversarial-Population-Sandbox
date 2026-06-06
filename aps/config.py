"""APS configuration loaded from environment variables."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMMode(str, Enum):
    """LLM inference mode."""
    MOCK = "mock"
    LIVE = "live"


class Settings(BaseSettings):
    """Application settings, loaded from .env or environment variables."""

    # --- LLM Keys ---
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    huggingface_api_key: str = Field(default="", description="Hugging Face API token")

    # --- Tier 2: Local inference ---
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")

    # --- Infrastructure ---
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant vector DB URL")

    # --- Observability ---
    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangChain tracing")

    # --- APS-specific ---
    aps_llm_mode: LLMMode = Field(default=LLMMode.LIVE, description="LLM mode: mock or live")
    aps_log_dir: Path = Field(default=Path("logs"), description="Directory for simulation logs")
    aps_default_dissident_ratio: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Default dissident ratio (1%)",
    )

    # --- Tier 1 model selection ---
    tier1_model: str = Field(
        default="gemini/gemini-2.5-pro",
        description="Tier 1 (Frontier) model — used for dissidents, synthesis, reports",
    )
    tier2_model: str = Field(
        default="gemini/gemini-2.5-flash",
        description="Tier 2 (Swarm) model — used for conformist worker agents",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
