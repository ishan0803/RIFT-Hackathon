"""Application configuration using Pydantic BaseSettings."""

from __future__ import annotations

import json
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://aml_user:aml_secret_2024@localhost:5432/aml_network"

    # Synchronous URL for Alembic / Celery (replaces asyncpg with psycopg2)
    @property
    def DATABASE_URL_SYNC(self) -> str:
        return self.DATABASE_URL.replace("+asyncpg", "")

    # ── Redis / Celery ────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Clerk Authentication ──────────────────────────────────
    CLERK_SECRET_KEY: str = ""
    CLERK_ISSUER_URL: str = ""
    CLERK_JWKS_URL: str = ""

    # ── CORS ──────────────────────────────────────────────────
    CORS_ORIGINS: str = '["http://localhost:5173"]'

    @property
    def cors_origin_list(self) -> List[str]:
        try:
            return json.loads(self.CORS_ORIGINS)
        except (json.JSONDecodeError, TypeError):
            return ["http://localhost:5173"]

    # ── File uploads ──────────────────────────────────────────
    UPLOAD_DIR: str = "/app/uploads"
    MAX_UPLOAD_SIZE_MB: int = 100


settings = Settings()
