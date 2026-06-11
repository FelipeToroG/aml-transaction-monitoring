"""Typed configuration loader for the AML service.

Two configuration surfaces are loaded here:

* **Environment-variable settings** via Pydantic Settings - secrets,
  per-environment overrides, and runtime feature flags. The
  ``Settings`` class is constructed once at API startup and injected
  through FastAPI's dependency system.
* **YAML configs** - the structural configuration that does not vary
  between environments (scoring weights, alert thresholds, cost
  matrix, model search spaces). Each is loaded lazily from disk on
  first access and cached for the life of the process.

Why split env from YAML
-----------------------
Secrets and per-environment knobs belong in environment variables
because that is what every container runtime (Docker, Kubernetes,
ECS) is designed to inject safely. Structural configuration belongs
in YAML because it should be readable, diffable, and version-controlled.
Mixing the two confuses both audiences: operators reading a YAML to
adjust a database URL is friction; code reviewers reading an env-var
schema to understand the cost matrix is friction.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime environment configuration.

    All fields default to safe development values so the service can
    boot locally with only ``ANTHROPIC_API_KEY`` set. Production
    deployments override the deploy-relevant fields via environment
    variables; the field names match the documented variable names in
    ``.env.example``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # Disable pydantic's default "model_" protected namespace so the
        # ``model_path`` field name (which refers to the on-disk ML model
        # artifact, not pydantic's own model machinery) doesn't trip the
        # protected-namespace UserWarning on every Settings instantiation.
        protected_namespaces=(),
    )

    # ----- LLM provider -----
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    langfuse_public_key: str = Field(default="", description="Langfuse tracing public key")
    langfuse_secret_key: str = Field(default="", description="Langfuse tracing secret key")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # ----- Persistence -----
    database_url: str = Field(
        default="sqlite:///./aml_alerts.db",
        description="SQLAlchemy URL. SQLite default; Postgres in production.",
    )

    # ----- Webhook -----
    webhook_url: str = Field(default="", description="Slack-compatible webhook URL")
    webhook_enabled: bool = Field(default=False)

    # ----- Model artifact -----
    model_path: str = Field(default="models/ensemble.pkl")

    # ----- API runtime -----
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    environment: str = Field(default="local")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings instance.

    Cached via ``lru_cache`` so every call after the first is a dict
    lookup. The cache is process-wide because Settings is immutable
    after construction and the env vars do not change at runtime.
    """
    return Settings()


@lru_cache(maxsize=8)
def load_yaml_config(path: str) -> dict[str, Any]:
    """Load and cache a YAML configuration file.

    Cached per path so the configuration is parsed once at first
    access and reused thereafter. Operators who need to reload a
    config without restarting the service should call ``cache_clear``
    explicitly on this function.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            "Verify the project structure or set the correct working directory."
        )
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
