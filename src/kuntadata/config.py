"""Runtime configuration, read from the environment or a local .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Azure OpenAI. When the endpoint or key is missing the assistant falls back
    # to a deterministic offline responder, so tests and demos run without a
    # subscription and CI needs no secrets.
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_deployment: str = "gpt-4o-mini"

    statfin_cache_dir: str = ".cache/statfin"
    request_timeout: float = 60.0

    @property
    def azure_configured(self) -> bool:
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key)


settings = Settings()
