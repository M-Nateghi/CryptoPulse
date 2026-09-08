from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration values used by CryptoPulse."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CRYPTOPULSE_",
        extra="ignore",
    )

    database_path: Path = Path("data/cryptopulse.db")
    binance_base_url: str = "https://data-api.binance.vision"
    gdelt_base_url: str = "https://api.gdeltproject.org"
    request_timeout_seconds: float = Field(default=10.0, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
