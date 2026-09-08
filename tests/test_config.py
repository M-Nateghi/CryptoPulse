from pathlib import Path

from cryptopulse.config import Settings


def test_settings_use_defaults():
    settings = Settings(_env_file=None)

    assert settings.database_path == Path("data/cryptopulse.db")
    assert settings.binance_base_url == "https://data-api.binance.vision"
    assert settings.gdelt_base_url == "https://api.gdeltproject.org"
    assert settings.request_timeout_seconds == 10.0
    assert settings.log_level == "INFO"


def test_settings_read_environment_variables(monkeypatch, tmp_path):
    database_path = tmp_path / "test.db"
    monkeypatch.setenv("CRYPTOPULSE_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("CRYPTOPULSE_REQUEST_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("CRYPTOPULSE_LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.database_path == database_path
    assert settings.request_timeout_seconds == 20.0
    assert settings.log_level == "DEBUG"
