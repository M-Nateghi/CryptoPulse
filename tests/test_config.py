from pathlib import Path

from cryptopulse.config import Settings


def test_settings_use_defaults():
    settings = Settings(_env_file=None)

    assert settings.database_path == Path("data/cryptopulse.db")
    assert settings.binance_base_url == "https://data-api.binance.vision"
    assert settings.gdelt_base_url == "https://api.gdeltproject.org"
    assert settings.request_timeout_seconds == 10.0
    assert settings.gdelt_request_timeout_seconds == 45.0
    assert settings.openai_api_key is None
    assert settings.openai_model == "gpt-5.6-luna"
    assert settings.openai_timeout_seconds == 30.0
    assert settings.log_level == "INFO"


def test_settings_read_environment_variables(monkeypatch, tmp_path):
    database_path = tmp_path / "test.db"
    monkeypatch.setenv("CRYPTOPULSE_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("CRYPTOPULSE_REQUEST_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("CRYPTOPULSE_GDELT_REQUEST_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("CRYPTOPULSE_OPENAI_API_KEY", "test-secret-key")
    monkeypatch.setenv("CRYPTOPULSE_OPENAI_MODEL", "test-model")
    monkeypatch.setenv("CRYPTOPULSE_OPENAI_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("CRYPTOPULSE_LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.database_path == database_path
    assert settings.request_timeout_seconds == 20.0
    assert settings.gdelt_request_timeout_seconds == 60.0
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "test-secret-key"
    assert "test-secret-key" not in repr(settings)
    assert settings.openai_model == "test-model"
    assert settings.openai_timeout_seconds == 45.0
    assert settings.log_level == "DEBUG"
