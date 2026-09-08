import logging

import pytest

from cryptopulse.logging_config import UtcFormatter, configure_logging


def test_configure_logging_uses_requested_level_and_utc_formatter(monkeypatch):
    captured_options = {}

    def capture_basic_config(**options):
        captured_options.update(options)

    monkeypatch.setattr(logging, "basicConfig", capture_basic_config)

    configure_logging("DEBUG")

    assert captured_options["level"] == logging.DEBUG
    assert captured_options["force"] is True
    handler = captured_options["handlers"][0]
    assert isinstance(handler.formatter, UtcFormatter)


def test_configure_logging_rejects_invalid_level():
    with pytest.raises(ValueError, match="Invalid log level"):
        configure_logging("VERBOSE")
