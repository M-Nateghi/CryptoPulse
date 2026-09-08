import logging
import time


class UtcFormatter(logging.Formatter):
    """Format log timestamps in UTC."""

    converter = time.gmtime


def configure_logging(log_level: str = "INFO") -> None:
    """Configure application-wide console logging."""
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    try:
        numeric_level = levels[log_level.upper()]
    except KeyError:
        raise ValueError(f"Invalid log level: {log_level}") from None

    handler = logging.StreamHandler()
    handler.setFormatter(
        UtcFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
    )

    logging.basicConfig(
        level=numeric_level,
        handlers=[handler],
        force=True,
    )
