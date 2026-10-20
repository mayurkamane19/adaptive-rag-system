"""Logging configuration module for enterprise logging setup."""

import logging
import sys
from app.core.config import settings


def setup_logging() -> None:
    """Configures system-wide structured logging for the application."""
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
    else:
        root_logger.handlers = [console_handler]

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


setup_logging()
logger = logging.getLogger("adaptive_rag")
