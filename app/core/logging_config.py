"""Application logging setup."""

import logging
import sys
from typing import Final

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S%z"


def configure_logging(log_level: str) -> None:
    """Configure root logging once for the process.

    Repeated calls update the level without attaching duplicate handlers.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        root_logger.addHandler(handler)

    for handler in root_logger.handlers:
        handler.setLevel(level)

    # Keep uvicorn loggers aligned with the application level.
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(logger_name).setLevel(level)
