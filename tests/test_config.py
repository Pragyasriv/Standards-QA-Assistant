"""Tests for environment-backed settings and logging setup."""

import logging

import pytest
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.main import create_app


def test_settings_load_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_NAME", "Configured Assistant")
    monkeypatch.setenv("APP_ENV", "Production")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("API_PORT", "9001")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.app_name == "Configured Assistant"
    assert settings.app_env == "production"
    assert settings.log_level == "DEBUG"
    assert settings.api_port == 9001


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "verbose")
    get_settings.cache_clear()

    with pytest.raises(ValidationError):
        get_settings()


def test_invalid_port_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PORT", "70000")
    get_settings.cache_clear()

    with pytest.raises(ValidationError):
        get_settings()


def test_configure_logging_sets_root_level() -> None:
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    try:
        configure_logging("ERROR")
        assert root_logger.level == logging.ERROR
        assert root_logger.handlers
    finally:
        root_logger.setLevel(previous_level)


def test_production_disables_api_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    get_settings.cache_clear()

    application = create_app()

    assert application.docs_url is None
    assert application.redoc_url is None
    assert application.openapi_url is None
