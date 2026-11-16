"""Tests for structured API error handling."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.main import create_app


@pytest.fixture
def application(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    get_settings.cache_clear()
    app = create_app()

    @app.get("/_test/boom")
    def boom() -> None:
        raise RuntimeError("secret database password")

    @app.get("/_test/app-error")
    def app_error() -> None:
        raise AppError("The requested operation cannot be completed.", status_code=409)

    @app.get("/_test/needs-int")
    def needs_int(value: int) -> dict[str, int]:
        return {"value": value}

    return app


def test_unknown_path_returns_json_error(application: FastAPI) -> None:
    with TestClient(application) as client:
        response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"error": "http_error", "message": "Not Found"}


def test_validation_error_returns_details(application: FastAPI) -> None:
    with TestClient(application) as client:
        response = client.get("/_test/needs-int", params={"value": "abc"})

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["message"] == "Request validation failed."
    assert isinstance(body["details"], list)
    assert body["details"]


def test_app_error_uses_status_and_message(application: FastAPI) -> None:
    with TestClient(application) as client:
        response = client.get("/_test/app-error")

    assert response.status_code == 409
    assert response.json() == {
        "error": "application_error",
        "message": "The requested operation cannot be completed.",
    }


def test_unexpected_error_hides_internal_details(application: FastAPI) -> None:
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get("/_test/boom")

    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_error"
    assert body["message"] == "An unexpected error occurred."
    assert "secret database password" not in response.text
