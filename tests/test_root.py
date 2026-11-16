"""Tests for GET /."""

from fastapi.testclient import TestClient

from app import __version__


def test_root_returns_project_info(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Standards Q&A Assistant"
    assert body["version"] == __version__
    assert "RoHS" in body["description"]
    assert "REACH" in body["description"]
    assert body["environment"] == "test"


def test_root_uses_configured_name(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("APP_NAME", "Phase One Assistant")
    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    with TestClient(create_app()) as configured_client:
        response = configured_client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "Phase One Assistant"
