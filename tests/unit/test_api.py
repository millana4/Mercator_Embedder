"""Юнит-тесты HTTP-слоя: авторизация и валидация запроса. Модель подменена фейком."""
import pytest
from fastapi.testclient import TestClient

import app.main as main

AUTH = {"Authorization": "Bearer test-key"}


class FakeModel:
    loaded = False

    def __init__(self) -> None:
        self.calls = []

    def encode(self, texts, mode):
        self.calls.append((list(texts), mode))
        return [[0.0] * main.VECTOR_SIZE for _ in texts]


@pytest.fixture
def fake_model(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(main, "model", fake)
    return fake


@pytest.fixture
def client(fake_model):
    return TestClient(main.app)


# ---------- 1. Health без ключа ----------
def test_health_without_key(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is False


# ---------- 2. Авторизация ----------
def test_embed_without_key_rejected(client, fake_model):
    r = client.post("/embed", json={"texts": ["a"]})
    assert r.status_code == 401
    assert fake_model.calls == []


def test_embed_wrong_key_rejected(client, fake_model):
    r = client.post("/embed", json={"texts": ["a"]},
                    headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert fake_model.calls == []


# ---------- 3. Успешный запрос ----------
def test_embed_ok_default_mode_query(client, fake_model):
    r = client.post("/embed", json={"texts": ["a", "b"]}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["dim"] == main.VECTOR_SIZE
    assert len(body["vectors"]) == 2
    assert fake_model.calls == [(["a", "b"], "query")]


def test_embed_passage_mode(client, fake_model):
    r = client.post("/embed", json={"texts": ["a"], "mode": "passage"}, headers=AUTH)
    assert r.status_code == 200
    assert fake_model.calls[0][1] == "passage"


# ---------- 4. Невалидные запросы ----------
@pytest.mark.parametrize(
    "payload",
    [
        {"texts": []},
        {"texts": ["  "]},
        {"texts": ["a"] * (main.MAX_TEXTS + 1)},
        {"texts": ["a"], "mode": "document"},
        {},
    ],
    ids=["empty_list", "blank_text", "too_many", "bad_mode", "no_texts"],
)
def test_embed_invalid_payload(client, fake_model, payload):
    r = client.post("/embed", json=payload, headers=AUTH)
    assert r.status_code == 422
    assert fake_model.calls == []