"""Юнит-тесты загрузки по требованию. Настоящая модель подменена фейком."""
import pytest

from app.model import IDLE_SECONDS, OnDemandModel


class _Vectors(list):
    """Имитация numpy-массива: у результата encode вызывается .tolist()."""

    def tolist(self):
        return list(self)


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
        self.calls.append({"texts": list(texts), "normalize": normalize_embeddings})
        return _Vectors([[1.0, 0.0] for _ in texts])


@pytest.fixture
def setup(monkeypatch):
    loaded = []
    m = OnDemandModel(start_unload_thread=False)

    def fake_load():
        fake = FakeSentenceTransformer()
        loaded.append(fake)
        return fake

    monkeypatch.setattr(m, "_load", fake_load)
    return m, loaded


# ---------- 1. Загрузка по требованию ----------
def test_not_loaded_until_first_request(setup):
    m, loaded = setup
    assert m.loaded is False
    assert loaded == []


def test_loads_once_and_adds_query_prefix(setup):
    m, loaded = setup
    m.encode(["a", "b"], "query")
    m.encode(["c"], "query")
    assert len(loaded) == 1
    assert loaded[0].calls[0]["texts"] == ["query: a", "query: b"]
    assert loaded[0].calls[0]["normalize"] is True


def test_passage_prefix(setup):
    m, loaded = setup
    m.encode(["a"], "passage")
    assert loaded[0].calls[0]["texts"] == ["passage: a"]


def test_returns_vector_per_text(setup):
    m, _ = setup
    assert m.encode(["a", "b", "c"], "query") == [[1.0, 0.0]] * 3


# ---------- 2. Выгрузка после простоя ----------
def test_no_unload_before_idle(setup):
    m, _ = setup
    m.encode(["a"], "query")
    assert m.unload_if_idle(now=m._last_used + IDLE_SECONDS - 1) is False
    assert m.loaded is True


def test_unload_after_idle_and_reload(setup):
    m, loaded = setup
    m.encode(["a"], "query")
    assert m.unload_if_idle(now=m._last_used + IDLE_SECONDS + 1) is True
    assert m.loaded is False
    m.encode(["b"], "query")
    assert len(loaded) == 2


def test_unload_when_not_loaded(setup):
    m, _ = setup
    assert m.unload_if_idle(now=10**9) is False