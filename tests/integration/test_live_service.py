"""
Интеграционные тесты против запущенного контейнера emb-app с настоящей моделью.

Ключ берётся из .env проекта. Если контейнер не запущен — тесты пропускаются.
trust_env=False: запросы к localhost идут напрямую, без системного прокси
(VPN-клиент прописывает socks-прокси в переменные окружения).
"""
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration

BASE_URL = "http://localhost:8030"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _read_key() -> str:
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("SERVICE_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


@pytest.fixture(scope="module")
def client():
    try:
        httpx.get(f"{BASE_URL}/health", timeout=3, trust_env=False).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("emb-app не запущен на localhost:8030")
    headers = {"Authorization": f"Bearer {_read_key()}"}
    # таймаут с запасом: первый запрос загружает модель
    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=600, trust_env=False) as c:
        yield c


# ---------- 1. Размерность и нормировка ----------
def test_vector_shape_and_norm(client):
    r = client.post("/embed", json={"texts": ["Блузка женская белая из хлопка"]})
    assert r.status_code == 200
    body = r.json()
    assert body["dim"] == 1024
    v = body["vectors"][0]
    assert len(v) == 1024
    assert abs(_dot(v, v) - 1.0) < 1e-3


# ---------- 2. Похожие ближе непохожих ----------
def test_similar_closer_than_different(client):
    texts = [
        "Блузка женская белая из хлопка",
        "Белая хлопковая рубашка для женщин",
        "Носки мужские шерстяные",
    ]
    v = client.post("/embed", json={"texts": texts}).json()["vectors"]
    assert _dot(v[0], v[1]) > _dot(v[0], v[2])


# ---------- 3. Порядок векторов в пачке ----------
def test_batch_order_preserved(client):
    single = client.post("/embed", json={"texts": ["Носки мужские шерстяные"]}).json()["vectors"][0]
    batch = client.post(
        "/embed",
        json={"texts": ["Блузка женская белая из хлопка", "Носки мужские шерстяные"]},
    ).json()["vectors"]
    assert _dot(batch[1], single) > 0.999