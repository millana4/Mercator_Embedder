"""
HTTP-сервис эмбеддингов для ядра Mercator.

POST /embed  — список текстов -> список нормированных векторов (1024),
               требует заголовок Authorization: Bearer <ключ>
GET  /health — состояние сервиса и признак, загружена ли модель (без ключа)
"""
import logging
import os
import secrets
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.model import MODEL_NAME, VECTOR_SIZE, OnDemandModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("embedder")

MAX_TEXTS = 256  # предел текстов в одном запросе

# Межсервисный ключ (тот же, что SERVICE_API_KEY парсера).
# Без него сервис не стартует, чтобы не оказаться открытым.
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY", "")
if not SERVICE_API_KEY:
    raise RuntimeError("SERVICE_API_KEY не задан: сервис не запускается без межсервисного ключа")

app = FastAPI(title="Mercator Embedder")
model = OnDemandModel()

# Схема безопасности — Swagger по ней рисует кнопку "Authorize"
bearer_scheme = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    """Проверка межсервисного ключа (Authorization: Bearer <ключ>)."""
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode(), SERVICE_API_KEY.encode()
    ):
        logger.warning("Отклонён запрос: неверный или отсутствующий ключ")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_TEXTS)
    mode: Literal["query", "passage"] = "query"


class EmbedResponse(BaseModel):
    model: str
    dim: int
    vectors: list[list[float]]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "model_loaded": model.loaded}


# Обычный def (не async): FastAPI выполняет его в пуле потоков,
# долгий расчёт не блокирует обработку /health
@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(verify_api_key)])
def embed(req: EmbedRequest) -> EmbedResponse:
    if any(not t.strip() for t in req.texts):
        raise HTTPException(status_code=422, detail="Пустой текст в списке")
    vectors = model.encode(req.texts, req.mode)
    return EmbedResponse(model=MODEL_NAME, dim=VECTOR_SIZE, vectors=vectors)