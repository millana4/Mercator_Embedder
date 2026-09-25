# Mercator Embedder

Сервис эмбеддингов проекта Mercator. Превращает тексты карточек в векторы
модели `intfloat/multilingual-e5-large` (1024 числа, нормированные). По близости
векторов ядро отбирает карточки, похожие на референс, который задет пользователь для сборки товарных ниш.
Ядро обращается к сервису внутри своей docker-сети: `http://emb-app:8000/embed`.

Модель загружается при первом запросе и выгружается после 5 минут простоя.
Файлы модели (~2,2 ГБ) скачиваются один раз и хранятся в томе `emb-models`.

## Стек технологий

Python 3.12, FastAPI, sentence-transformers, PyTorch (CPU), Docker Compose, pytest.

## API

- `GET /health` — состояние сервиса, без авторизации.
- `POST /embed` — заголовок `Authorization: Bearer <SERVICE_API_KEY>`.
  Запрос `{"texts": [...], "mode": "query"}`, ответ — векторы в порядке текстов.

Swagger: http://localhost:8030/docs

## Настройки (`.env`)

| Переменная | Значение |
|---|---|
| `CORE_NETWORK` | локально `taskmanager_default`, на сервере `mercator_taskmanager_default` |
| `SERVICE_API_KEY` | значение `PARSER_API_KEY` из `.env` ядра на этой машине |

## Тесты

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/unit -v          # без Docker и модели
python -m pytest tests/integration -v   # против запущенного emb-app
```

## Деплой на сервер

Требуется запущенное ядро.

```bash
git clone https://github.com/millana4/Mercator_Embedder.git ~/mercator_embedder
cd ~/mercator_embedder
cp .env.example .env && nano .env
docker compose up -d --build
curl -s localhost:8030/health
```

Обновление: `git pull && docker compose up -d --build`.
После правки `.env`: `docker compose up -d --force-recreate`.