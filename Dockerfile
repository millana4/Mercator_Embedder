FROM python:3.12-slim

# HF_HUB_DISABLE_XET=1 — скачивание модели обычным HTTP:
# загрузка через Xet останавливалась на ~670 МБ
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_XET=1

WORKDIR /srv

# torch ставится отдельно из CPU-индекса: иначе pip тянет CUDA-сборку (~5 ГБ)
RUN pip install --no-cache-dir torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]