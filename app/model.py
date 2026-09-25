"""
Модель эмбеддингов multilingual-e5-large с загрузкой по требованию.

Модель загружается при первом запросе и выгружается из памяти, если
не использовалась IDLE_SECONDS секунд. В простое контейнер почти
не занимает памяти.

Префиксы e5: для сравнения карточки с карточкой (симметричная задача)
все тексты кодируются с одним префиксом "query: ".
"""
import ctypes
import gc
import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger("embedder")

MODEL_NAME = "intfloat/multilingual-e5-large"
VECTOR_SIZE = 1024
PREFIXES = {"query": "query: ", "passage": "passage: "}

ENCODE_BATCH_SIZE = 16      # размер внутреннего батча: ограничивает пик памяти
IDLE_SECONDS = 300          # простой до выгрузки модели
UNLOAD_CHECK_SECONDS = 30   # как часто проверять простой


def _release_memory() -> None:
    """Вернуть освобождённую память ОС (glibc держит её у процесса)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


class OnDemandModel:
    def __init__(self, start_unload_thread: bool = True) -> None:
        self._model: "SentenceTransformer | None" = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        if start_unload_thread:
            threading.Thread(target=self._unload_loop, daemon=True).start()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> "SentenceTransformer":
        # Импорт здесь, а не в начале модуля: юнит-тестам не нужен torch
        from sentence_transformers import SentenceTransformer

        logger.info("Загрузка модели %s", MODEL_NAME)
        started = time.monotonic()
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        logger.info("Модель загружена за %.1f с", time.monotonic() - started)
        return model

    def encode(self, texts: list[str], mode: str) -> list[list[float]]:
        prefix = PREFIXES[mode]
        # Блокировка на весь расчёт: запросы идут по одному,
        # в памяти никогда нет двух копий модели
        with self._lock:
            if self._model is None:
                self._model = self._load()

            vectors = self._model.encode(
                [prefix + t for t in texts],
                batch_size=ENCODE_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._last_used = time.monotonic()
        return vectors.tolist()

    def unload_if_idle(self, now: float | None = None) -> bool:
        """Выгрузить модель, если простой превысил IDLE_SECONDS. True — выгружена."""
        now = time.monotonic() if now is None else now
        with self._lock:
            idle = now - self._last_used
            if self._model is None or idle <= IDLE_SECONDS:
                return False
            self._model = None
        gc.collect()
        _release_memory()
        logger.info("Модель выгружена после простоя %.0f с", idle)
        return True

    def _unload_loop(self) -> None:
        while True:
            time.sleep(UNLOAD_CHECK_SECONDS)
            self.unload_if_idle()