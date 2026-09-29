"""LLM Engine for Whisper Studio.
Provides GGUF inference via llama-cpp-python with NVIDIA CUDA 12 hardware acceleration,
full GPU offloading, FlashAttention, and token streaming.
"""
import os
import sys
import gc
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator

from core.cuda_utils import setup_cuda_dlls
from core.ensure_llamacpp import ensure_compatible_llamacpp
from core.hf_downloader import get_llm_model_path, is_llm_cached, LLM_REGISTRY
from core.vram_manager import get_vram_manager

# Гарантируем регистрацию DLL путей CUDA 12 и совместимость с CPU (Alder Lake) перед загрузкой llama_cpp
try:
    ensure_compatible_llamacpp()
except Exception:
    pass
setup_cuda_dlls()
try:
    import llama_cpp
except ImportError:
    llama_cpp = None


class LLMCancelledException(Exception):
    """Исключение при прерывании генерации пользователем."""
    pass


class LLMEngine:
    """Обертка над llama-cpp-python для локального инференса GGUF моделей."""

    def __init__(self):
        self.llm = None
        self.loaded_model_id: Optional[str] = None
        self.loaded_model_path: Optional[str] = None
        self.loaded_n_ctx: int = 16384
        self._lock = threading.Lock()

        # Регистрируем движок в VRAM Arbiter
        get_vram_manager().register_llm_engine(self)

    def is_loaded(self) -> bool:
        """Проверяет, загружена ли модель в память."""
        return self.llm is not None

    def load_model(
        self,
        model_id: str,
        n_ctx: int = 16384,
        n_gpu_layers: int = -1,
        flash_attn: bool = True,
        force_reload: bool = False,
    ) -> None:
        """
        Загружает GGUF-модель в GPU VRAM.
        Автоматически освобождает Whisper через VRAM Arbiter.
        """
        if llama_cpp is None:
            raise RuntimeError(
                "Библиотека llama-cpp-python не установлена!\n"
                "Запустите run.bat для автоматической установки с поддержкой CUDA."
            )

        if not is_llm_cached(model_id):
            model_path = get_llm_model_path(model_id)
            raise FileNotFoundError(f"Файл модели не найден по пути: {model_path}. Сначала выполните загрузку.")

        model_path = get_llm_model_path(model_id)

        with self._lock:
            if (
                not force_reload
                and self.llm is not None
                and self.loaded_model_id == model_id
                and self.loaded_n_ctx >= n_ctx
            ):
                return

            # Выгружаем предыдущие модели (включая Whisper) для предотвращения OOM
            get_vram_manager().prepare_for_llm()
            if self.llm is not None:
                del self.llm
                self.llm = None
                gc.collect()

            # Параметры инициализации llama.cpp
            init_kwargs = {
                "model_path": str(model_path.resolve()),
                "n_ctx": n_ctx,
                "n_gpu_layers": n_gpu_layers,
                "verbose": False,
            }

            # FlashAttention поддерживается в новых сборках llama.cpp
            try:
                self.llm = llama_cpp.Llama(
                    **init_kwargs,
                    flash_attn=flash_attn,
                )
            except Exception:
                # Резервная инициализация без flash_attn, если флаг не поддержан
                self.llm = llama_cpp.Llama(**init_kwargs)

            self.loaded_model_id = model_id
            self.loaded_model_path = str(model_path)
            self.loaded_n_ctx = n_ctx

    def unload(self) -> None:
        """Полная выгрузка LLM из VRAM и очистка памяти."""
        with self._lock:
            if self.llm is not None:
                del self.llm
                self.llm = None
            self.loaded_model_id = None
            self.loaded_model_path = None
            gc.collect()

    def count_tokens(self, text: str) -> int:
        """
        Вычисляет точное количество токенов в тексте.
        Если модель загружена, использует токенизатор модели, иначе приближенную оценку.
        """
        if not text:
            return 0
        if self.llm is not None:
            try:
                tokens = self.llm.tokenize(text.encode("utf-8", errors="ignore"))
                return len(tokens)
            except Exception:
                pass
        # Эвристика для русскоязычного текста: ~3-4 символа на токен
        return max(1, len(text) // 3)

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.9,
        cancel_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """
        Потоковая генерация ответа (стриминг токенов).
        Генерирует строковые фрагменты по мере их появления в LLM.
        """
        if self.llm is None:
            raise RuntimeError("LLM-модель не загружена в память.")

        if cancel_event and cancel_event.is_set():
            raise LLMCancelledException("Генерация отменена до начала.")

        stream_gen = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )

        for chunk in stream_gen:
            if cancel_event and cancel_event.is_set():
                raise LLMCancelledException("Генерация остановлена пользователем.")

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
