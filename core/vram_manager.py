"""VRAM Arbiter for Whisper Studio.
Manages GPU memory allocation and safely alternates between Whisper and LLM models
to prevent CUDA Out of Memory (OOM) errors on 8 GB GPUs like RTX 4060 Laptop.
"""
import os
import sys
import gc
import time
import subprocess
from typing import Dict, Any, Optional

from core.cuda_utils import get_gpu_info, setup_cuda_dlls


class VRAMManager:
    """Арбитр видеопамяти: отслеживает VRAM и обеспечивает взаимную выгрузку моделей."""

    _instance: Optional["VRAMManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VRAMManager, cls).__new__(cls)
            cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        self.transcriber_ref = None
        self.llm_engine_ref = None
        self._last_vram_query = 0.0
        self._cached_vram: Dict[str, Any] = {
            "used_mb": 0,
            "total_mb": 8192,
            "free_mb": 8192,
            "percent": 0.0,
            "status_text": "VRAM: N/A",
        }
        setup_cuda_dlls()

    def register_transcriber(self, transcriber):
        """Регистрирует экземпляр WhisperTranscriber."""
        self.transcriber_ref = transcriber

    def register_llm_engine(self, llm_engine):
        """Регистрирует экземпляр LLMEngine."""
        self.llm_engine_ref = llm_engine

    def get_vram_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Возвращает актуальную информацию об использовании VRAM.
        Кэшируется на 1.0 сек для исключения избыточных вызовов nvidia-smi.
        """
        now = time.time()
        if not force_refresh and (now - self._last_vram_query < 1.0):
            return self._cached_vram

        self._last_vram_query = now

        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ]
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=2,
                creationflags=flags,
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 3:
                    used = int(parts[0])
                    total = int(parts[1])
                    free = int(parts[2])
                    pct = round((used / total) * 100, 1) if total > 0 else 0.0

                    self._cached_vram = {
                        "used_mb": used,
                        "total_mb": total,
                        "free_mb": free,
                        "percent": pct,
                        "status_text": f"VRAM: {used / 1024:.1f} / {total / 1024:.1f} GB ({pct}%)",
                    }
                    return self._cached_vram
        except Exception:
            pass

        # Резервный ответ
        gpu_info = get_gpu_info()
        total_gb = gpu_info.get("vram_gb", 8.0)
        self._cached_vram = {
            "used_mb": 0,
            "total_mb": int(total_gb * 1024),
            "free_mb": int(total_gb * 1024),
            "percent": 0.0,
            "status_text": f"VRAM: ~{total_gb:.1f} GB",
        }
        return self._cached_vram

    def prepare_for_transcription(self) -> None:
        """
        Освобождает VRAM перед запуском распознавания Whisper.
        Если в памяти находится LLM (Qwen / DeepSeek), она полностью выгружается.
        """
        if self.llm_engine_ref is not None:
            if hasattr(self.llm_engine_ref, "is_loaded") and self.llm_engine_ref.is_loaded():
                self.llm_engine_ref.unload()

        # Принудительная сборка мусора Python
        gc.collect()
        time.sleep(0.05)

    def prepare_for_llm(self) -> None:
        """
        Освобождает VRAM перед запуском генерации конспекта LLM.
        Если в памяти находится модель Whisper, она выгружается из GPU.
        """
        if self.transcriber_ref is not None:
            if hasattr(self.transcriber_ref, "unload_model"):
                self.transcriber_ref.unload_model()

        # Принудительная сборка мусора Python
        gc.collect()
        time.sleep(0.05)

    def unload_all(self) -> None:
        """Выгружает обе модели и очищает видеопамять."""
        if self.transcriber_ref is not None and hasattr(self.transcriber_ref, "unload_model"):
            self.transcriber_ref.unload_model()
        if self.llm_engine_ref is not None and hasattr(self.llm_engine_ref, "unload"):
            self.llm_engine_ref.unload()
        gc.collect()


def get_vram_manager() -> VRAMManager:
    """Возвращает глобальный синглтон VRAMManager."""
    return VRAMManager()
