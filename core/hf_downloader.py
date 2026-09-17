"""Hugging Face Hub accelerator & GGUF model downloader for Whisper Studio.
Manages HF authentication token, Rust-based hf_transfer multi-threaded downloads,
and provides a local GGUF model registry for Qwen 2.5 and DeepSeek LLMs.
"""
import os
import sys
import json
import time
import threading
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Callable

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

from huggingface_hub import HfApi, hf_hub_download

CONFIG_FILE = Path("./config.json").resolve()
LLM_MODELS_DIR = Path("./models/llm").resolve()

# Каталог поддерживаемых локальных GGUF-моделей
LLM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen-2.5-7b": {
        "id": "qwen-2.5-7b",
        "name": "Qwen 2.5 7B Instruct (Q4_K_M)",
        "short_name": "Qwen 2.5 7B",
        "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size_mb": 4680,
        "vram_req_gb": 5.5,
        "description": "Лучшее качество для русского языка и академических конспектов. Идеально оптимизирована под RTX 4060 (8 GB).",
        "recommended": True,
    },
    "qwen-2.5-3b": {
        "id": "qwen-2.5-3b",
        "name": "Qwen 2.5 3B Instruct (Q4_K_M)",
        "short_name": "Qwen 2.5 3B",
        "repo_id": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "size_mb": 2170,
        "vram_req_gb": 3.0,
        "description": "Молниеносная скорость генерации (до 50+ токенов/сек). Потребляет всего ~3 GB VRAM.",
        "recommended": False,
    },
    "deepseek-r1-7b": {
        "id": "deepseek-r1-7b",
        "name": "DeepSeek R1 Distill Qwen 7B (Q4_K_M)",
        "short_name": "DeepSeek R1 7B",
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "size_mb": 4680,
        "vram_req_gb": 5.5,
        "description": "Глубокий логический анализ, математические формулы и разбор сложных технических лекций.",
        "recommended": False,
    },
}


# ==============================================================================
# Управление конфигурацией (config.json)
# ==============================================================================

def load_config() -> Dict[str, Any]:
    """Загружает настройки из config.json."""
    default_cfg = {
        "hf_token": None,
        "hf_transfer": True,
        "default_llm": "qwen-2.5-7b",
        "autosave_summary": True,
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                default_cfg.update(data)
        except Exception:
            pass
    return default_cfg


def save_config(config_data: Dict[str, Any]) -> None:
    """Сохраняет настройки в config.json."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save config.json: {e}")


def get_hf_token() -> Optional[str]:
    """Возвращает сохранённый токен Hugging Face или токен из переменной окружения."""
    cfg = load_config()
    token = cfg.get("hf_token")
    if token and str(token).strip():
        return str(token).strip()
    return os.environ.get("HF_TOKEN", None)


def set_hf_token(token: Optional[str]) -> None:
    """Обновляет и сохраняет токен Hugging Face."""
    cfg = load_config()
    cleaned = token.strip() if token else None
    cfg["hf_token"] = cleaned
    save_config(cfg)
    apply_hf_environment()


def apply_hf_environment() -> None:
    """Применяет параметры Hugging Face к текущему процессу."""
    cfg = load_config()
    token = cfg.get("hf_token")
    if token:
        os.environ["HF_TOKEN"] = str(token)
    elif "HF_TOKEN" in os.environ and not token:
        del os.environ["HF_TOKEN"]

    # Включение высокопроизводительной передачи данных (Xet / High Performance)
    if cfg.get("hf_transfer", True):
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    else:
        os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)


# ==============================================================================
# Проверка и валидация токена Hugging Face
# ==============================================================================

def verify_hf_token(token: str) -> Dict[str, Any]:
    """
    Проверяет валидность токена Hugging Face через официальный API.
    Возвращает словарь: {"valid": bool, "username": str, "auth_type": str, "error": str}
    """
    if not token or not token.strip():
        return {"valid": False, "username": "", "auth_type": "", "error": "Токен не может быть пустым."}

    cleaned = token.strip()
    try:
        api = HfApi()
        info = api.whoami(token=cleaned)
        username = info.get("name", "Пользователь HF")
        auth_type = info.get("auth", {}).get("type", "read")
        return {
            "valid": True,
            "username": username,
            "auth_type": auth_type,
            "error": None,
        }
    except Exception as e:
        err = str(e)
        if "401" in err:
            msg = "Неверный или просроченный токен (401 Unauthorized)."
        else:
            msg = f"Ошибка связи с Hugging Face: {err[:80]}"
        return {"valid": False, "username": "", "auth_type": "", "error": msg}


# ==============================================================================
# Работа с каталогом моделей GGUF
# ==============================================================================

def get_llm_model_path(model_id: str) -> Path:
    """Возвращает ожидаемый локальный путь к файлу GGUF-модели."""
    info = LLM_REGISTRY.get(model_id, LLM_REGISTRY["qwen-2.5-7b"])
    filename = info["filename"]
    return LLM_MODELS_DIR / filename


def is_llm_cached(model_id: str) -> bool:
    """Проверяет, скачан ли GGUF-файл указанной модели."""
    p = get_llm_model_path(model_id)
    if not p.exists():
        return False
    # Файл должен быть не пустым (хотя бы > 500 МБ)
    return p.stat().st_size > (500 * 1024 * 1024)


def download_llm_model(
    model_id: str,
    progress_callback: Optional[Callable[[float, float, float, float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Path:
    """
    Скачивает выбранную GGUF-модель с Hugging Face Hub с поддержкой токена и hf_transfer.
    progress_callback(fraction, current_mb, total_mb, speed_mb_s, status_text)
    """
    if model_id not in LLM_REGISTRY:
        raise ValueError(f"Неизвестная модель: {model_id}")

    info = LLM_REGISTRY[model_id]
    repo_id = info["repo_id"]
    filename = info["filename"]
    expected_mb = info["size_mb"]

    LLM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = LLM_MODELS_DIR / filename

    if is_llm_cached(model_id):
        if progress_callback:
            progress_callback(1.0, expected_mb, expected_mb, 0.0, "Модель уже скачана.")
        return target_path

    # Применяем токен и hf_transfer
    apply_hf_environment()
    token = get_hf_token()

    if progress_callback:
        progress_callback(0.0, 0.0, expected_mb, 0.0, f"Инициализация скачивания {filename}...")

    start_time = time.time()
    last_check_time = start_time
    last_size = 0

    # Фоновый мониторинг размера файла для плавной индикации прогресса и скорости
    stop_monitor = threading.Event()

    def size_monitor():
        nonlocal last_check_time, last_size
        while not stop_monitor.is_set():
            time.sleep(0.5)
            if cancel_event and cancel_event.is_set():
                break

            current_size = 0
            if target_path.exists():
                current_size = target_path.stat().st_size
            else:
                # Проверяем возможные временные файлы .incomplete / .download
                for f in LLM_MODELS_DIR.glob(f"*{filename}*"):
                    current_size = max(current_size, f.stat().st_size)

            current_mb = current_size / (1024 * 1024)
            now = time.time()
            dt = now - last_check_time
            if dt >= 0.5:
                speed_mb_s = max(0.0, (current_size - last_size) / (1024 * 1024) / dt)
                last_check_time = now
                last_size = current_size
            else:
                speed_mb_s = 0.0

            frac = min(0.99, current_mb / expected_mb) if expected_mb > 0 else 0.0
            st_text = f"Загрузка: {current_mb:.1f} / {expected_mb:.0f} MB ({speed_mb_s:.1f} MB/s)"
            if progress_callback:
                progress_callback(frac, current_mb, expected_mb, speed_mb_s, st_text)

    monitor_thread = threading.Thread(target=size_monitor, daemon=True)
    monitor_thread.start()

    try:
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(LLM_MODELS_DIR),
            token=token,
        )
        stop_monitor.set()
        monitor_thread.join(timeout=1.0)

        total_elapsed = time.time() - start_time
        avg_speed = (expected_mb / total_elapsed) if total_elapsed > 0 else 0.0
        if progress_callback:
            progress_callback(1.0, expected_mb, expected_mb, avg_speed, "Загрузка модели завершена!")

        return Path(downloaded_file)

    except Exception as e:
        stop_monitor.set()
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Загрузка отменена пользователем.") from e
        raise RuntimeError(f"Ошибка скачивания модели с Hugging Face: {e}") from e
