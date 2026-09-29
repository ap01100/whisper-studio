"""Hugging Face Hub accelerator & GGUF model downloader for Whisper Studio.
Manages HF authentication token, Rust-based hf_transfer multi-threaded downloads,
and provides a local GGUF model registry for Qwen 2.5 and DeepSeek LLMs.
"""
import os
import sys

# 0. Настройка кодировки терминала на Windows и отключение сбоящего Xet до импорта huggingface_hub
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import json
import time
import threading
import io
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Callable

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

import huggingface_hub.constants as _hf_constants
_hf_constants.HF_HUB_DISABLE_XET = True
_hf_constants.HF_HUB_DISABLE_SYMLINKS_WARNING = True

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils.tqdm import tqdm as HfTqdm

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
        "hf_endpoint": "https://hf-mirror.com",
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
    """Применяет параметры Hugging Face к текущему процессу и выполняет системную авторизацию."""
    # 1. Настройка кодировки терминала на Windows для корректного вывода логов
    if sys.platform == "win32":
        try:
            if sys.stdout and hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if sys.stderr and hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # 2. Отключаем предупреждения о симлинках на Windows
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # 3. Отключаем Xet (на Windows часто зависает соединение с cas-server.xethub.hf.co)
    # и переключаемся на быстрый, надёжный HTTP-стриминг через Hugging Face CDN с докачкой
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

    # 4. Настройка сетевых таймаутов и быстрого зеркала для стабильной загрузки в РФ
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    cfg = load_config()
    endpoint = cfg.get("hf_endpoint", "https://hf-mirror.com")
    if endpoint and str(endpoint).strip():
        os.environ["HF_ENDPOINT"] = str(endpoint).strip()
    elif "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # 5. Авторизация токеном в окружении и кэше Hugging Face Hub
    token = cfg.get("hf_token")
    if token and str(token).strip():
        clean_token = str(token).strip()
        try:
            import huggingface_hub
            if huggingface_hub.get_token() != clean_token:
                huggingface_hub.login(token=clean_token, add_to_git_credential=False)
        except Exception:
            pass
        os.environ["HF_TOKEN"] = clean_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = clean_token
    else:
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        try:
            import huggingface_hub
            huggingface_hub.logout()
        except Exception:
            pass


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
    Скачивает выбранную GGUF-модель с Hugging Face Hub с поддержкой токена и прямого HTTP-стриминга.
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

    # Применяем токен и настройки сети
    apply_hf_environment()
    token = get_hf_token()

    print(f"[*] Подключение к Hugging Face Hub: {repo_id}/{filename} (~{expected_mb} MB)...", flush=True)
    if progress_callback:
        progress_callback(0.0, 0.0, expected_mb, 0.0, f"Подключение к Hugging Face Hub ({filename})...")

    start_time = time.time()

    class DownloadProgressTqdm(HfTqdm):
        """Потоковый класс отслеживания прогресса и скорости загрузки для hf_hub_download."""
        def __init__(self, *args, **kwargs):
            # Принудительно отключаем подавление tqdm в HF Hub
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)
            self.disable = False

            self._last_time = time.time()
            self._last_n = 0
            self.current_bytes = 0
            self.last_bytes = 0
            self.total_bytes = getattr(self, "total", None) or int(expected_mb * 1024 * 1024)

        def update(self, n=1):
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Загрузка отменена пользователем.")

            try:
                super().update(n)
            except Exception:
                pass

            self.current_bytes = self.n
            now = time.time()
            dt = now - self._last_time
            tot = self.total or self.total_bytes or int(expected_mb * 1024 * 1024)
            if dt >= 0.25 or (tot and self.n >= tot):
                delta = self.n - self._last_n
                speed_mb_s = (delta / (1024 * 1024)) / dt if dt > 0 else 0.0
                self._last_time = now
                self._last_n = self.n
                self.last_bytes = self.n

                cur_mb = self.n / (1024 * 1024)
                tot_mb = tot / (1024 * 1024)
                frac = min(0.999, self.n / tot) if tot > 0 else 0.0
                st_text = f"Загрузка: {cur_mb:.1f} / {tot_mb:.0f} MB ({speed_mb_s:.1f} MB/s)"
                if progress_callback:
                    progress_callback(frac, cur_mb, tot_mb, speed_mb_s, st_text)

        def update_transfer(self, inc=1):
            # Внимание: http_get уже вызывает progress.update(len(chunk)).
            # Вызов update() здесь приводил бы к двойному учёту скачанных байт.
            pass

        def set_postfix_str(self, s="", refresh=True):
            pass

        def set_transfer_postfix_str(self, s="", refresh=True):
            pass

        def close(self):
            try:
                super().close()
            except Exception:
                pass

    try:
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(LLM_MODELS_DIR),
            token=token,
            tqdm_class=DownloadProgressTqdm,
        )

        total_elapsed = time.time() - start_time
        avg_speed = (expected_mb / total_elapsed) if total_elapsed > 0 else 0.0
        print(f"[OK] Модель {filename} успешно сохранена в {downloaded_file} (средняя скорость: {avg_speed:.1f} MB/s)", flush=True)
        if progress_callback:
            progress_callback(1.0, expected_mb, expected_mb, avg_speed, "Загрузка модели завершена!")

        return Path(downloaded_file)

    except Exception as e:
        if cancel_event and cancel_event.is_set():
            print(f"[!] Загрузка {filename} прервана пользователем.", flush=True)
            raise RuntimeError("Загрузка отменена пользователем.") from e
        print(f"[ERROR] Ошибка скачивания {filename}: {e}", flush=True)
        raise RuntimeError(f"Ошибка скачивания модели с Hugging Face: {e}") from e


# Автоматическое применение настроек окружения Hugging Face при загрузке модуля
apply_hf_environment()
