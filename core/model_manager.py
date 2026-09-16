import os
import sys
import time
import shutil
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable

MODELS_DIR = Path("./models").resolve()

# Информация о поддерживаемых моделях
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "large-v3-turbo": {
        "label": "large-v3-turbo (Рекомендуется для RTX 4060/5070, ~1.6 GB)",
        "short_label": "large-v3-turbo",
        "repo_id": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "size_mb": 1620,
        "description": "Идеальный баланс точности large-v3 и в 8 раз более высокой скорости",
        "recommended": True,
        "vram_req_gb": 4.0,
    },
    "large-v3": {
        "label": "large-v3 (Максимальная точность, ~3.1 GB)",
        "short_label": "large-v3",
        "repo_id": "Systran/faster-whisper-large-v3",
        "size_mb": 3100,
        "description": "Максимальное качество распознавания для сложного шумного аудио",
        "recommended": False,
        "vram_req_gb": 6.0,
    },
    "medium": {
        "label": "medium (Высокая точность, ~1.5 GB)",
        "short_label": "medium",
        "repo_id": "Systran/faster-whisper-medium",
        "size_mb": 1530,
        "description": "Универсальная точная модель для любых задач",
        "recommended": False,
        "vram_req_gb": 3.0,
    },
    "small": {
        "label": "small (Быстрая и легкая, ~480 MB)",
        "short_label": "small",
        "repo_id": "Systran/faster-whisper-small",
        "size_mb": 485,
        "description": "Быстрая модель со сбалансированным качеством",
        "recommended": False,
        "vram_req_gb": 2.0,
    },
    "base": {
        "label": "base (Мгновенный тест, ~145 MB)",
        "short_label": "base",
        "repo_id": "Systran/faster-whisper-base",
        "size_mb": 145,
        "description": "Быстрый старт для проверки работоспособности за секунды",
        "recommended": False,
        "vram_req_gb": 1.0,
    },
    "tiny": {
        "label": "tiny (Сверхлегкая, ~75 MB)",
        "short_label": "tiny",
        "repo_id": "Systran/faster-whisper-tiny",
        "size_mb": 75,
        "description": "Минимальный размер и моментальное скачивание",
        "recommended": False,
        "vram_req_gb": 1.0,
    },
}


def get_model_target_dir(model_name: str) -> Path:
    """Возвращает директорию локальной модели в ./models/"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / model_name


def _check_dir_has_weights(directory: Path) -> bool:
    """Проверяет, содержит ли директория необходимые файлы весов CTranslate2."""
    if not directory.exists() or not directory.is_dir():
        return False
    
    # Необходимые файлы для CTranslate2 Whisper
    has_model = (directory / "model.bin").exists() or (directory / "model.safetensors").exists()
    has_config = (directory / "config.json").exists()
    
    if has_model and has_config:
        return True

    # Проверка подпапки snapshots (стандартный кэш HuggingFace)
    snapshots_dir = directory / "snapshots"
    if snapshots_dir.exists():
        for snap in snapshots_dir.iterdir():
            if snap.is_dir() and ((snap / "model.bin").exists() or (snap / "model.safetensors").exists()):
                return True

    return False


def is_model_cached(model_name: str) -> bool:
    """Проверяет, скачана ли модель локально."""
    # 1. Проверяем ./models/{model_name}
    target_dir = get_model_target_dir(model_name)
    if _check_dir_has_weights(target_dir):
        return True

    # 2. Проверяем ./models/models--*
    info = MODEL_REGISTRY.get(model_name)
    if info:
        repo_id = info["repo_id"]
        hf_dirname = "models--" + repo_id.replace("/", "--")
        if _check_dir_has_weights(MODELS_DIR / hf_dirname):
            return True

    # 3. Проверяем системный кэш HuggingFace (~/.cache/huggingface/hub)
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
    if info:
        repo_id = info["repo_id"]
        hf_dirname = "models--" + repo_id.replace("/", "--")
        if _check_dir_has_weights(hf_home / hf_dirname):
            return True

    return False


def get_resolved_model_path(model_name: str) -> str:
    """
    Возвращает путь к модели: если она скачана локально — возвращает путь к папке,
    иначе возвращает имя модели или repo_id для загрузки faster-whisper.
    """
    target_dir = get_model_target_dir(model_name)
    if _check_dir_has_weights(target_dir):
        return str(target_dir)

    info = MODEL_REGISTRY.get(model_name)
    if info:
        repo_id = info["repo_id"]
        hf_dirname = "models--" + repo_id.replace("/", "--")
        local_hf = MODELS_DIR / hf_dirname
        if _check_dir_has_weights(local_hf):
            # Проверяем snapshot
            snapshots = local_hf / "snapshots"
            if snapshots.exists():
                for snap in snapshots.iterdir():
                    if snap.is_dir() and _check_dir_has_weights(snap):
                        return str(snap)
            return str(local_hf)

        # Если это кастомный repo_id для turbo
        if model_name == "large-v3-turbo":
            return repo_id

    return model_name


def _calculate_dir_size_bytes(directory: Path) -> int:
    """Подсчитывает суммарный размер файлов в директории."""
    total = 0
    if not directory.exists():
        return 0
    try:
        for p in directory.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except Exception:
        pass
    return total


def download_model(
    model_name: str,
    progress_callback: Optional[Callable[[float, float, float, float, str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> str:
    """
    Скачивает веса модели в ./models/{model_name} с отображением прогресса.
    progress_callback(fraction, current_mb, total_mb, speed_mb_s, status_text)
    """
    info = MODEL_REGISTRY.get(model_name)
    if not info:
        raise ValueError(f"Неизвестная модель: {model_name}")

    target_dir = get_model_target_dir(model_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    expected_mb = float(info["size_mb"])
    expected_bytes = expected_mb * 1024 * 1024
    repo_id = info["repo_id"]

    download_error = []
    download_done = threading.Event()

    def _worker():
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                max_workers=4,
            )
        except Exception as e:
            download_error.append(e)
        finally:
            download_done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    start_time = time.time()
    thread.start()

    last_bytes = 0
    last_time = start_time

    while not download_done.is_set():
        if stop_event and stop_event.is_set():
            break

        time.sleep(0.15)
        current_bytes = _calculate_dir_size_bytes(target_dir)
        now = time.time()
        elapsed = now - last_time

        if elapsed >= 0.5:
            delta_bytes = current_bytes - last_bytes
            speed_mb_s = (delta_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
            last_bytes = current_bytes
            last_time = now
        else:
            speed_mb_s = (current_bytes / (1024 * 1024)) / (now - start_time) if (now - start_time) > 0 else 0.0

        current_mb = current_bytes / (1024 * 1024)
        fraction = min(0.99, current_bytes / expected_bytes) if expected_bytes > 0 else 0.0

        if progress_callback:
            status_text = f"Загрузка весов {model_name}: {current_mb:.1f} / {expected_mb:.0f} MB ({speed_mb_s:.1f} MB/s)"
            progress_callback(fraction, current_mb, expected_mb, speed_mb_s, status_text)

    thread.join(timeout=2.0)

    if download_error:
        raise download_error[0]

    # Финальный колбэк 100%
    if progress_callback:
        final_mb = _calculate_dir_size_bytes(target_dir) / (1024 * 1024)
        progress_callback(1.0, final_mb, final_mb, 0.0, f"Модель {model_name} успешно загружена и готова!")

    return str(target_dir)


def any_model_cached() -> bool:
    """Возвращает True, если хотя бы одна модель уже скачана."""
    for name in MODEL_REGISTRY:
        if is_model_cached(name):
            return True
    return False
