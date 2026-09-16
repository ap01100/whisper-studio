import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any


def setup_cuda_dlls() -> bool:
    """
    Регистрирует пути к DLL пакетов nvidia-cublas, nvidia-cudnn и nvrtc в Windows.
    Обязательно вызывается до любого импорта ctranslate2 или faster_whisper.
    """
    if sys.platform != "win32":
        return True

    site_packages = [Path(p) for p in sys.path if "site-packages" in p.lower()]
    found_any = False

    # Также проверяем типичные пути виртуального окружения
    venv_dir = Path(sys.prefix) / "Lib" / "site-packages"
    if venv_dir.exists() and venv_dir not in site_packages:
        site_packages.append(venv_dir)

    for sp in site_packages:
        nvidia_dir = sp / "nvidia"
        if nvidia_dir.exists():
            for subpkg in ["cublas", "cudnn", "cuda_nvrtc"]:
                bin_dir = nvidia_dir / subpkg / "bin"
                if bin_dir.exists():
                    bin_str = str(bin_dir)
                    try:
                        os.add_dll_directory(bin_str)
                    except (AttributeError, OSError):
                        pass
                    if bin_str not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = bin_str + os.pathsep + os.environ.get("PATH", "")
                    found_any = True

    return found_any


def get_gpu_info() -> Dict[str, Any]:
    """
    Получает информацию о видеокарте NVIDIA, драйвере и готовности CUDA.
    Возвращает словарь с параметрами для отображения в UI и настройки инференса.
    """
    info = {
        "has_gpu": False,
        "name": "CPU (Процессор)",
        "vram_gb": 0.0,
        "driver_version": "N/A",
        "cuda_ready": False,
        "badge_text": "● CPU Mode (GPU не обнаружен)",
        "status_color": "#F59E0B",  # Янтарный предупреждающий
        "warning": None,
    }

    # 1. Попытка запросить nvidia-smi
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if res.returncode == 0 and res.stdout.strip():
            first_line = res.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 3:
                info["has_gpu"] = True
                info["name"] = parts[0]
                try:
                    vram_mb = float(parts[1])
                    info["vram_gb"] = round(vram_mb / 1024.0, 1)
                except ValueError:
                    info["vram_gb"] = 0.0
                info["driver_version"] = parts[2]

                # Проверка для серии Blackwell (RTX 50xx)
                try:
                    major = int(parts[2].split(".")[0])
                    if ("RTX 50" in parts[0] or "5070" in parts[0] or "5080" in parts[0] or "5090" in parts[0]) and major < 570:
                        info["warning"] = f"Для RTX 50-й серии рекомендуется драйвер >= 570.xx (текущий {parts[2]})."
                except Exception:
                    pass
    except Exception:
        pass

    # 2. Проверка доступности CUDA в CTranslate2
    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
        if cuda_count > 0:
            info["cuda_ready"] = True
            info["status_color"] = "#10B981"  # Изумрудный зелёный
            info["badge_text"] = f"● GPU: {info['name']} | VRAM: {info['vram_gb']} GB | CUDA Ready"
        else:
            if info["has_gpu"]:
                info["status_color"] = "#F59E0B"
                info["badge_text"] = f"● GPU: {info['name']} (CUDA DLL не загружены -> CPU)"
            else:
                info["status_color"] = "#9CA3AF"
                info["badge_text"] = "● Режим CPU (Без ускорения GPU)"
    except Exception:
        if info["has_gpu"]:
            info["status_color"] = "#F59E0B"
            info["badge_text"] = f"● GPU: {info['name']} | {info['vram_gb']} GB"

    return info
