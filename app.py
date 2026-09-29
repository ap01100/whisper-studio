"""Whisper Studio — Local Speech Recognition Application.
Entry point for launching the CustomTkinter desktop interface.
"""
import sys
import os
from pathlib import Path

# 0. Настройка кодировки терминала на Windows и отключение сбоящего Xet
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

# Добавляем корневую директорию проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 1. Системная авторизация и настройки Hugging Face Hub (токен и надёжная HTTP-загрузка)
from core.hf_downloader import apply_hf_environment
apply_hf_environment()

# 2. Критическая регистрация путей DLL библиотек NVIDIA CUDA 12
from core.cuda_utils import setup_cuda_dlls
cuda_dlls_registered = setup_cuda_dlls()

# 2. Инициализация главного окна GUI
from ui.main_window import MainWindow


def main():
    # Проверка версии Python
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10) or (major, minor) > (3, 13):
        print(f"[ВНИМАНИЕ] Текущая версия Python {major}.{minor} может иметь проблемы совместимости с CTranslate2.")
        print("Рекомендуемая версия: Python 3.10 - 3.13.")

    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
