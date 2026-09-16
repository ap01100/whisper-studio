"""Whisper Studio — Local Speech Recognition Application.
Entry point for launching the CustomTkinter desktop interface.
"""
import sys
import os
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 1. Критическая регистрация путей DLL библиотек NVIDIA CUDA 12
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
