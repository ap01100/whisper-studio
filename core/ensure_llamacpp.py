"""Utility to ensure llama-cpp-python binaries are compatible with the host CPU.
Resolves Windows Error 0xc000001d (STATUS_ILLEGAL_INSTRUCTION) on Intel 12th+ Gen (Alder Lake)
and other CPUs lacking AVX-512.
"""
import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path

LLAMACPP_RELEASE_TAG = "b10453"
LLAMACPP_CUDA_ZIP_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_RELEASE_TAG}/"
    f"llama-{LLAMACPP_RELEASE_TAG}-bin-win-cuda-12.4-x64.zip"
)


def is_avx512_supported() -> bool:
    """Проверяет поддержку AVX-512 процессором через Windows API."""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32")
        # PF_AVX512F_INSTRUCTIONS_AVAILABLE = 49
        return bool(kernel32.IsProcessorFeaturePresent(49))
    except Exception:
        return True


def ensure_compatible_llamacpp(force: bool = False) -> bool:
    """
    Проверяет совместимость установленных бинарников llama-cpp-python.
    Если процессор не поддерживает AVX-512, а установленные библиотеки
    содержат инструкции AVX-512 или отсутствует ggml-cpu-alderlake.dll,
    автоматически устанавливает официальные DLL с динамическим выбором архитектуры.
    """
    site_packages = [Path(p) for p in sys.path if "site-packages" in p.lower()]
    venv_dir = Path(sys.prefix) / "Lib" / "site-packages"
    if venv_dir.exists() and venv_dir not in site_packages:
        site_packages.append(venv_dir)

    target_lib_dir = None
    for sp in site_packages:
        candidate = sp / "llama_cpp" / "lib"
        if candidate.exists():
            target_lib_dir = candidate
            break

    if not target_lib_dir:
        return False

    alderlake_dll = target_lib_dir / "ggml-cpu-alderlake.dll"
    has_alderlake = alderlake_dll.exists()
    avx512_ok = is_avx512_supported()

    # Если AVX-512 есть или уже установлены DLL с поддержкой Alder Lake
    if not force and (avx512_ok or has_alderlake):
        return True

    print(f"[*] Обнаружен процессор без AVX-512 (Alder Lake/Raptor Lake). Установка совместимых DLL {LLAMACPP_RELEASE_TAG}...")
    temp_zip = target_lib_dir.parent / f"llama-{LLAMACPP_RELEASE_TAG}-cuda.zip"

    try:
        if not temp_zip.exists():
            print(f"[*] Загрузка совместимых DLL llama.cpp ({LLAMACPP_RELEASE_TAG})...")
            urllib.request.urlretrieve(LLAMACPP_CUDA_ZIP_URL, temp_zip)

        with zipfile.ZipFile(temp_zip) as z:
            for item in z.namelist():
                if item.endswith(".dll"):
                    dest = target_lib_dir / Path(item).name
                    with z.open(item) as src_f, open(dest, "wb") as dst_f:
                        dst_f.write(src_f.read())

        # Создаем копию ggml-cpu-alderlake как ggml-cpu.dll
        alderlake_src = target_lib_dir / "ggml-cpu-alderlake.dll"
        if alderlake_src.exists():
            shutil.copy2(alderlake_src, target_lib_dir / "ggml-cpu.dll")

        if temp_zip.exists():
            try:
                temp_zip.unlink()
            except Exception:
                pass

        print("[OK] Совместимые бинарники llama.cpp успешно установлены!")
        return True
    except Exception as e:
        print(f"[WARNING] Не удалось автоматически обновить DLL llama.cpp: {e}")
        return False


if __name__ == "__main__":
    ensure_compatible_llamacpp()
