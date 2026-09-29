"""PyAV compatibility patch for faster-whisper.
PyAV 19.0.0 removed the `metadata_errors` and `metadata_encoding` arguments from av.open().
However, faster-whisper (versions up to 1.2.1+) still calls:
    av.open(input_file, mode="r", metadata_errors="ignore")
This module intercepts av.open() and strips unsupported arguments when needed.
"""
from typing import Any


def apply_pyav_patch() -> bool:
    """Патчит av.open для совместимости с PyAV >= 19.0.0 и faster-whisper."""
    try:
        import av
    except ImportError:
        return False

    _orig_av_open = av.open
    if getattr(_orig_av_open, "_is_compat_patched", False):
        return True

    def _safe_av_open(*args: Any, **kwargs: Any) -> Any:
        try:
            return _orig_av_open(*args, **kwargs)
        except TypeError as te:
            err_msg = str(te)
            if "metadata_errors" in err_msg or "metadata_encoding" in err_msg:
                kwargs.pop("metadata_errors", None)
                kwargs.pop("metadata_encoding", None)
                return _orig_av_open(*args, **kwargs)
            raise

    _safe_av_open._is_compat_patched = True
    av.open = _safe_av_open

    try:
        import av.container
        if hasattr(av.container, "open"):
            av.container.open = _safe_av_open
        if hasattr(av.container, "core") and hasattr(av.container.core, "open"):
            av.container.core.open = _safe_av_open
    except Exception:
        pass

    return True
