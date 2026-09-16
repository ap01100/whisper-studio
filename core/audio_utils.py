import os
import sys
import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts"
}


def get_ffmpeg_path() -> Optional[str]:
    """Возвращает путь к статическому ffmpeg из imageio-ffmpeg или системному PATH."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    return "ffmpeg"


def format_seconds_to_time(seconds: float, include_ms: bool = False, ms_sep: str = ",") -> str:
    """
    Форматирует секунды в строку времени.
    include_ms=False: '04:12' или '01:23:45'
    include_ms=True: '00:04:12,345' (SRT) или '00:04:12.345' (VTT)
    """
    if seconds < 0:
        seconds = 0.0
    total_sec = int(seconds)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    ms = int(round((seconds - total_sec) * 1000))
    if ms >= 1000:
        ms = 999

    if include_ms:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{ms_sep}{ms:03d}"
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_media_info(file_path: str) -> Dict[str, Any]:
    """
    Возвращает информацию о медиафайле: размер, длительность, форматированную длительность.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    size_bytes = path.stat().st_size
    size_mb = round(size_bytes / (1024 * 1024), 2)
    duration_sec = 0.0

    # 1. Попытка через PyAV (быстро и точно)
    try:
        import av
        with av.open(str(path)) as container:
            if container.duration is not None:
                duration_sec = float(container.duration) / av.time_base
            else:
                for stream in container.streams:
                    if stream.type == "audio" and stream.duration and stream.time_base:
                        duration_sec = float(stream.duration * stream.time_base)
                        break
    except Exception:
        # 2. Резервная попытка через FFmpeg
        ffmpeg_exe = get_ffmpeg_path()
        try:
            cmd = [ffmpeg_exe, "-i", str(path)]
            res = subprocess.run(
                cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", res.stderr)
            if match:
                h, m, s = match.groups()
                duration_sec = int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            duration_sec = 0.0

    return {
        "path": str(path),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_mb": size_mb,
        "duration_sec": duration_sec,
        "duration_str": format_seconds_to_time(duration_sec),
        "is_supported": path.suffix.lower() in SUPPORTED_EXTENSIONS,
    }


def format_clean_paragraphs(segments: List[Dict[str, Any]]) -> str:
    """
    Форматирует список сегментов в естественные структурированные абзацы.
    Разделяет текст по паузам лектора (>1.8 сек) или логическим предложениям (3-5 предложений / ~400 символов).
    """
    if not segments:
        return ""

    paragraphs = []
    current_para = []
    current_chars = 0
    prev_end = None

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)

        # Проверяем естественную паузу в речи (> 1.8 сек)
        long_pause = (prev_end is not None and (start - prev_end) > 1.8)

        # Проверяем завершение предложения знаком препинания
        ends_sentence = text.endswith((".", "!", "?", "…", "..."))

        current_para.append(text)
        current_chars += len(text) + 1
        prev_end = end

        # Разделяем на абзацы при завершении мысли и достаточном объёме
        if (len(current_para) >= 3 and ends_sentence) or (current_chars >= 400 and ends_sentence) or (long_pause and len(current_para) >= 2):
            paragraphs.append(" ".join(current_para))
            current_para = []
            current_chars = 0

    if current_para:
        paragraphs.append(" ".join(current_para))

    return "\n\n".join(paragraphs)


def export_txt(segments: List[Dict[str, Any]], output_path: str, include_timestamps: bool = False) -> None:
    """Экспорт транскрипции в текстовый файл (.txt)."""
    with open(output_path, "w", encoding="utf-8") as f:
        if include_timestamps:
            for seg in segments:
                text = seg["text"].strip()
                if not text:
                    continue
                t_start = format_seconds_to_time(seg["start"])
                t_end = format_seconds_to_time(seg["end"])
                f.write(f"[{t_start} -> {t_end}] {text}\n")
        else:
            text = format_clean_paragraphs(segments)
            f.write(text + "\n")


def export_srt(segments: List[Dict[str, Any]], output_path: str) -> None:
    """Экспорт транскрипции в формат субтитров SubRip (.srt)."""
    with open(output_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            start_str = format_seconds_to_time(seg["start"], include_ms=True, ms_sep=",")
            end_str = format_seconds_to_time(seg["end"], include_ms=True, ms_sep=",")
            f.write(f"{idx}\n{start_str} --> {end_str}\n{text}\n\n")
            idx += 1


def export_vtt(segments: List[Dict[str, Any]], output_path: str) -> None:
    """Экспорт транскрипции в формат WebVTT (.vtt)."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        idx = 1
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            start_str = format_seconds_to_time(seg["start"], include_ms=True, ms_sep=".")
            end_str = format_seconds_to_time(seg["end"], include_ms=True, ms_sep=".")
            f.write(f"{idx}\n{start_str} --> {end_str}\n{text}\n\n")
            idx += 1


def export_json(segments: List[Dict[str, Any]], metadata: Dict[str, Any], output_path: str) -> None:
    """Экспорт структурированных данных транскрипции в JSON (.json)."""
    data = {
        "metadata": metadata,
        "segments": segments,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
