import os
import sys
import gc
import time
import threading
from typing import Dict, Any, List, Optional, Callable

from core.cuda_utils import setup_cuda_dlls, get_gpu_info
from core.audio_utils import get_media_info, format_seconds_to_time
from core.model_manager import get_resolved_model_path, MODELS_DIR

# Регистрация путей к DLL до импорта faster_whisper
setup_cuda_dlls()


class TranscriptionCancelledException(Exception):
    """Исключение при прерывании процесса пользователем."""
    pass


class WhisperTranscriber:
    """
    Класс-обёртка для управления моделью faster-whisper и потоковой расшифровки.
    """

    def __init__(self):
        self.model = None
        self.loaded_model_name: Optional[str] = None
        self.loaded_device: Optional[str] = None
        self.loaded_compute_type: Optional[str] = None
        self._lock = threading.Lock()

    def unload_model(self):
        """Выгружает модель из VRAM/RAM и выполняет сборку мусора."""
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
            self.loaded_model_name = None
            self.loaded_device = None
            self.loaded_compute_type = None
            gc.collect()

    def load_model(
        self,
        model_name: str,
        compute_type: str = "float16",
        force_reload: bool = False,
    ):
        """
        Загружает модель Whisper в память или возвращает уже загруженную.
        Автоматически определяет доступность CUDA.
        """
        with self._lock:
            gpu_info = get_gpu_info()
            device = "cuda" if gpu_info.get("cuda_ready", False) else "cpu"

            # Для CPU float16 обычно не поддерживается, переключаем на int8 или float32
            if device == "cpu" and compute_type in ("float16", "int8_float16"):
                compute_type = "int8"

            if (
                not force_reload
                and self.model is not None
                and self.loaded_model_name == model_name
                and self.loaded_device == device
                and self.loaded_compute_type == compute_type
            ):
                return self.model

            # Выгружаем предыдущую модель
            if self.model is not None:
                del self.model
                self.model = None
                gc.collect()

            model_path = get_resolved_model_path(model_name)

            try:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute_type,
                    download_root=str(MODELS_DIR),
                    cpu_threads=4,
                    num_workers=1,
                )
                self.loaded_model_name = model_name
                self.loaded_device = device
                self.loaded_compute_type = compute_type
                return self.model

            except Exception as e:
                err_msg = str(e).lower()
                self.model = None
                gc.collect()

                if "out of memory" in err_msg or "cuda oom" in err_msg:
                    raise RuntimeError(
                        "Недостаточно видеопамяти GPU (CUDA Out of Memory)!\n"
                        "Совет: переключите тип вычислений на 'int8_float16' в расширенных настройках "
                        "или выберите модель меньшего размера (medium / small)."
                    ) from e
                
                # Если упала загрузка CUDA из-за dll или драйвера, пробуем fallback на CPU
                if device == "cuda" and ("cublas" in err_msg or "cudnn" in err_msg or "cuda" in err_msg):
                    try:
                        from faster_whisper import WhisperModel
                        self.model = WhisperModel(
                            model_path,
                            device="cpu",
                            compute_type="int8",
                            download_root=str(MODELS_DIR),
                            cpu_threads=4,
                        )
                        self.loaded_model_name = model_name
                        self.loaded_device = "cpu"
                        self.loaded_compute_type = "int8"
                        return self.model
                    except Exception as cpu_e:
                        raise RuntimeError(f"Ошибка загрузки модели на GPU и CPU fallback: {cpu_e}") from cpu_e

                raise e

    def transcribe_file(
        self,
        audio_path: str,
        model_name: str = "large-v3-turbo",
        compute_type: str = "float16",
        language: Optional[str] = None,
        vad_filter: bool = True,
        beam_size: int = 5,
        word_timestamps: bool = False,
        condition_on_previous_text: bool = False,
        progress_callback: Optional[Callable[[float, Dict[str, Any], float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """
        Транскрибирует аудио/видеофайл с потоковым информированием о прогрессе.
        condition_on_previous_text=False предотвращает зацикливания и деградацию на файлах длиннее 20 минут.
        """
        media_info = get_media_info(audio_path)
        total_duration = media_info["duration_sec"]

        if cancel_event and cancel_event.is_set():
            raise TranscriptionCancelledException("Транскрибирование отменено пользователем.")

        # Загрузка/проверка модели
        self.load_model(model_name=model_name, compute_type=compute_type)

        lang_param = None if (not language or language == "auto" or language == "Авто") else language

        start_time = time.time()
        all_segments: List[Dict[str, Any]] = []

        try:
            vad_opts = dict(min_silence_duration_ms=500) if vad_filter else None
            segments_generator, info = self.model.transcribe(
                audio_path,
                language=lang_param,
                vad_filter=vad_filter,
                vad_parameters=vad_opts,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
                condition_on_previous_text=condition_on_previous_text,
                temperature=[0.0, 0.2, 0.4],
                repetition_penalty=1.05,
            )

            actual_duration = info.duration if info.duration > 0 else total_duration

            for segment in segments_generator:
                if cancel_event and cancel_event.is_set():
                    raise TranscriptionCancelledException("Транскрибирование остановлено пользователем.")

                seg_data = {
                    "id": segment.id,
                    "seek": segment.seek,
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                    "avg_logprob": round(segment.avg_logprob, 3),
                    "no_speech_prob": round(segment.no_speech_prob, 3),
                }
                all_segments.append(seg_data)

                elapsed = time.time() - start_time
                current_time = segment.end
                speed = (current_time / elapsed) if elapsed > 0 else 0.0
                fraction = min(1.0, current_time / actual_duration) if actual_duration > 0 else 0.0

                cur_str = format_seconds_to_time(current_time)
                tot_str = format_seconds_to_time(actual_duration)
                percent = int(fraction * 100)
                status_str = f"Обработка: {cur_str} / {tot_str} ({percent}%)"
                if progress_callback:
                    progress_callback(fraction, seg_data, speed, status_str)

                # Освобождаем GIL на 1мс, чтобы главный поток GUI работал на полных 60+ FPS
                time.sleep(0.001)

        except TranscriptionCancelledException:
            raise
        except Exception as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "cuda oom" in err_msg:
                raise RuntimeError(
                    "Недостаточно видеопамяти GPU во время инференса!\n"
                    "Рекомендация: переключитесь на 'int8_float16' в расширенных настройках."
                ) from e
            raise e

        total_elapsed = time.time() - start_time
        final_speed = (actual_duration / total_elapsed) if total_elapsed > 0 else 0.0

        return {
            "media_info": media_info,
            "detected_language": info.language if info else "ru",
            "language_probability": round(info.language_probability, 3) if info else 1.0,
            "duration": actual_duration,
            "elapsed_seconds": round(total_elapsed, 1),
            "speed_multiplier": round(final_speed, 1),
            "model_used": model_name,
            "compute_type_used": compute_type,
            "device_used": self.loaded_device,
            "segments": all_segments,
        }
