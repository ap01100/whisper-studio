"""Main application window for Whisper Studio.
"""
import os
import sys
import time
import queue
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

import customtkinter as ctk
from tkinter import filedialog, messagebox

from ui.theme import (
    COLOR_BG_PRIMARY,
    COLOR_BG_SECONDARY,
    COLOR_BG_CARD,
    COLOR_BG_CARD_HOVER,
    COLOR_BG_INPUT,
    COLOR_BORDER,
    COLOR_BORDER_FOCUS,
    COLOR_ACCENT_WHITE,
    COLOR_ACCENT_HOVER,
    COLOR_ACCENT_TEXT,
    COLOR_BTN_SECONDARY,
    COLOR_BTN_SECONDARY_HOVER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_DANGER,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED,
    FONT_TITLE,
    FONT_SUBTITLE,
    FONT_BODY,
    FONT_BODY_BOLD,
    FONT_CAPTION,
    FONT_CODE,
    CORNER_RADIUS_SM,
    CORNER_RADIUS_MD,
    CORNER_RADIUS_LG,
    apply_global_theme,
)
from ui.components import GPUBadge, DropZone, CollapsibleSection, ToastMessage
from ui.onboarding import OnboardingDialog
from core.cuda_utils import get_gpu_info
from core.model_manager import (
    MODEL_REGISTRY,
    is_model_cached,
    any_model_cached,
    download_model,
)
from core.transcriber import WhisperTranscriber, TranscriptionCancelledException
from core.audio_utils import (
    get_media_info,
    format_seconds_to_time,
    format_clean_paragraphs,
    export_txt,
    export_srt,
    export_vtt,
    export_json,
)


class MainWindow(ctk.CTk):
    """Главное окно приложения Whisper Studio."""

    def __init__(self):
        super().__init__()
        apply_global_theme()

        self.title("Whisper Studio — Local Speech Recognition")
        self.geometry("1180x760")
        self.minsize(980, 640)
        self.configure(fg_color=COLOR_BG_PRIMARY)

        # Состояние приложения
        self.transcriber = WhisperTranscriber()
        self.gpu_info = get_gpu_info()
        self.selected_file: Optional[str] = None
        self.last_saved_path: Optional[str] = None
        self.current_results: Optional[Dict[str, Any]] = None
        self.current_segments: List[Dict[str, Any]] = []
        self.is_transcribing = False
        self.cancel_event = threading.Event()
        self.transcribe_queue = queue.Queue()

        # Состояние потоковой разбивки на абзацы (для 200+ FPS)
        self._stream_para_count = 0
        self._stream_char_count = 0
        self._stream_last_end = 0.0

        # Построение интерфейса
        self._build_header()
        self._build_main_layout()

        # Toast для уведомлений
        self.toast = ToastMessage(self)

        # Проверка первого запуска (онбординг моделей)
        self.after(300, self._check_first_run)

    def _build_header(self):
        """Верхняя панель: название приложения и аппаратный бейдж."""
        header = ctk.CTkFrame(self, fg_color="transparent", height=56)
        header.pack(fill="x", padx=24, pady=(16, 12))

        # Левая часть шапки: Логотип и имя
        left_header = ctk.CTkFrame(header, fg_color="transparent")
        left_header.pack(side="left")

        app_title = ctk.CTkLabel(
            left_header,
            text="Whisper Studio",
            font=FONT_TITLE,
            text_color=COLOR_TEXT_PRIMARY,
        )
        app_title.pack(side="left")

        version_label = ctk.CTkLabel(
            left_header,
            text="v1.0 (CUDA 12)",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_MUTED,
        )
        version_label.pack(side="left", padx=(8, 0), pady=(3, 0))

        # Правая часть шапки: GPU Badge
        self.gpu_badge = GPUBadge(header, self.gpu_info)
        self.gpu_badge.pack(side="right")

    def _build_main_layout(self):
        """Основной макет из двух колонок: слева настройки и файл, справа результат."""
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        # ==========================================================
        # Левая колонка (Управление, ввод файла, параметры) ~ 400px
        # ==========================================================
        left_col = ctk.CTkScrollableFrame(
            container,
            width=410,
            fg_color="transparent",
            scrollbar_button_color=COLOR_BG_CARD,
            scrollbar_button_hover_color=COLOR_BORDER,
        )
        left_col.pack(side="left", fill="y", padx=(0, 16))

        # 1. Drag & Drop зона
        self.drop_zone = DropZone(left_col, on_file_selected=self._on_file_selected)
        self.drop_zone.pack(fill="x", pady=(0, 14))

        # 2. Карточка выбора модели
        model_card = ctk.CTkFrame(
            left_col,
            fg_color=COLOR_BG_CARD,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=CORNER_RADIUS_MD,
        )
        model_card.pack(fill="x", pady=(0, 12))

        model_title_row = ctk.CTkFrame(model_card, fg_color="transparent")
        model_title_row.pack(fill="x", padx=14, pady=(12, 6))

        ctk.CTkLabel(
            model_title_row,
            text="Модель распознавания:",
            font=FONT_SUBTITLE,
            text_color=COLOR_TEXT_PRIMARY,
        ).pack(side="left")

        self.model_status_tag = ctk.CTkLabel(
            model_title_row,
            text="",
            font=FONT_CAPTION,
            text_color=COLOR_SUCCESS,
        )
        self.model_status_tag.pack(side="right")

        self.model_options = self._get_model_display_list()
        self.model_dropdown_var = ctk.StringVar(value=self.model_options[0])
        self.model_dropdown = ctk.CTkOptionMenu(
            model_card,
            values=self.model_options,
            variable=self.model_dropdown_var,
            font=FONT_BODY,
            fg_color=COLOR_BG_INPUT,
            button_color=COLOR_BORDER,
            button_hover_color=COLOR_BORDER_FOCUS,
            text_color=COLOR_TEXT_PRIMARY,
            dropdown_fg_color=COLOR_BG_CARD,
            dropdown_text_color=COLOR_TEXT_PRIMARY,
            dropdown_hover_color=COLOR_BG_CARD_HOVER,
            corner_radius=CORNER_RADIUS_SM,
            height=34,
            command=self._on_model_changed,
        )
        self.model_dropdown.pack(fill="x", padx=14, pady=(0, 6))

        self.model_desc_label = ctk.CTkLabel(
            model_card,
            text=MODEL_REGISTRY["large-v3-turbo"]["description"],
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_MUTED,
            wraplength=370,
            justify="left",
        )
        self.model_desc_label.pack(fill="x", padx=14, pady=(0, 12))

        # 3. Карточка выбора языка
        lang_card = ctk.CTkFrame(
            left_col,
            fg_color=COLOR_BG_CARD,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=CORNER_RADIUS_MD,
        )
        lang_card.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            lang_card,
            text="Язык аудио:",
            font=FONT_SUBTITLE,
            text_color=COLOR_TEXT_PRIMARY,
        ).pack(anchor="w", padx=14, pady=(12, 6))

        self.lang_map = {
            "Автоопределение (Auto)": None,
            "Русский (ru)": "ru",
            "Английский (en)": "en",
            "Немецкий (de)": "de",
            "Французский (fr)": "fr",
            "Испанский (es)": "es",
            "Китайский (zh)": "zh",
        }
        self.lang_dropdown_var = ctk.StringVar(value="Автоопределение (Auto)")
        self.lang_dropdown = ctk.CTkOptionMenu(
            lang_card,
            values=list(self.lang_map.keys()),
            variable=self.lang_dropdown_var,
            font=FONT_BODY,
            fg_color=COLOR_BG_INPUT,
            button_color=COLOR_BORDER,
            button_hover_color=COLOR_BORDER_FOCUS,
            text_color=COLOR_TEXT_PRIMARY,
            dropdown_fg_color=COLOR_BG_CARD,
            dropdown_text_color=COLOR_TEXT_PRIMARY,
            dropdown_hover_color=COLOR_BG_CARD_HOVER,
            corner_radius=CORNER_RADIUS_SM,
            height=34,
        )
        self.lang_dropdown.pack(fill="x", padx=14, pady=(0, 12))

        # 4. Сворачиваемая секция «Расширенные настройки»
        self.adv_section = CollapsibleSection(left_col, title="Расширенные настройки")
        self.adv_section.pack(fill="x", pady=(0, 14))

        # Контент расширенных настроек
        adv_content = self.adv_section.content_frame

        # Compute Type
        ctk.CTkLabel(
            adv_content,
            text="Тип вычислений (Compute Type):",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_SECONDARY,
        ).pack(anchor="w", pady=(6, 2))

        default_compute = "float16" if self.gpu_info.get("cuda_ready", False) else "int8"
        self.compute_options = [
            "float16 (Рекомендуется для RTX)",
            "int8_float16 (Экономия памяти)",
            "int8 (Быстрый)",
            "float32 (Резервный)",
        ]
        self.compute_type_var = ctk.StringVar(value=self.compute_options[0] if default_compute == "float16" else self.compute_options[2])
        self.compute_dropdown = ctk.CTkOptionMenu(
            adv_content,
            values=self.compute_options,
            variable=self.compute_type_var,
            font=FONT_CAPTION,
            fg_color=COLOR_BG_INPUT,
            button_color=COLOR_BORDER,
            button_hover_color=COLOR_BORDER_FOCUS,
            text_color=COLOR_TEXT_PRIMARY,
            dropdown_fg_color=COLOR_BG_CARD,
            dropdown_text_color=COLOR_TEXT_PRIMARY,
            dropdown_hover_color=COLOR_BG_CARD_HOVER,
            corner_radius=CORNER_RADIUS_SM,
            height=30,
        )
        self.compute_dropdown.pack(fill="x", pady=(0, 10))

        # VAD Filter
        self.vad_var = ctk.BooleanVar(value=True)
        self.vad_switch = ctk.CTkSwitch(
            adv_content,
            text="VAD-фильтр пауз и шумов (Silero)",
            variable=self.vad_var,
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_PRIMARY,
            fg_color=COLOR_BG_INPUT,
            progress_color=COLOR_ACCENT_WHITE,
            button_color=COLOR_ACCENT_HOVER,
        )
        self.vad_switch.pack(anchor="w", pady=(0, 10))

        # Beam Size
        beam_row = ctk.CTkFrame(adv_content, fg_color="transparent")
        beam_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            beam_row,
            text="Точность поиска (Beam Size):",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_SECONDARY,
        ).pack(side="left")
        self.beam_val_label = ctk.CTkLabel(
            beam_row,
            text="5",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_PRIMARY,
        )
        self.beam_val_label.pack(side="right")

        self.beam_slider = ctk.CTkSlider(
            adv_content,
            from_=1,
            to=5,
            number_of_steps=4,
            height=14,
            fg_color=COLOR_BG_INPUT,
            progress_color=COLOR_ACCENT_WHITE,
            button_color=COLOR_ACCENT_WHITE,
            button_hover_color=COLOR_ACCENT_HOVER,
            command=self._on_beam_slider_changed,
        )
        self.beam_slider.set(5)
        self.beam_slider.pack(fill="x", pady=(0, 10))

        # Опции оптимизации и сохранения
        self.autosave_var = ctk.BooleanVar(value=True)
        self.autosave_switch = ctk.CTkSwitch(
            adv_content,
            text="Автосохранение (.txt рядом с аудио)",
            variable=self.autosave_var,
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_PRIMARY,
            fg_color=COLOR_BG_INPUT,
            progress_color=COLOR_ACCENT_WHITE,
            button_color=COLOR_ACCENT_HOVER,
        )
        self.autosave_switch.pack(anchor="w", pady=(0, 8))

        self.live_preview_var = ctk.BooleanVar(value=True)
        self.live_preview_switch = ctk.CTkSwitch(
            adv_content,
            text="Живой предпросмотр текста",
            variable=self.live_preview_var,
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_PRIMARY,
            fg_color=COLOR_BG_INPUT,
            progress_color=COLOR_ACCENT_WHITE,
            button_color=COLOR_ACCENT_HOVER,
        )
        self.live_preview_switch.pack(anchor="w", pady=(0, 6))

        # 5. Главная кнопка действия (CTA)
        self.action_button = ctk.CTkButton(
            left_col,
            text="Начать транскрибирование",
            font=FONT_BODY_BOLD,
            fg_color=COLOR_ACCENT_WHITE,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_ACCENT_TEXT,
            height=46,
            corner_radius=CORNER_RADIUS_MD,
            command=self._toggle_transcription,
        )
        self.action_button.pack(fill="x", pady=(8, 12))

        # ==========================================================
        # Правая колонка (Транскрипт, прогресс, поиск, экспорт)
        # ==========================================================
        right_col = ctk.CTkFrame(
            container,
            fg_color=COLOR_BG_SECONDARY,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=CORNER_RADIUS_LG,
        )
        right_col.pack(side="right", fill="both", expand=True)

        # 1. Верхняя панель статуса и прогресса
        progress_card = ctk.CTkFrame(right_col, fg_color="transparent")
        progress_card.pack(fill="x", padx=20, pady=(16, 10))

        prog_label_row = ctk.CTkFrame(progress_card, fg_color="transparent")
        prog_label_row.pack(fill="x", pady=(0, 6))

        self.status_text_label = ctk.CTkLabel(
            prog_label_row,
            text="Готов к работе. Выберите или перетащите файл.",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_SECONDARY,
        )
        self.status_text_label.pack(side="left")

        self.speed_badge = ctk.CTkLabel(
            prog_label_row,
            text="",
            font=FONT_CAPTION,
            text_color=COLOR_SUCCESS,
        )
        self.speed_badge.pack(side="right")

        self.progress_bar = ctk.CTkProgressBar(
            progress_card,
            height=4,
            corner_radius=CORNER_RADIUS_SM,
            fg_color=COLOR_BG_CARD,
            progress_color=COLOR_ACCENT_WHITE,
        )
        self.progress_bar.pack(fill="x")
        self.progress_bar.set(0.0)

        # 2. Тулбар транскрипта (Переключатель режимов, поиск, копирование)
        transcript_toolbar = ctk.CTkFrame(right_col, fg_color="transparent")
        transcript_toolbar.pack(fill="x", padx=20, pady=(6, 8))

        # Переключатель: Чистый текст / С таймкодами
        self.view_mode_segmented = ctk.CTkSegmentedButton(
            transcript_toolbar,
            values=["Чистый текст", "С таймкодами"],
            font=FONT_CAPTION,
            fg_color=COLOR_BG_INPUT,
            selected_color=COLOR_ACCENT_WHITE,
            selected_hover_color=COLOR_ACCENT_HOVER,
            corner_radius=CORNER_RADIUS_SM,
            height=32,
            command=self._on_view_mode_changed,
        )
        # Гарантируем контраст черного текста на белом фоне при выборе любого сегмента
        orig_select = self.view_mode_segmented._select_button_by_value
        orig_unselect = self.view_mode_segmented._unselect_button_by_value

        def custom_select(value: str):
            orig_select(value)
            if hasattr(self.view_mode_segmented, "_buttons_dict") and value in self.view_mode_segmented._buttons_dict:
                self.view_mode_segmented._buttons_dict[value].configure(text_color=COLOR_ACCENT_TEXT)

        def custom_unselect(value: str):
            orig_unselect(value)
            if hasattr(self.view_mode_segmented, "_buttons_dict") and value in self.view_mode_segmented._buttons_dict:
                self.view_mode_segmented._buttons_dict[value].configure(text_color=COLOR_TEXT_SECONDARY)

        self.view_mode_segmented._select_button_by_value = custom_select
        self.view_mode_segmented._unselect_button_by_value = custom_unselect

        self.view_mode_segmented.set("Чистый текст")
        self.view_mode_segmented.pack(side="left", padx=(0, 12))
        self.after(50, self._update_view_mode_styles)

        # Поиск по транскрипту
        self.search_entry = ctk.CTkEntry(
            transcript_toolbar,
            placeholder_text="Поиск в тексте...",
            font=FONT_CAPTION,
            fg_color=COLOR_BG_INPUT,
            border_color=COLOR_BORDER,
            text_color=COLOR_TEXT_PRIMARY,
            height=32,
            width=200,
            corner_radius=CORNER_RADIUS_SM,
        )
        self.search_entry.pack(side="left", padx=(0, 6))
        self.search_entry.bind("<KeyRelease>", self._on_search_text_changed)

        # Кнопка копирования
        self.copy_btn = ctk.CTkButton(
            transcript_toolbar,
            text="Скопировать",
            font=FONT_CAPTION,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            height=32,
            corner_radius=CORNER_RADIUS_SM,
            command=self._copy_to_clipboard,
        )
        self.copy_btn.pack(side="right")

        # 3. Текстовое поле результата
        self.textbox_frame = ctk.CTkFrame(right_col, fg_color="transparent")
        self.textbox_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        self.textbox = ctk.CTkTextbox(
            self.textbox_frame,
            font=FONT_BODY,
            fg_color=COLOR_BG_CARD,
            border_color=COLOR_BORDER,
            border_width=1,
            text_color=COLOR_TEXT_PRIMARY,
            wrap="word",
            corner_radius=CORNER_RADIUS_MD,
        )
        self.textbox.pack(fill="both", expand=True)
        # Настройка тега подсветки поиска
        self.textbox._textbox.tag_configure("search_highlight", background="#3F3F46", foreground="#FFFFFF")

        # 4. Панель экспорта
        export_toolbar = ctk.CTkFrame(right_col, fg_color="transparent", height=42)
        export_toolbar.pack(fill="x", padx=20, pady=(0, 16))

        ctk.CTkLabel(
            export_toolbar,
            text="Экспорт:",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_MUTED,
        ).pack(side="left", padx=(0, 10))

        export_buttons = [
            (".TXT (Текст)", self._export_txt),
            (".SRT (Субтитры)", self._export_srt),
            (".VTT (Веб)", self._export_vtt),
            (".JSON (Данные)", self._export_json),
            ("📂 Открыть папку", self._open_file_location),
        ]
        for label, cmd in export_buttons:
            btn = ctk.CTkButton(
                export_toolbar,
                text=label,
                font=FONT_CAPTION,
                fg_color=COLOR_BTN_SECONDARY,
                hover_color=COLOR_BTN_SECONDARY_HOVER,
                text_color=COLOR_TEXT_PRIMARY,
                height=30,
                corner_radius=CORNER_RADIUS_SM,
                command=cmd,
            )
            btn.pack(side="left", padx=4)

        # Обновление бейджа модели
        self._update_model_badge()

    # ==========================================================
    # Логика работы с моделями
    # ==========================================================
    def _get_model_display_list(self) -> List[str]:
        items = []
        for key, info in MODEL_REGISTRY.items():
            cached = is_model_cached(key)
            tag = " [Скачана]" if cached else " [Требуется загрузка]"
            items.append(f"{key}{tag}")
        return items

    def _get_selected_model_key(self) -> str:
        raw = self.model_dropdown_var.get()
        return raw.split(" ")[0].strip()

    def _on_model_changed(self, choice: str):
        model_key = self._get_selected_model_key()
        info = MODEL_REGISTRY.get(model_key, {})
        self.model_desc_label.configure(text=info.get("description", ""))
        self._update_model_badge()

    def _update_model_badge(self):
        model_key = self._get_selected_model_key()
        cached = is_model_cached(model_key)
        if cached:
            self.model_status_tag.configure(text="● Готова", text_color=COLOR_SUCCESS)
        else:
            self.model_status_tag.configure(text="○ Не загружена", text_color=COLOR_WARNING)

    def _on_beam_slider_changed(self, value):
        self.beam_val_label.configure(text=str(int(value)))

    def _check_first_run(self):
        """Если ни одна модель не скачана, предлагает Onboarding."""
        if not any_model_cached():
            OnboardingDialog(self, on_complete=self._on_onboarding_completed)

    def _on_onboarding_completed(self, downloaded_model: str):
        # Обновляем список моделей
        self.model_options = self._get_model_display_list()
        self.model_dropdown.configure(values=self.model_options)
        for opt in self.model_options:
            if opt.startswith(downloaded_model):
                self.model_dropdown_var.set(opt)
                break
        self._update_model_badge()

    # ==========================================================
    # Выбор файла
    # ==========================================================
    def _on_file_selected(self, file_path: str):
        self.selected_file = file_path
        media_info = get_media_info(file_path)
        self.status_text_label.configure(
            text=f"Выбран: {media_info['filename']} ({media_info['duration_str']}, {media_info['size_mb']} MB)",
            text_color=COLOR_TEXT_PRIMARY,
        )

    # ==========================================================
    # Процесс транскрибирования
    # ==========================================================
    def _toggle_transcription(self):
        if self.is_transcribing:
            # Остановка
            self.cancel_event.set()
            self.action_button.configure(text="Останавливаю...", state="disabled")
            self.status_text_label.configure(text="Остановка процесса...", text_color=COLOR_WARNING)
            return

        if not self.selected_file:
            messagebox.showwarning("Файл не выбран", "Пожалуйста, перетащите или выберите аудиофайл перед запуском.")
            return

        model_key = self._get_selected_model_key()

        # Если модель не скачана, запускаем скачивание
        if not is_model_cached(model_key):
            confirm = messagebox.askyesno(
                "Загрузка модели",
                f"Модель '{model_key}' еще не скачана на компьютер.\nСкачать ее сейчас?",
            )
            if not confirm:
                return

            self._download_and_start(model_key)
            return

        self._start_transcription_pipeline(model_key)

    def _download_and_start(self, model_key: str):
        self.action_button.configure(state="disabled", text="Загрузка весов...")
        self.status_text_label.configure(text="Загрузка модели с Hugging Face...")

        def prog_cb(fraction, current_mb, total_mb, speed_mb_s, status_text):
            self.after(0, lambda: self._update_download_progress(fraction, status_text))

        def worker():
            try:
                download_model(model_key, progress_callback=prog_cb)
                self.after(0, lambda: self._on_download_complete_and_start(model_key))
            except Exception as e:
                self.after(0, lambda: self._on_transcription_error(f"Ошибка загрузки модели: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _update_download_progress(self, fraction: float, text: str):
        self.progress_bar.set(fraction)
        self.status_text_label.configure(text=text)

    def _on_download_complete_and_start(self, model_key: str):
        self._update_model_badge()
        self.model_options = self._get_model_display_list()
        self.model_dropdown.configure(values=self.model_options)
        self._start_transcription_pipeline(model_key)

    def _start_transcription_pipeline(self, model_key: str):
        self.is_transcribing = True
        self.cancel_event.clear()
        self.current_segments = []
        self._stream_para_count = 0
        self._stream_char_count = 0
        self._stream_last_end = 0.0

        self.textbox.delete("1.0", "end")
        if not self.live_preview_var.get():
            self.textbox.insert(
                "1.0",
                "[Живой предпросмотр текста отключен для максимальной производительности]\n\n"
                "Распознавание выполняется в фоновом режиме на максимальной скорости GPU.\n"
                "Полный текст с разбивкой на абзацы появится здесь сразу по окончании процесса.\n"
            )

        # Очищаем очередь от прошлых сообщений
        while not self.transcribe_queue.empty():
            try:
                self.transcribe_queue.get_nowait()
            except queue.Empty:
                break

        # Переключаем кнопку в режим "Остановить"
        self.action_button.configure(
            text="Остановить",
            fg_color=COLOR_DANGER,
            hover_color="#DC2626",
            text_color=COLOR_TEXT_PRIMARY,
            state="normal",
        )
        self.progress_bar.set(0.0)
        self.speed_badge.configure(text="")
        self.status_text_label.configure(text="Инициализация модели и анализ аудио...", text_color=COLOR_TEXT_PRIMARY)

        # Параметры инференса
        compute_choice = self.compute_type_var.get().split(" ")[0]
        lang_choice = self.lang_map.get(self.lang_dropdown_var.get(), None)
        vad_choice = self.vad_var.get()
        beam_choice = int(self.beam_slider.get())

        def progress_cb(fraction, segment, speed, status_str):
            # Потокобезопасная передача в очередь без перегрузки Tcl/Tk главного потока
            self.transcribe_queue.put(("SEGMENT", (fraction, segment, speed, status_str)))

        def worker():
            try:
                res = self.transcriber.transcribe_file(
                    audio_path=self.selected_file,
                    model_name=model_key,
                    compute_type=compute_choice,
                    language=lang_choice,
                    vad_filter=vad_choice,
                    beam_size=beam_choice,
                    condition_on_previous_text=False,
                    progress_callback=progress_cb,
                    cancel_event=self.cancel_event,
                )
                self.transcribe_queue.put(("SUCCESS", res))
            except TranscriptionCancelledException:
                self.transcribe_queue.put(("CANCELLED", None))
            except Exception as e:
                self.transcribe_queue.put(("ERROR", str(e)))

        threading.Thread(target=worker, daemon=True).start()
        # Запуск мягкого периодического опроса очереди каждые 100 мс
        self.after(100, self._poll_transcription_queue)

    def _poll_transcription_queue(self):
        new_segments_batch = []
        latest_fraction = None
        latest_speed = None
        latest_status = None
        terminal_event = None

        # Забираем все накопившиеся за 100 мс сегменты
        while not self.transcribe_queue.empty():
            try:
                msg_type, payload = self.transcribe_queue.get_nowait()
                if msg_type == "SEGMENT":
                    frac, seg, spd, st_str = payload
                    new_segments_batch.append(seg)
                    latest_fraction = frac
                    latest_speed = spd
                    latest_status = st_str
                elif msg_type in ("SUCCESS", "CANCELLED", "ERROR"):
                    terminal_event = (msg_type, payload)
            except queue.Empty:
                break

        # Батчевая вставка текста с естественной разбивкой на абзацы (200+ FPS)
        if new_segments_batch:
            self.current_segments.extend(new_segments_batch)

            if self.live_preview_var.get():
                mode = self.view_mode_segmented.get()
                if mode == "С таймкодами":
                    lines = []
                    for s in new_segments_batch:
                        t_start = format_seconds_to_time(s["start"])
                        t_end = format_seconds_to_time(s["end"])
                        lines.append(f"[{t_start} -> {t_end}]  {s['text']}\n")
                    batch_text = "".join(lines)
                else:
                    pieces = []
                    for s in new_segments_batch:
                        text = s.get("text", "").strip()
                        if not text:
                            continue
                        start = s.get("start", 0.0)
                        end = s.get("end", 0.0)

                        # Естественная пауза лектора в речи (> 1.8 сек)
                        long_pause = (self._stream_last_end > 0 and (start - self._stream_last_end) > 1.8)
                        ends_sentence = text.endswith((".", "!", "?", "…", "..."))

                        self._stream_para_count += 1
                        self._stream_char_count += len(text) + 1
                        self._stream_last_end = end

                        if (self._stream_para_count >= 3 and ends_sentence) or (self._stream_char_count >= 400 and ends_sentence) or (long_pause and self._stream_para_count >= 2):
                            pieces.append(text + "\n\n")
                            self._stream_para_count = 0
                            self._stream_char_count = 0
                        else:
                            pieces.append(text + " ")
                    batch_text = "".join(pieces)

                if batch_text:
                    self.textbox.insert("end", batch_text)
                    self.textbox.see("end")

        # Плавное обновление прогресса не чаще 10 раз в секунду
        if latest_fraction is not None:
            self.progress_bar.set(latest_fraction)
        if latest_speed is not None:
            self.speed_badge.configure(text=f"● {latest_speed:.1f}x")
        if latest_status is not None:
            self.status_text_label.configure(text=latest_status)

        # Обработка финального статуса
        if terminal_event is not None:
            msg_type, payload = terminal_event
            if msg_type == "SUCCESS":
                self._on_transcription_success(payload)
            elif msg_type == "CANCELLED":
                self._on_transcription_cancelled()
            elif msg_type == "ERROR":
                self._on_transcription_error(payload)
            return

        # Если инференс продолжается, планируем следующий такт
        if self.is_transcribing:
            self.after(100, self._poll_transcription_queue)

    def _on_transcription_success(self, results: dict):
        self.is_transcribing = False
        self.current_results = results
        self.progress_bar.set(1.0)

        elapsed = results.get("elapsed_seconds", 0)
        speed = results.get("speed_multiplier", 0)
        lang = results.get("detected_language", "ru")

        # Автоматическое сохранение .txt файла рядом с аудиозаписью
        saved_msg = ""
        if self.autosave_var.get() and self.selected_file:
            try:
                audio_path = Path(self.selected_file)
                out_txt = audio_path.parent / f"{audio_path.stem}_transcript.txt"
                export_txt(self.current_segments, str(out_txt), include_timestamps=False)
                self.last_saved_path = str(out_txt)
                saved_msg = f" • Сохранён: {out_txt.name}"
            except Exception as e:
                print(f"Auto-save warning: {e}")

        self.status_text_label.configure(
            text=f"✓ Готово за {elapsed:.1f}с! Скорость: {speed:.1f}x (Язык: {lang.upper()}){saved_msg}",
            text_color=COLOR_SUCCESS,
        )
        self.speed_badge.configure(text=f"● Средняя скорость: {speed:.1f}x")

        # Возвращаем белую CTA кнопку
        self.action_button.configure(
            text="Начать транскрибирование",
            fg_color=COLOR_ACCENT_WHITE,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_ACCENT_TEXT,
            state="normal",
        )

        # Полное форматирование текста в окне
        self._refresh_textbox_content()

        if self.last_saved_path:
            self.toast = ToastMessage(self, f"✓ Текст автосохранён: {Path(self.last_saved_path).name}")
            self.toast.show()

    def _on_transcription_cancelled(self):
        self.is_transcribing = False
        self.status_text_label.configure(text="Процесс остановлен пользователем.", text_color=COLOR_WARNING)
        self.action_button.configure(
            text="Начать транскрибирование",
            fg_color=COLOR_ACCENT_WHITE,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_ACCENT_TEXT,
            state="normal",
        )

    def _on_transcription_error(self, err_msg: str):
        self.is_transcribing = False
        self.status_text_label.configure(text=f"Ошибка: {err_msg[:60]}...", text_color=COLOR_DANGER)
        self.action_button.configure(
            text="Начать транскрибирование",
            fg_color=COLOR_ACCENT_WHITE,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_ACCENT_TEXT,
            state="normal",
        )
        messagebox.showerror("Ошибка распознавания", err_msg)

    # ==========================================================
    # Отображение, поиск и буфер
    # ==========================================================
    def _update_view_mode_styles(self):
        """Обеспечивает контрастный цвет текста на активной кнопке режима."""
        selected_mode = self.view_mode_segmented.get()
        if hasattr(self.view_mode_segmented, "_buttons_dict"):
            for val, btn in self.view_mode_segmented._buttons_dict.items():
                if val == selected_mode:
                    btn.configure(text_color=COLOR_ACCENT_TEXT)
                else:
                    btn.configure(text_color=COLOR_TEXT_SECONDARY)

    def _on_view_mode_changed(self, mode: str):
        self._update_view_mode_styles()
        self._refresh_textbox_content()

    def _refresh_textbox_content(self):
        mode = self.view_mode_segmented.get()
        self.textbox.delete("1.0", "end")

        if not self.current_segments:
            return

        if mode == "С таймкодами":
            lines = []
            for s in self.current_segments:
                t_start = format_seconds_to_time(s["start"])
                t_end = format_seconds_to_time(s["end"])
                lines.append(f"[{t_start} -> {t_end}]  {s['text']}")
            self.textbox.insert("1.0", "\n".join(lines))
        else:
            # Чистый текст структурируем на естественные абзацы по паузам и предложениям
            formatted_text = format_clean_paragraphs(self.current_segments)
            self.textbox.insert("1.0", formatted_text)

        self._on_search_text_changed()

    def _on_search_text_changed(self, event=None):
        query = self.search_entry.get().strip().lower()
        self.textbox._textbox.tag_remove("search_highlight", "1.0", "end")
        if not query:
            return

        start_pos = "1.0"
        while True:
            pos = self.textbox._textbox.search(query, start_pos, stopindex="end", nocase=True)
            if not pos:
                break
            end_pos = f"{pos}+{len(query)}c"
            self.textbox._textbox.tag_add("search_highlight", pos, end_pos)
            start_pos = end_pos

    def _copy_to_clipboard(self):
        text = self.textbox.get("1.0", "end-1c").strip()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.toast.show()

    # ==========================================================
    # Экспорт результатов
    # ==========================================================
    def _get_export_default_name(self, ext: str) -> str:
        if self.selected_file:
            stem = Path(self.selected_file).stem
            return f"{stem}_transcript.{ext}"
        return f"transcript.{ext}"

    def _export_txt(self):
        if not self.current_segments:
            messagebox.showinfo("Нет данных", "Сначала выполните транскрибирование.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=self._get_export_default_name("txt"),
            filetypes=[("Text file", "*.txt")],
        )
        if file_path:
            include_time = (self.view_mode_segmented.get() == "С таймкодами")
            export_txt(self.current_segments, file_path, include_timestamps=include_time)
            self.toast = ToastMessage(self, "✓ Файл .TXT успешно сохранён!")
            self.toast.show()

    def _export_srt(self):
        if not self.current_segments:
            messagebox.showinfo("Нет данных", "Сначала выполните транскрибирование.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".srt",
            initialfile=self._get_export_default_name("srt"),
            filetypes=[("SubRip Subtitles", "*.srt")],
        )
        if file_path:
            export_srt(self.current_segments, file_path)
            self.toast = ToastMessage(self, "✓ Субтитры .SRT успешно сохранены!")
            self.toast.show()

    def _export_vtt(self):
        if not self.current_segments:
            messagebox.showinfo("Нет данных", "Сначала выполните транскрибирование.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".vtt",
            initialfile=self._get_export_default_name("vtt"),
            filetypes=[("WebVTT Subtitles", "*.vtt")],
        )
        if file_path:
            export_vtt(self.current_segments, file_path)
            self.toast = ToastMessage(self, "✓ Субтитры .VTT успешно сохранены!")
            self.toast.show()

    def _export_json(self):
        if not self.current_segments:
            messagebox.showinfo("Нет данных", "Сначала выполните транскрибирование.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=self._get_export_default_name("json"),
            filetypes=[("JSON File", "*.json")],
        )
        if file_path:
            meta = self.current_results if self.current_results else {"total_segments": len(self.current_segments)}
            export_json(self.current_segments, meta, file_path)
            self.toast = ToastMessage(self, "✓ Файл .JSON успешно сохранён!")
            self.toast.show()

    def _open_file_location(self):
        """Открывает папку с сохраненным файлом или исходным аудио в Проводнике Windows."""
        target = self.last_saved_path or self.selected_file
        if target and os.path.exists(target):
            try:
                subprocess.Popen(f'explorer /select,"{os.path.abspath(target)}"')
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось открыть папку: {e}")
        elif self.selected_file and os.path.exists(self.selected_file):
            try:
                subprocess.Popen(f'explorer /select,"{os.path.abspath(self.selected_file)}"')
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось открыть папку: {e}")
        else:
            messagebox.showinfo("Папка с файлом", "Сначала выберите файл или запустите транскрибирование.")
