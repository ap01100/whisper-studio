"""AI-Summarizer View for Whisper Studio.
Provides modern, two-mode UI for Qwen 2.5 and DeepSeek LLM inference:
- 'Исходный текст' (Lecture source editor with real-time stats, file loading, and direct editing)
- 'Готовый конспект' (Live token streaming, preset selection, and dual Markdown / Plain-text export)
"""
import os
import sys
import time
import queue
import threading
import subprocess
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

import customtkinter as ctk
import tkinter as tk
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
    FONT_CAPTION_BOLD,
    CORNER_RADIUS_SM,
    CORNER_RADIUS_MD,
    CORNER_RADIUS_LG,
)
from ui.components import CollapsibleSection, ToastMessage, bind_text_shortcuts, bind_paste_support
from ui.hf_token_dialog import HFTokenDialog
from core.hf_downloader import (
    LLM_REGISTRY,
    is_llm_cached,
    download_llm_model,
    get_llm_model_path,
    get_hf_token,
)
from core.llm_engine import LLMEngine, LLMCancelledException
from core.summarizer import (
    LectureSummarizer,
    PROMPT_PRESETS,
    strip_markdown,
    export_summary_md,
    export_summary_txt,
    load_text_from_file,
)
from core.vram_manager import get_vram_manager

try:
    import windnd
    HAS_WINDND = True
except ImportError:
    HAS_WINDND = False


class SummaryView(ctk.CTkFrame):
    """Вкладка 'AI-Конспект' с локальным инференсом Qwen 2.5 / DeepSeek."""

    def __init__(
        self,
        master,
        llm_engine: Optional[LLMEngine] = None,
        on_open_hf_dialog: Optional[Callable[[], None]] = None,
        get_transcription_text: Optional[Callable[[], str]] = None,
        get_transcription_filename: Optional[Callable[[], Optional[str]]] = None,
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.engine = llm_engine if llm_engine is not None else LLMEngine()
        self.summarizer = LectureSummarizer(self.engine)
        self.on_open_hf_dialog = on_open_hf_dialog
        self.get_transcription_text = get_transcription_text
        self.get_transcription_filename = get_transcription_filename

        # Состояние вкладки
        self.source_filename: Optional[str] = None
        self.current_summary: str = ""
        self.is_generating = False
        self.is_downloading = False
        self.cancel_event = threading.Event()
        self.token_queue = queue.Queue()
        self.last_saved_path: Optional[str] = None
        self.active_mode = "source"  # "source" или "summary"

        # Toast для уведомлений
        self.toast = ToastMessage(self)

        self._build_layout()
        self._setup_drag_and_drop()

    def _build_layout(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True)

        # ==========================================================
        # Левая колонка (Управление, выбор модели, пресеты) ~ 410px
        # ==========================================================
        left_col = ctk.CTkScrollableFrame(
            container,
            width=410,
            fg_color="transparent",
            scrollbar_button_color=COLOR_BG_CARD,
            scrollbar_button_hover_color=COLOR_BORDER,
        )
        left_col.pack(side="left", fill="y", padx=(0, 16))

        # 1. Карточка выбора LLM-модели
        model_card = ctk.CTkFrame(
            left_col,
            fg_color=COLOR_BG_CARD,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=CORNER_RADIUS_MD,
        )
        model_card.pack(fill="x", pady=(0, 12))

        m_header = ctk.CTkFrame(model_card, fg_color="transparent")
        m_header.pack(fill="x", padx=14, pady=(12, 6))

        ctk.CTkLabel(
            m_header,
            text="Модель конспектирования:",
            font=FONT_SUBTITLE,
            text_color=COLOR_TEXT_PRIMARY,
        ).pack(side="left")

        self.model_status_badge = ctk.CTkLabel(
            m_header,
            text="○ Проверка",
            font=FONT_CAPTION,
            text_color=COLOR_WARNING,
        )
        self.model_status_badge.pack(side="right")

        # Выпадающий список моделей
        self.model_options_map = {
            f"{info['name']}": m_id for m_id, info in LLM_REGISTRY.items()
        }
        self.model_dropdown_var = ctk.StringVar(value=list(self.model_options_map.keys())[0])
        self.model_dropdown = ctk.CTkOptionMenu(
            model_card,
            values=list(self.model_options_map.keys()),
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
            text="",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_SECONDARY,
            wraplength=370,
            justify="left",
        )
        self.model_desc_label.pack(fill="x", padx=14, pady=(0, 10))

        # Кнопка ручной загрузки модели (отображается, если модель не скачана)
        self.download_model_btn = ctk.CTkButton(
            model_card,
            text="⬇️ Скачать выбранную модель",
            font=FONT_CAPTION_BOLD,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            height=30,
            corner_radius=CORNER_RADIUS_SM,
            command=self._on_direct_download_clicked,
        )

        # Прогресс-бар скачивания модели (скрыт по умолчанию)
        self.download_prog_frame = ctk.CTkFrame(model_card, fg_color="transparent")
        self.download_prog_label = ctk.CTkLabel(
            self.download_prog_frame,
            text="",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_SECONDARY,
        )
        self.download_prog_label.pack(anchor="w", pady=(0, 2))
        self.download_prog_bar = ctk.CTkProgressBar(
            self.download_prog_frame,
            height=4,
            corner_radius=CORNER_RADIUS_SM,
            fg_color=COLOR_BG_INPUT,
            progress_color=COLOR_ACCENT_WHITE,
        )
        self.download_prog_bar.pack(fill="x")
        self.download_prog_bar.set(0.0)

        # 2. Карточка выбора шаблона (пресета)
        preset_card = ctk.CTkFrame(
            left_col,
            fg_color=COLOR_BG_CARD,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=CORNER_RADIUS_MD,
        )
        preset_card.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            preset_card,
            text="Шаблон конспекта:",
            font=FONT_SUBTITLE,
            text_color=COLOR_TEXT_PRIMARY,
        ).pack(anchor="w", padx=14, pady=(12, 6))

        self.preset_options_map = {
            info["title"]: key for key, info in PROMPT_PRESETS.items()
        }
        self.preset_dropdown_var = ctk.StringVar(value=list(self.preset_options_map.keys())[0])
        self.preset_dropdown = ctk.CTkOptionMenu(
            preset_card,
            values=list(self.preset_options_map.keys()),
            variable=self.preset_dropdown_var,
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
            command=self._on_preset_changed,
        )
        self.preset_dropdown.pack(fill="x", padx=14, pady=(0, 6))

        self.preset_desc_label = ctk.CTkLabel(
            preset_card,
            text="",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_SECONDARY,
            wraplength=370,
            justify="left",
        )
        self.preset_desc_label.pack(fill="x", padx=14, pady=(0, 10))

        # Поле пользовательского промпта (для пресета 'custom')
        self.custom_prompt_frame = ctk.CTkFrame(preset_card, fg_color="transparent")
        ctk.CTkLabel(
            self.custom_prompt_frame,
            text="Пользовательская инструкция:",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_PRIMARY,
        ).pack(anchor="w", pady=(0, 4))
        self.custom_prompt_entry = ctk.CTkTextbox(
            self.custom_prompt_frame,
            font=FONT_CAPTION,
            fg_color=COLOR_BG_INPUT,
            border_color=COLOR_BORDER,
            border_width=1,
            text_color=COLOR_TEXT_PRIMARY,
            height=70,
            corner_radius=CORNER_RADIUS_SM,
        )
        self.custom_prompt_entry.pack(fill="x", pady=(0, 10))
        bind_text_shortcuts(self.custom_prompt_entry, allow_edit=True)

        # 3. Аккордеон расширенных настроек LLM
        self.adv_section = CollapsibleSection(left_col, title="Расширенные настройки LLM")
        self.adv_section.pack(fill="x", pady=(0, 12))
        adv_content = self.adv_section.content_frame

        # Размер контекстного окна
        ctk.CTkLabel(
            adv_content,
            text="Размер контекста (Context Window):",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_SECONDARY,
        ).pack(anchor="w", pady=(4, 2))

        self.ctx_options = ["16384 токенов (Рекомендуется для 8GB)", "8192 токенов (Быстрый)", "32768 токенов (Максимальный)"]
        self.ctx_var = ctk.StringVar(value=self.ctx_options[0])
        self.ctx_dropdown = ctk.CTkOptionMenu(
            adv_content,
            values=self.ctx_options,
            variable=self.ctx_var,
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
        self.ctx_dropdown.pack(fill="x", pady=(0, 10))

        # Температура
        temp_row = ctk.CTkFrame(adv_content, fg_color="transparent")
        temp_row.pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(
            temp_row,
            text="Креативность (Temperature):",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_SECONDARY,
        ).pack(side="left")
        self.temp_val_label = ctk.CTkLabel(
            temp_row,
            text="0.6",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_PRIMARY,
        )
        self.temp_val_label.pack(side="right")

        self.temp_slider = ctk.CTkSlider(
            adv_content,
            from_=0.1,
            to=1.0,
            number_of_steps=9,
            height=14,
            fg_color=COLOR_BG_INPUT,
            progress_color=COLOR_ACCENT_WHITE,
            button_color=COLOR_ACCENT_WHITE,
            button_hover_color=COLOR_ACCENT_HOVER,
            command=lambda v: self.temp_val_label.configure(text=f"{v:.1f}"),
        )
        self.temp_slider.set(0.6)
        self.temp_slider.pack(fill="x", pady=(0, 10))

        # 4. Кнопка настройки токена Hugging Face
        self.hf_token_btn = ctk.CTkButton(
            left_col,
            text="🔑 Настройки токена Hugging Face Hub",
            font=FONT_CAPTION,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            height=34,
            corner_radius=CORNER_RADIUS_SM,
            command=self._open_hf_dialog,
        )
        self.hf_token_btn.pack(fill="x", pady=(0, 12))

        # 5. Главная кнопка действия (CTA)
        self.action_btn = ctk.CTkButton(
            left_col,
            text="✨ Сформировать конспект",
            font=FONT_BODY_BOLD,
            fg_color=COLOR_ACCENT_WHITE,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_ACCENT_TEXT,
            height=46,
            corner_radius=CORNER_RADIUS_MD,
            command=self._toggle_generation,
        )
        self.action_btn.pack(fill="x", pady=(4, 12))

        # ==========================================================
        # Правая колонка: Двухрежимная панель (Исходный текст / Результат)
        # ==========================================================
        self.right_col = ctk.CTkFrame(
            container,
            fg_color=COLOR_BG_SECONDARY,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=CORNER_RADIUS_LG,
        )
        self.right_col.pack(side="right", fill="both", expand=True)

        # 1. Верхняя панель переключения режимов (Подвкладки)
        self._build_right_header()

        # 2. Контейнер режима «Исходный текст»
        self._build_source_view()

        # 3. Контейнер режима «Готовый конспект»
        self._build_summary_result_view()

        # По умолчанию показываем «Исходный текст»
        self._switch_mode("📝 Исходный текст")

        # Начальная инициализация
        self._on_model_changed()
        self._on_preset_changed()

    # ==========================================================
    # Отрисовка правой части интерфейса
    # ==========================================================

    def _build_right_header(self):
        header_bar = ctk.CTkFrame(self.right_col, fg_color="transparent")
        header_bar.pack(fill="x", padx=16, pady=(14, 8))

        # Переключатель режимов
        self.mode_switch = ctk.CTkSegmentedButton(
            header_bar,
            values=["📝 Исходный текст", "✨ Готовый конспект"],
            font=FONT_BODY_BOLD,
            fg_color=COLOR_BG_INPUT,
            selected_color=COLOR_ACCENT_WHITE,
            selected_hover_color=COLOR_ACCENT_HOVER,
            corner_radius=CORNER_RADIUS_SM,
            height=34,
            command=self._switch_mode,
        )

        orig_select = self.mode_switch._select_button_by_value
        orig_unselect = self.mode_switch._unselect_button_by_value

        def custom_select(value: str):
            orig_select(value)
            if hasattr(self.mode_switch, "_buttons_dict") and value in self.mode_switch._buttons_dict:
                self.mode_switch._buttons_dict[value].configure(text_color=COLOR_ACCENT_TEXT)

        def custom_unselect(value: str):
            orig_unselect(value)
            if hasattr(self.mode_switch, "_buttons_dict") and value in self.mode_switch._buttons_dict:
                self.mode_switch._buttons_dict[value].configure(text_color=COLOR_TEXT_SECONDARY)

        self.mode_switch._select_button_by_value = custom_select
        self.mode_switch._unselect_button_by_value = custom_unselect

        self.mode_switch.set("📝 Исходный текст")
        self.mode_switch.pack(side="left")

        # Индикатор файла/источника
        self.file_indicator_label = ctk.CTkLabel(
            header_bar,
            text="",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_MUTED,
        )
        self.file_indicator_label.pack(side="left", padx=(14, 0))

        # Бейдж скорости генерации
        self.speed_badge = ctk.CTkLabel(
            header_bar,
            text="",
            font=FONT_CAPTION,
            text_color=COLOR_SUCCESS,
        )
        self.speed_badge.pack(side="right")

    def _update_mode_switch_styles(self):
        selected = self.mode_switch.get()
        if hasattr(self.mode_switch, "_buttons_dict"):
            for val, btn in self.mode_switch._buttons_dict.items():
                if val == selected:
                    btn.configure(text_color=COLOR_ACCENT_TEXT)
                else:
                    btn.configure(text_color=COLOR_TEXT_SECONDARY)

    def _build_source_view(self):
        """Вью режима '📝 Исходный текст' для ввода, вставки и загрузки файлов."""
        self.source_view_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")

        # Тулбар действий с текстом
        toolbar = ctk.CTkFrame(self.source_view_frame, fg_color="transparent", height=38)
        toolbar.pack(fill="x", padx=16, pady=(0, 8))

        ctk.CTkButton(
            toolbar,
            text="📂 Загрузить файл (.txt / .md / .srt)",
            font=FONT_CAPTION,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            height=30,
            corner_radius=CORNER_RADIUS_SM,
            command=self._load_file_dialog,
        ).pack(side="left", padx=(0, 6))

        self.import_trans_btn = ctk.CTkButton(
            toolbar,
            text="🎙️ Взять из транскрибации",
            font=FONT_CAPTION,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            height=30,
            corner_radius=CORNER_RADIUS_SM,
            command=self._import_from_transcription,
        )
        self.import_trans_btn.pack(side="left", padx=3)

        ctk.CTkButton(
            toolbar,
            text="📋 Вставить",
            font=FONT_CAPTION,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            height=30,
            corner_radius=CORNER_RADIUS_SM,
            command=self._paste_from_clipboard,
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            toolbar,
            text="🗑️ Очистить",
            font=FONT_CAPTION,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_MUTED,
            height=30,
            corner_radius=CORNER_RADIUS_SM,
            command=self._clear_source_text,
        ).pack(side="left", padx=3)

        # Текстовое поле ввода исходной лекции
        textbox_frame = ctk.CTkFrame(self.source_view_frame, fg_color="transparent")
        textbox_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        self.source_textbox = ctk.CTkTextbox(
            textbox_frame,
            font=FONT_BODY,
            fg_color=COLOR_BG_CARD,
            border_color=COLOR_BORDER,
            border_width=1,
            text_color=COLOR_TEXT_PRIMARY,
            wrap="word",
            corner_radius=CORNER_RADIUS_MD,
        )
        self.source_textbox.pack(fill="both", expand=True)
        bind_text_shortcuts(self.source_textbox, allow_edit=True)

        # Overlay-подсказка (placeholder), видимая когда поле пустое
        self.source_placeholder = ctk.CTkLabel(
            self.source_textbox,
            text=(
                "Вставьте текст лекции сюда (Ctrl+V), перетащите текстовый файл\n"
                "или нажмите «Загрузить файл» / «Взять из транскрибации» выше..."
            ),
            font=FONT_BODY,
            text_color=COLOR_TEXT_MUTED,
            justify="left",
            anchor="nw",
        )
        self.source_placeholder.place(relx=0.02, rely=0.03, anchor="nw")
        self.source_placeholder.bind("<Button-1>", lambda e: self.source_textbox.focus())

        # Привязка событий для подсчета статистики и управления плейсхолдером
        inner_tb = getattr(self.source_textbox, "_textbox", None)
        if inner_tb:
            inner_tb.bind("<KeyRelease>", self._update_source_stats)
            inner_tb.bind("<FocusIn>", lambda e: self.source_placeholder.place_forget())
            inner_tb.bind("<FocusOut>", lambda e: self._on_source_focus_out())

        # Нижняя строка статистики
        bottom_bar = ctk.CTkFrame(self.source_view_frame, fg_color="transparent", height=28)
        bottom_bar.pack(fill="x", padx=16, pady=(0, 14))

        self.source_stats_label = ctk.CTkLabel(
            bottom_bar,
            text="Слов: 0  |  Токенов: ~0  |  Символов: 0",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_MUTED,
        )
        self.source_stats_label.pack(side="left")

        ctk.CTkLabel(
            bottom_bar,
            text="💡 Вы можете свободно редактировать текст перед запуском",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_MUTED,
        ).pack(side="right")

    def _build_summary_result_view(self):
        """Вью режима '✨ Готовый конспект' для потокового вывода и экспорта."""
        self.summary_view_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")

        # Статусная панель
        status_panel = ctk.CTkFrame(self.summary_view_frame, fg_color="transparent")
        status_panel.pack(fill="x", padx=16, pady=(0, 8))

        status_row = ctk.CTkFrame(status_panel, fg_color="transparent")
        status_row.pack(fill="x", pady=(0, 4))

        self.status_label = ctk.CTkLabel(
            status_row,
            text="Ожидание запуска конспектирования.",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_SECONDARY,
        )
        self.status_label.pack(side="left")

        self.progress_bar = ctk.CTkProgressBar(
            status_panel,
            height=4,
            corner_radius=CORNER_RADIUS_SM,
            fg_color=COLOR_BG_CARD,
            progress_color=COLOR_ACCENT_WHITE,
        )
        self.progress_bar.pack(fill="x")
        self.progress_bar.set(0.0)

        # Текстовая область конспекта
        textbox_frame = ctk.CTkFrame(self.summary_view_frame, fg_color="transparent")
        textbox_frame.pack(fill="both", expand=True, padx=16, pady=(4, 8))

        self.summary_textbox = ctk.CTkTextbox(
            textbox_frame,
            font=FONT_BODY,
            fg_color=COLOR_BG_CARD,
            border_color=COLOR_BORDER,
            border_width=1,
            text_color=COLOR_TEXT_PRIMARY,
            wrap="word",
            corner_radius=CORNER_RADIUS_MD,
        )
        self.summary_textbox.pack(fill="both", expand=True)
        bind_text_shortcuts(self.summary_textbox, allow_edit=True)

        self._show_summary_initial_placeholder()

        # Панель экспорта (Markdown и чистый текст)
        export_toolbar = ctk.CTkFrame(self.summary_view_frame, fg_color="transparent", height=38)
        export_toolbar.pack(fill="x", padx=16, pady=(0, 14))

        # Буфер
        ctk.CTkLabel(
            export_toolbar,
            text="Буфер:",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_MUTED,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            export_toolbar,
            text="Копировать MD",
            font=FONT_CAPTION,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            height=30,
            corner_radius=CORNER_RADIUS_SM,
            command=self._copy_markdown,
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            export_toolbar,
            text="Копировать текст",
            font=FONT_CAPTION,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            height=30,
            corner_radius=CORNER_RADIUS_SM,
            command=self._copy_plain_text,
        ).pack(side="left", padx=3)

        # Файлы
        ctk.CTkLabel(
            export_toolbar,
            text="Сохранить:",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_MUTED,
        ).pack(side="left", padx=(12, 6))

        ctk.CTkButton(
            export_toolbar,
            text=".MD (Markdown)",
            font=FONT_CAPTION,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            height=30,
            corner_radius=CORNER_RADIUS_SM,
            command=self._export_markdown,
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            export_toolbar,
            text=".TXT (Без разметки)",
            font=FONT_CAPTION,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            height=30,
            corner_radius=CORNER_RADIUS_SM,
            command=self._export_plain_text,
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            export_toolbar,
            text="📂 Папка",
            font=FONT_CAPTION,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            height=30,
            corner_radius=CORNER_RADIUS_SM,
            command=self._open_file_location,
        ).pack(side="left", padx=3)

    def _show_summary_initial_placeholder(self):
        self.summary_textbox.delete("1.0", "end")
        self.summary_textbox.insert(
            "1.0",
            "Здесь появится структурированный конспект лекции.\n\n"
            "1. Вставьте или загрузите текст лекции на вкладке «📝 Исходный текст».\n"
            "2. Выберите модель Qwen 2.5 и желаемый шаблон (Академический конспект, Тезисы, Вопросы).\n"
            "3. Нажмите кнопку «✨ Сформировать конспект».\n\n"
            "Готовый конспект можно сохранить как в Markdown (.md), так и в чистый текст без спецсимволов (.txt)."
        )

    # ==========================================================
    # Переключение подвкладок
    # ==========================================================

    def _switch_mode(self, mode_name: str):
        self.mode_switch.set(mode_name)
        self._update_mode_switch_styles()
        if mode_name == "📝 Исходный текст":
            self.summary_view_frame.pack_forget()
            self.source_view_frame.pack(fill="both", expand=True)
            self.active_mode = "source"
        else:
            self.source_view_frame.pack_forget()
            self.summary_view_frame.pack(fill="both", expand=True)
            self.active_mode = "summary"

    # ==========================================================
    # Работа с исходным текстом лекции
    # ==========================================================

    def _update_source_stats(self, event=None):
        text = self.source_textbox.get("1.0", "end-1c").strip()
        if text:
            self.source_placeholder.place_forget()
            words = len(text.split())
            chars = len(text)
            tokens = int(words * 1.35)
            self.source_stats_label.configure(
                text=f"Слов: {words:,}  |  Токенов: ~{tokens:,}  |  Символов: {chars:,}"
            )
        else:
            self.source_stats_label.configure(
                text="Слов: 0  |  Токенов: ~0  |  Символов: 0"
            )

    def _on_source_focus_out(self):
        text = self.source_textbox.get("1.0", "end-1c").strip()
        if not text:
            self.source_placeholder.place(relx=0.02, rely=0.03, anchor="nw")

    def get_source_text(self) -> str:
        """Возвращает актуальный текст из редактора исходного текста."""
        return self.source_textbox.get("1.0", "end-1c").strip()

    def set_source_text(self, text: str, filename: Optional[str] = None):
        """Заполняет поле исходного текста и обновляет статистику."""
        clean_text = text.strip()
        self.source_textbox.delete("1.0", "end")
        if clean_text:
            self.source_textbox.insert("1.0", clean_text)
            self.source_placeholder.place_forget()
        else:
            self.source_placeholder.place(relx=0.02, rely=0.03, anchor="nw")

        self.source_filename = filename
        if filename:
            self.file_indicator_label.configure(text=f"Источник: «{Path(filename).name}»")
        else:
            self.file_indicator_label.configure(text="Источник: Введён вручную")

        self._update_source_stats()

    def load_transcript(self, text: str, source_filename: Optional[str] = None):
        """Совместимость с внешними вызовами: загрузка стенограммы в редактор."""
        self.set_source_text(text, source_filename)

    def _paste_from_clipboard(self):
        try:
            text = self.clipboard_get()
            if not text:
                messagebox.showinfo("Буфер обмена", "Буфер обмена пуст.")
                return
            self.source_textbox.insert("end", text)
            self.source_placeholder.place_forget()
            self._update_source_stats()
            self.toast = ToastMessage(self, "✓ Текст вставлен из буфера")
            self.toast.show()
        except Exception:
            messagebox.showinfo("Буфер обмена", "Не удалось прочитать текст из буфера обмена.")

    def _clear_source_text(self):
        text = self.get_source_text()
        if not text:
            return
        confirm = messagebox.askyesno("Очистить текст", "Удалить введённый текст лекции?")
        if confirm:
            self.source_textbox.delete("1.0", "end")
            self.source_placeholder.place(relx=0.02, rely=0.03, anchor="nw")
            self.source_filename = None
            self.file_indicator_label.configure(text="")
            self._update_source_stats()
            self.toast = ToastMessage(self, "Поле очищено")
            self.toast.show()

    def _load_file_dialog(self):
        path = filedialog.askopenfilename(
            title="Выберите файл с текстом лекции",
            filetypes=[
                ("Текстовые документы и субтитры", "*.txt *.md *.srt *.vtt *.json"),
                ("Текстовые файлы (*.txt)", "*.txt"),
                ("Markdown (*.md)", "*.md"),
                ("Субтитры (*.srt, *.vtt)", "*.srt *.vtt"),
                ("JSON стенограммы (*.json)", "*.json"),
                ("Все файлы (*.*)", "*.*"),
            ],
        )
        if path:
            self._load_file_path(path)

    def _load_file_path(self, path: str):
        try:
            text = load_text_from_file(path)
            if not text:
                messagebox.showwarning("Пустой файл", f"Файл «{Path(path).name}» не содержит текста.")
                return
            self.set_source_text(text, path)
            self._switch_mode("📝 Исходный текст")
            self.toast = ToastMessage(self, f"✓ Загружен: {Path(path).name}")
            self.toast.show()
        except Exception as e:
            messagebox.showerror("Ошибка чтения файла", f"Не удалось загрузить файл:\n{e}")

    def _import_from_transcription(self):
        text = ""
        filename = None
        if self.get_transcription_text:
            text = self.get_transcription_text()
        if self.get_transcription_filename:
            filename = self.get_transcription_filename()

        if not text:
            messagebox.showinfo(
                "Нет транскрипта",
                "На первой вкладке («Аудио и Транскрибация») еще нет распознанного текста.\n"
                "Выполните распознавание аудио или загрузите текстовый файл по кнопке «Загрузить файл».",
            )
            return

        self.set_source_text(text, filename)
        self.toast = ToastMessage(self, "✓ Текст перенесён из транскрибации")
        self.toast.show()

    def _setup_drag_and_drop(self):
        """Подключает перетаскивание текстовых файлов через windnd."""
        if HAS_WINDND:
            def on_drop(files):
                if files:
                    f = files[0]
                    if isinstance(f, bytes):
                        try:
                            f = f.decode("utf-8")
                        except UnicodeDecodeError:
                            f = f.decode("mbcs", errors="replace")
                    self.after(0, lambda: self._load_file_path(f))

            try:
                windnd.hook_dropfiles(self.source_view_frame, func=on_drop)
                windnd.hook_dropfiles(self.source_textbox, func=on_drop)
            except Exception:
                pass

    # ==========================================================
    # Управление моделью и пресетами
    # ==========================================================

    def _get_selected_model_id(self) -> str:
        label = self.model_dropdown_var.get()
        return self.model_options_map.get(label, "qwen-2.5-7b")

    def _get_selected_preset_key(self) -> str:
        title = self.preset_dropdown_var.get()
        return self.preset_options_map.get(title, "academic")

    def _on_model_changed(self, choice=None):
        m_id = self._get_selected_model_id()
        info = LLM_REGISTRY.get(m_id, {})
        self.model_desc_label.configure(text=info.get("description", ""))

        if is_llm_cached(m_id):
            self.model_status_badge.configure(text="● Скачана", text_color=COLOR_SUCCESS)
            self.download_model_btn.pack_forget()
        else:
            size_mb = info.get("size_mb", 4000)
            self.model_status_badge.configure(
                text=f"○ Требуется загрузка (~{size_mb / 1024:.1f} GB)",
                text_color=COLOR_WARNING,
            )
            if not self.is_downloading:
                self.download_model_btn.pack(fill="x", padx=14, pady=(0, 10))

    def _on_direct_download_clicked(self):
        """Прямая загрузка модели по кнопке из карточки модели."""
        if self.is_downloading or self.is_generating:
            return
        m_id = self._get_selected_model_id()
        if is_llm_cached(m_id):
            messagebox.showinfo("Модель готова", "Выбранная модель уже загружена в локальное хранилище.")
            return

        info = LLM_REGISTRY.get(m_id, {})
        size_gb = info.get("size_mb", 4000) / 1024
        m_name = info.get("name", m_id)
        has_token = bool(get_hf_token())
        token_hint = "" if has_token else "\n(Совет: добавьте токен Hugging Face для повышенной скорости загрузки!)"

        confirm = messagebox.askyesno(
            "Загрузка модели LLM",
            f"Начать загрузку модели {m_name} (~{size_gb:.1f} GB) с Hugging Face Hub?{token_hint}",
        )
        if confirm:
            source_text = self.get_source_text()
            self._download_and_start(m_id, pending_text=source_text)

    def _on_preset_changed(self, choice=None):
        preset_key = self._get_selected_preset_key()
        info = PROMPT_PRESETS.get(preset_key, {})
        self.preset_desc_label.configure(text=info.get("description", ""))

        if preset_key == "custom":
            self.custom_prompt_frame.pack(fill="x", padx=14, pady=(0, 10))
        else:
            self.custom_prompt_frame.pack_forget()

    def _open_hf_dialog(self):
        HFTokenDialog(self.winfo_toplevel(), on_saved=lambda t: self._on_model_changed())

    # ==========================================================
    # Генерация конспекта
    # ==========================================================

    def _toggle_generation(self):
        if self.is_downloading:
            self.cancel_event.set()
            self.action_btn.configure(text="Останавливаю загрузку...", state="disabled")
            self.status_label.configure(text="Отмена загрузки модели...", text_color=COLOR_WARNING)
            return

        if self.is_generating:
            # Остановка
            self.cancel_event.set()
            self.action_btn.configure(text="Останавливаю...", state="disabled")
            self.status_label.configure(text="Остановка генерации...", text_color=COLOR_WARNING)
            return

        source_text = self.get_source_text()

        # Если в исходном тексте пусто, проверяем, не вставил ли пользователь текст в область конспекта
        if not source_text:
            summary_raw = self.summary_textbox.get("1.0", "end-1c").strip()
            if summary_raw and not summary_raw.startswith("Здесь появится структурированный конспект"):
                source_text = summary_raw
                self.set_source_text(source_text)
                self.toast = ToastMessage(self, "Текст перемещён в «Исходный текст»")
                self.toast.show()

        # Если всё ещё пусто, проверяем, доступен ли текст на вкладке транскрибации
        if not source_text and self.get_transcription_text:
            trans_text = self.get_transcription_text()
            if trans_text:
                fname = self.get_transcription_filename() if self.get_transcription_filename else None
                import_confirm = messagebox.askyesno(
                    "Использовать распознанный текст",
                    "Поле исходного текста пусто, но на вкладке «Аудио и Транскрибация» есть готовый текст.\n\n"
                    "Использовать его для составления конспекта лекции?",
                )
                if import_confirm:
                    self.set_source_text(trans_text, fname)
                    source_text = trans_text

        if not source_text:
            messagebox.showwarning(
                "Нет текста для конспекта",
                "Пожалуйста, вставьте текст лекции в поле «Исходный текст», "
                "загрузите файл (.txt, .md, .srt) или выполните распознавание аудиозаписи.",
            )
            self._switch_mode("📝 Исходный текст")
            self.source_textbox.focus()
            return

        model_id = self._get_selected_model_id()

        # Если модель не скачана, предлагаем загрузить
        if not is_llm_cached(model_id):
            info = LLM_REGISTRY.get(model_id, {})
            size_gb = info.get("size_mb", 4000) / 1024
            m_name = info.get("name", model_id)
            has_token = bool(get_hf_token())
            token_hint = "" if has_token else "\n(Совет: добавьте бесплатный токен HF для высокой скорости загрузки!)"

            confirm = messagebox.askyesno(
                "Загрузка модели LLM",
                f"Модель {m_name} (~{size_gb:.1f} GB) еще не скачана на компьютер.\n"
                f"Начать загрузку с Hugging Face Hub прямо сейчас?{token_hint}",
            )
            if not confirm:
                return

            self._download_and_start(model_id, source_text)
            return

        # Переключаемся на вкладку готового конспекта и запускаем
        self._switch_mode("✨ Готовый конспект")
        self._start_summarization_pipeline(model_id, source_text)

    def _download_and_start(self, model_id: str, pending_text: str = ""):
        self.is_downloading = True
        self.cancel_event.clear()
        self.download_model_btn.pack_forget()
        self.action_btn.configure(
            state="normal",
            text="❌ Отменить загрузку",
            fg_color=COLOR_DANGER,
            hover_color="#DC2626",
            text_color=COLOR_TEXT_PRIMARY,
        )
        self.download_prog_bar.set(0.0)
        self.download_prog_label.configure(text="Подключение к Hugging Face Hub...")
        self.status_label.configure(text="Подключение к Hugging Face Hub...", text_color=COLOR_TEXT_PRIMARY)
        self.download_prog_frame.pack(fill="x", padx=14, pady=(0, 10))

        def prog_cb(fraction, current_mb, total_mb, speed_mb_s, status_text):
            self.after(0, lambda: self._update_download_progress(fraction, status_text))

        def worker():
            try:
                download_llm_model(model_id, progress_callback=prog_cb, cancel_event=self.cancel_event)
                self.after(0, lambda: self._on_download_complete(model_id, pending_text))
            except Exception as e:
                self.after(0, lambda: self._on_download_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _update_download_progress(self, fraction: float, text: str):
        self.download_prog_bar.set(fraction)
        self.download_prog_label.configure(text=text)
        self.status_label.configure(text=text)

    def _on_download_complete(self, model_id: str, pending_text: str):
        self.is_downloading = False
        self.download_prog_frame.pack_forget()
        self.action_btn.configure(
            state="normal",
            text="✨ Сформировать конспект",
            fg_color=COLOR_ACCENT_WHITE,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_ACCENT_TEXT,
        )
        self._on_model_changed()
        if pending_text and pending_text.strip():
            self._switch_mode("✨ Готовый конспект")
            self._start_summarization_pipeline(model_id, pending_text)
        else:
            self.status_label.configure(text="Модель успешно загружена и готова к работе!", text_color=COLOR_SUCCESS)
            messagebox.showinfo("Загрузка завершена", "Модель успешно загружена и готова к формированию конспектов!")

    def _on_download_error(self, err_msg: str):
        self.is_downloading = False
        self.download_prog_frame.pack_forget()
        self.action_btn.configure(
            state="normal",
            text="✨ Сформировать конспект",
            fg_color=COLOR_ACCENT_WHITE,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_ACCENT_TEXT,
        )
        self._on_model_changed()
        if "отменена" in err_msg.lower():
            self.status_label.configure(text="Загрузка модели отменена пользователем.", text_color=COLOR_TEXT_SECONDARY)
        else:
            self.status_label.configure(text="Ошибка загрузки модели.", text_color=COLOR_DANGER)
            messagebox.showerror("Ошибка загрузки", err_msg)

    def _start_summarization_pipeline(self, model_id: str, transcript_text: str):
        self.is_generating = True
        self.cancel_event.clear()
        self.current_summary = ""
        self.summary_textbox.delete("1.0", "end")

        # Очищаем очередь токенов
        while not self.token_queue.empty():
            try:
                self.token_queue.get_nowait()
            except queue.Empty:
                break

        # Кнопка в режим "Остановить"
        self.action_btn.configure(
            text="Остановить генерацию",
            fg_color=COLOR_DANGER,
            hover_color="#DC2626",
            text_color=COLOR_TEXT_PRIMARY,
            state="normal",
        )
        self.progress_bar.set(0.0)
        self.speed_badge.configure(text="")
        self.status_label.configure(text="Инициализация модели и анализ структуры лекции...", text_color=COLOR_TEXT_PRIMARY)

        # Параметры генерации
        preset_key = self._get_selected_preset_key()
        custom_inst = self.custom_prompt_entry.get("1.0", "end-1c").strip() if preset_key == "custom" else None
        ctx_choice = int(self.ctx_var.get().split()[0])
        temp_val = float(self.temp_slider.get())

        def token_cb(token_chunk, full_text_so_far, speed_tps, status_str):
            self.token_queue.put(("TOKEN", (token_chunk, speed_tps, status_str)))

        def worker():
            try:
                res = self.summarizer.summarize(
                    transcript_text=transcript_text,
                    model_id=model_id,
                    preset_key=preset_key,
                    custom_instructions=custom_inst,
                    context_window=ctx_choice,
                    temperature=temp_val,
                    progress_callback=token_cb,
                    cancel_event=self.cancel_event,
                )
                self.token_queue.put(("SUCCESS", res))
            except LLMCancelledException:
                self.token_queue.put(("CANCELLED", None))
            except Exception as e:
                self.token_queue.put(("ERROR", str(e)))

        threading.Thread(target=worker, daemon=True).start()
        self.after(50, self._poll_token_queue)

    def _poll_token_queue(self):
        new_tokens_batch = []
        latest_speed = None
        latest_status = None
        terminal_event = None

        while not self.token_queue.empty():
            try:
                msg_type, payload = self.token_queue.get_nowait()
                if msg_type == "TOKEN":
                    tok, spd, st = payload
                    new_tokens_batch.append(tok)
                    latest_speed = spd
                    latest_status = st
                elif msg_type in ("SUCCESS", "CANCELLED", "ERROR"):
                    terminal_event = (msg_type, payload)
            except queue.Empty:
                break

        if new_tokens_batch:
            batch_text = "".join(new_tokens_batch)
            self.summary_textbox.insert("end", batch_text)
            self.summary_textbox.see("end")

        if latest_speed is not None:
            self.speed_badge.configure(text=f"● {latest_speed:.1f} т/с")
        if latest_status is not None:
            self.status_label.configure(text=latest_status)

        if terminal_event is not None:
            msg_type, payload = terminal_event
            if msg_type == "SUCCESS":
                self._on_generation_success(payload)
            elif msg_type == "CANCELLED":
                self._on_generation_cancelled()
            elif msg_type == "ERROR":
                self._on_generation_error(payload)
            return

        if self.is_generating:
            self.after(50, self._poll_token_queue)

    def _on_generation_success(self, full_summary: str):
        self.is_generating = False
        self.current_summary = full_summary
        self.progress_bar.set(1.0)

        self.action_btn.configure(
            text="✨ Сформировать конспект",
            fg_color=COLOR_ACCENT_WHITE,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_ACCENT_TEXT,
            state="normal",
        )

        # Автоматическое сохранение конспекта рядом с файлом источника
        auto_saved_name = ""
        if self.source_filename:
            try:
                p = Path(self.source_filename)
                out_md = p.parent / f"{p.stem}_summary.md"
                out_txt = p.parent / f"{p.stem}_summary.txt"
                export_summary_md(full_summary, str(out_md))
                export_summary_txt(full_summary, str(out_txt))
                self.last_saved_path = str(out_md)
                auto_saved_name = f" • Сохранено: {out_md.name} и {out_txt.name}"
            except Exception as e:
                print(f"Auto-save summary error: {e}")

        self.status_label.configure(
            text=f"✓ Конспект готов!{auto_saved_name}",
            text_color=COLOR_SUCCESS,
        )

        if self.last_saved_path:
            self.toast = ToastMessage(self, "✓ Конспект сохранён (.md и .txt)")
            self.toast.show()

    def _on_generation_cancelled(self):
        self.is_generating = False
        self.status_label.configure(text="Генерация остановлена пользователем.", text_color=COLOR_WARNING)
        self.action_btn.configure(
            text="✨ Сформировать конспект",
            fg_color=COLOR_ACCENT_WHITE,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_ACCENT_TEXT,
            state="normal",
        )

    def _on_generation_error(self, err_msg: str):
        self.is_generating = False
        self.status_label.configure(text=f"Ошибка: {err_msg[:60]}...", text_color=COLOR_DANGER)
        self.action_btn.configure(
            text="✨ Сформировать конспект",
            fg_color=COLOR_ACCENT_WHITE,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_ACCENT_TEXT,
            state="normal",
        )
        messagebox.showerror("Ошибка генерации конспекта", err_msg)

    # ==========================================================
    # Экспорт и буфер обмена
    # ==========================================================

    def _get_current_content(self) -> str:
        text = self.summary_textbox.get("1.0", "end-1c").strip()
        if text.startswith("Здесь появится структурированный конспект"):
            return ""
        return text if text else self.current_summary

    def _copy_markdown(self):
        content = self._get_current_content()
        if not content:
            messagebox.showinfo("Нет конспекта", "Сначала сгенерируйте конспект лекции.")
            return
        self.clipboard_clear()
        self.clipboard_append(content)
        self.toast = ToastMessage(self, "✓ Скопирован Markdown")
        self.toast.show()

    def _copy_plain_text(self):
        content = self._get_current_content()
        if not content:
            messagebox.showinfo("Нет конспекта", "Сначала сгенерируйте конспект лекции.")
            return
        plain_text = strip_markdown(content)
        self.clipboard_clear()
        self.clipboard_append(plain_text)
        self.toast = ToastMessage(self, "✓ Скопирован чистый текст (без разметки)")
        self.toast.show()

    def _export_markdown(self):
        content = self._get_current_content()
        if not content:
            messagebox.showinfo("Нет конспекта", "Сначала сгенерируйте конспект.")
            return

        stem = Path(self.source_filename).stem if self.source_filename else "lecture"
        path = filedialog.asksaveasfilename(
            defaultextension=".md",
            initialfile=f"{stem}_summary.md",
            filetypes=[("Markdown Document", "*.md"), ("All files", "*.*")],
        )
        if path:
            export_summary_md(content, path)
            self.last_saved_path = path
            self.toast = ToastMessage(self, "✓ Конспект .MD сохранён!")
            self.toast.show()

    def _export_plain_text(self):
        content = self._get_current_content()
        if not content:
            messagebox.showinfo("Нет конспекта", "Сначала сгенерируйте конспект.")
            return

        stem = Path(self.source_filename).stem if self.source_filename else "lecture"
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"{stem}_summary.txt",
            filetypes=[("Text File (без разметки)", "*.txt"), ("All files", "*.*")],
        )
        if path:
            export_summary_txt(content, path)
            self.last_saved_path = path
            self.toast = ToastMessage(self, "✓ Чистый текст .TXT сохранён!")
            self.toast.show()

    def _open_file_location(self):
        target = self.last_saved_path or self.source_filename
        if target and os.path.exists(target):
            try:
                subprocess.Popen(f'explorer /select,"{os.path.abspath(target)}"')
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось открыть папку: {e}")
        else:
            messagebox.showinfo("Папка с файлом", "Сначала сохраните конспект или откройте файл.")
