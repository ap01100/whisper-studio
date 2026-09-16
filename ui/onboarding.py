"""Onboarding modal dialog for first-time model setup.
"""
import threading
from typing import Optional, Callable

import customtkinter as ctk

from ui.theme import (
    COLOR_BG_PRIMARY,
    COLOR_BG_CARD,
    COLOR_BG_CARD_HOVER,
    COLOR_BORDER,
    COLOR_BORDER_FOCUS,
    COLOR_ACCENT_WHITE,
    COLOR_ACCENT_HOVER,
    COLOR_ACCENT_TEXT,
    COLOR_BTN_SECONDARY,
    COLOR_BTN_SECONDARY_HOVER,
    COLOR_SUCCESS,
    COLOR_DANGER,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED,
    FONT_TITLE,
    FONT_SUBTITLE,
    FONT_BODY,
    FONT_BODY_BOLD,
    FONT_CAPTION,
    CORNER_RADIUS_SM,
    CORNER_RADIUS_MD,
    CORNER_RADIUS_LG,
)
from core.model_manager import MODEL_REGISTRY, download_model


class OnboardingDialog(ctk.CTkToplevel):
    """Модальное окно первого запуска с выбором и скачиванием стартовой модели."""

    def __init__(self, parent, on_complete: Optional[Callable[[str], None]] = None):
        super().__init__(parent)
        self.parent = parent
        self.on_complete = on_complete

        self.title("Первый запуск — Whisper Studio")
        self.geometry("640x560")
        self.resizable(False, False)
        self.configure(fg_color=COLOR_BG_PRIMARY)

        # Центрирование относительно родительского окна
        self.transient(parent)
        self.grab_set()

        self.selected_model_var = ctk.StringVar(value="large-v3-turbo")
        self.is_downloading = False
        self.stop_event = threading.Event()

        self._build_ui()

    def _build_ui(self):
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=28, pady=24)

        # Header
        header_frame = ctk.CTkFrame(content, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 16))

        title = ctk.CTkLabel(
            header_frame,
            text="Добро пожаловать в Whisper Studio",
            font=FONT_TITLE,
            text_color=COLOR_TEXT_PRIMARY,
        )
        title.pack(anchor="w")

        subtitle = ctk.CTkLabel(
            header_frame,
            text="Для начала работы выберите стартовую модель нейросети.\nВеса сохранятся локально в папке ./models и будут готовы для офлайн-работы.",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_SECONDARY,
            justify="left",
        )
        subtitle.pack(anchor="w", pady=(4, 0))

        # Карточки вариантов моделей
        models_frame = ctk.CTkFrame(content, fg_color="transparent")
        models_frame.pack(fill="x", pady=(0, 16))

        options = [
            ("large-v3-turbo", "Оптимальная (Рекомендуется для RTX 4060 / 5070)",
             "1.6 GB • В 8 раз быстрее large-v3 при сравнимом высочайшем качестве"),
            ("base", "Базовая (Мгновенный тест)",
             "145 MB • Быстрая загрузка за пару секунд для проверки системы"),
            ("large-v3", "Максимальная (Для сложных записей)",
             "3.1 GB • Предельная точность распознавания шумов, шепота и терминов"),
        ]

        self.radio_buttons = []
        for model_key, opt_title, opt_desc in options:
            card = ctk.CTkFrame(
                models_frame,
                fg_color=COLOR_BG_CARD,
                border_color=COLOR_BORDER,
                border_width=1,
                corner_radius=CORNER_RADIUS_MD,
            )
            card.pack(fill="x", pady=6)

            radio = ctk.CTkRadioButton(
                card,
                text=opt_title,
                value=model_key,
                variable=self.selected_model_var,
                font=FONT_BODY_BOLD,
                text_color=COLOR_TEXT_PRIMARY,
                fg_color=COLOR_ACCENT_WHITE,
                hover_color=COLOR_ACCENT_HOVER,
                border_color=COLOR_BORDER,
            )
            radio.pack(anchor="w", padx=16, pady=(12, 2))
            self.radio_buttons.append(radio)

            desc = ctk.CTkLabel(
                card,
                text=opt_desc,
                font=FONT_CAPTION,
                text_color=COLOR_TEXT_MUTED,
            )
            desc.pack(anchor="w", padx=44, pady=(0, 12))

        # Секция прогресса
        self.progress_frame = ctk.CTkFrame(content, fg_color="transparent")
        self.progress_frame.pack(fill="x", pady=(8, 12))

        self.status_label = ctk.CTkLabel(
            self.progress_frame,
            text="Выберите вариант и нажмите «Скачать и начать»",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        )
        self.status_label.pack(fill="x", pady=(0, 6))

        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame,
            height=6,
            corner_radius=CORNER_RADIUS_SM,
            fg_color=COLOR_BG_CARD,
            progress_color=COLOR_ACCENT_WHITE,
        )
        self.progress_bar.pack(fill="x")
        self.progress_bar.set(0.0)

        # Нижняя панель действий
        bottom_frame = ctk.CTkFrame(content, fg_color="transparent")
        bottom_frame.pack(fill="x", side="bottom", pady=(12, 0))

        self.action_btn = ctk.CTkButton(
            bottom_frame,
            text="Скачать и начать",
            font=FONT_BODY_BOLD,
            fg_color=COLOR_ACCENT_WHITE,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_ACCENT_TEXT,
            height=40,
            corner_radius=CORNER_RADIUS_MD,
            command=self._start_download,
        )
        self.action_btn.pack(side="right", padx=(10, 0))

        self.cancel_btn = ctk.CTkButton(
            bottom_frame,
            text="Пропустить",
            font=FONT_BODY,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_SECONDARY,
            height=40,
            corner_radius=CORNER_RADIUS_MD,
            command=self._on_skip,
        )
        self.cancel_btn.pack(side="right")

    def _start_download(self):
        if self.is_downloading:
            return

        chosen_model = self.selected_model_var.get()
        self.is_downloading = True
        self.stop_event.clear()

        # Блокируем выбор
        for r in self.radio_buttons:
            r.configure(state="disabled")
        self.action_btn.configure(state="disabled", text="Загрузка...")
        self.cancel_btn.configure(text="Отмена", command=self._cancel_download)

        def progress_callback(fraction, current_mb, total_mb, speed_mb_s, status_text):
            # Безопасное обновление GUI через after
            self.after(0, lambda: self._update_progress_ui(fraction, status_text))

        def worker():
            try:
                download_model(
                    chosen_model,
                    progress_callback=progress_callback,
                    stop_event=self.stop_event,
                )
                self.after(0, lambda: self._on_download_success(chosen_model))
            except Exception as e:
                self.after(0, lambda: self._on_download_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _update_progress_ui(self, fraction: float, status_text: str):
        self.progress_bar.set(fraction)
        self.status_label.configure(text=status_text)

    def _on_download_success(self, model_name: str):
        self.is_downloading = False
        self.progress_bar.configure(progress_color=COLOR_SUCCESS)
        self.progress_bar.set(1.0)
        self.status_label.configure(
            text=f"✓ Модель {model_name} успешно загружена и готова к работе!",
            text_color=COLOR_SUCCESS,
        )
        self.action_btn.configure(
            state="normal",
            text="Готово к работе",
            command=lambda: self._finish(model_name),
        )
        self.cancel_btn.pack_forget()

        # Автоматическое закрытие через 1.5 секунды
        self.after(1500, lambda: self._finish(model_name))

    def _on_download_error(self, err_msg: str):
        self.is_downloading = False
        for r in self.radio_buttons:
            r.configure(state="normal")
        self.action_btn.configure(state="normal", text="Повторить")
        self.cancel_btn.configure(text="Пропустить", command=self._on_skip)
        self.status_label.configure(
            text=f"Ошибка загрузки: {err_msg[:90]}...",
            text_color=COLOR_DANGER,
        )

    def _cancel_download(self):
        self.stop_event.set()
        self.is_downloading = False
        self.destroy()

    def _on_skip(self):
        self.destroy()

    def _finish(self, model_name: str):
        if self.on_complete:
            self.on_complete(model_name)
        self.destroy()
