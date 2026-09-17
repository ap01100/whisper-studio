"""Modal dialog for Hugging Face Hub token configuration and validation in Whisper Studio.
"""
import webbrowser
import threading
import customtkinter as ctk
from typing import Optional, Callable

from ui.theme import (
    COLOR_BG_PRIMARY,
    COLOR_BG_SECONDARY,
    COLOR_BG_CARD,
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
    CORNER_RADIUS_SM,
    CORNER_RADIUS_MD,
    CORNER_RADIUS_LG,
)
from core.hf_downloader import get_hf_token, set_hf_token, verify_hf_token


class HFTokenDialog(ctk.CTkToplevel):
    """Модальное окно для ввода, проверки и сохранения токена Hugging Face."""

    def __init__(self, master, on_saved: Optional[Callable[[str], None]] = None):
        super().__init__(master)
        self.on_saved = on_saved

        self.title("Hugging Face — Настройки доступа")
        self.geometry("540x400")
        self.resizable(False, False)
        self.configure(fg_color=COLOR_BG_PRIMARY)

        # Модальный режим
        self.transient(master)
        self.grab_set()

        # Центрирование окна
        self._center_window(master)
        self._build_ui()

    def _center_window(self, master):
        self.update_idletasks()
        try:
            x = master.winfo_x() + (master.winfo_width() // 2) - 270
            y = master.winfo_y() + (master.winfo_height() // 2) - 200
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

    def _build_ui(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=24, pady=20)

        # 1. Заголовок
        header_row = ctk.CTkFrame(container, fg_color="transparent")
        header_row.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            header_row,
            text="🔑",
            font=(FONT_TITLE[0], 22),
            text_color=COLOR_TEXT_PRIMARY,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkLabel(
            header_row,
            text="Токен Hugging Face Hub",
            font=FONT_SUBTITLE,
            text_color=COLOR_TEXT_PRIMARY,
        ).pack(side="left")

        # 2. Описание
        desc_text = (
            "Бесплатный токен Hugging Face (Read) снимает ограничения скорости скачивания "
            "и ошибки 429 Too Many Requests при загрузке больших моделей GGUF, "
            "а также активирует Rust-ускоритель hf_transfer (до 100+ МБ/с)."
        )
        ctk.CTkLabel(
            container,
            text=desc_text,
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_SECONDARY,
            wraplength=480,
            justify="left",
        ).pack(fill="x", pady=(0, 12))

        # Ссылка на получение токена
        link_btn = ctk.CTkButton(
            container,
            text="Получить бесплатный токен на huggingface.co ↗",
            font=FONT_CAPTION,
            fg_color="transparent",
            hover_color=COLOR_BG_CARD,
            text_color="#60A5FA",
            anchor="w",
            height=24,
            command=lambda: webbrowser.open("https://huggingface.co/settings/tokens"),
        )
        link_btn.pack(anchor="w", pady=(0, 14))

        # 3. Поле ввода токена
        input_label = ctk.CTkLabel(
            container,
            text="Ваш токен доступа (формат hf_...):",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_PRIMARY,
        )
        input_label.pack(anchor="w", pady=(0, 4))

        entry_row = ctk.CTkFrame(container, fg_color="transparent")
        entry_row.pack(fill="x", pady=(0, 10))

        current_token = get_hf_token() or ""
        self.token_entry = ctk.CTkEntry(
            entry_row,
            placeholder_text="hf_...",
            font=FONT_BODY,
            fg_color=COLOR_BG_INPUT,
            border_color=COLOR_BORDER,
            text_color=COLOR_TEXT_PRIMARY,
            height=36,
            show="•",
        )
        self.token_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        if current_token:
            self.token_entry.insert(0, current_token)

        # Кнопка «Показать / скрыть»
        self.show_token_var = False
        self.toggle_eye_btn = ctk.CTkButton(
            entry_row,
            text="👁",
            font=FONT_CAPTION,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            width=36,
            height=36,
            corner_radius=CORNER_RADIUS_SM,
            command=self._toggle_token_visibility,
        )
        self.toggle_eye_btn.pack(side="right")

        # Статус проверки
        self.status_label = ctk.CTkLabel(
            container,
            text="",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_MUTED,
            anchor="w",
            wraplength=480,
            justify="left",
        )
        self.status_label.pack(fill="x", pady=(0, 16))

        # 4. Панель кнопок действий
        actions_row = ctk.CTkFrame(container, fg_color="transparent")
        actions_row.pack(fill="x", side="bottom")

        # Удалить
        self.delete_btn = ctk.CTkButton(
            actions_row,
            text="Удалить токен",
            font=FONT_CAPTION,
            fg_color="transparent",
            hover_color="#3F1D1D",
            text_color=COLOR_DANGER,
            height=34,
            corner_radius=CORNER_RADIUS_SM,
            command=self._delete_token,
        )
        self.delete_btn.pack(side="left")

        # Закрыть
        close_btn = ctk.CTkButton(
            actions_row,
            text="Закрыть",
            font=FONT_CAPTION,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            height=34,
            corner_radius=CORNER_RADIUS_SM,
            command=self.destroy,
        )
        close_btn.pack(side="right", padx=(8, 0))

        # Сохранить (CTA)
        self.save_btn = ctk.CTkButton(
            actions_row,
            text="Сохранить",
            font=FONT_CAPTION,
            fg_color=COLOR_ACCENT_WHITE,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_ACCENT_TEXT,
            height=34,
            corner_radius=CORNER_RADIUS_SM,
            command=self._save_token,
        )
        self.save_btn.pack(side="right", padx=(8, 0))

        # Проверить
        self.verify_btn = ctk.CTkButton(
            actions_row,
            text="Проверить",
            font=FONT_CAPTION,
            fg_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            height=34,
            corner_radius=CORNER_RADIUS_SM,
            command=self._start_verify,
        )
        self.verify_btn.pack(side="right")

        # Если токен уже есть, проверяем в фоне
        if current_token:
            self._start_verify()

    def _toggle_token_visibility(self):
        self.show_token_var = not self.show_token_var
        if self.show_token_var:
            self.token_entry.configure(show="")
            self.toggle_eye_btn.configure(text="🔒")
        else:
            self.token_entry.configure(show="•")
            self.toggle_eye_btn.configure(text="👁")

    def _start_verify(self):
        token = self.token_entry.get().strip()
        if not token:
            self.status_label.configure(text="Поле токена пустое.", text_color=COLOR_WARNING)
            return

        self.verify_btn.configure(state="disabled", text="Проверка...")
        self.status_label.configure(text="Проверка связи с Hugging Face Hub...", text_color=COLOR_TEXT_PRIMARY)

        def worker():
            res = verify_hf_token(token)
            self.after(0, lambda: self._on_verify_complete(res))

        threading.Thread(target=worker, daemon=True).start()

    def _on_verify_complete(self, res: dict):
        self.verify_btn.configure(state="normal", text="Проверить")
        if res.get("valid"):
            username = res.get("username", "")
            auth_type = res.get("auth_type", "read")
            self.status_label.configure(
                text=f"✓ Успешно! Авторизован как @{username} (права: {auth_type})",
                text_color=COLOR_SUCCESS,
            )
        else:
            err = res.get("error", "Неверный токен.")
            self.status_label.configure(text=f"✗ {err}", text_color=COLOR_DANGER)

    def _save_token(self):
        token = self.token_entry.get().strip()
        set_hf_token(token if token else None)
        if self.on_saved:
            self.on_saved(token)
        self.destroy()

    def _delete_token(self):
        self.token_entry.delete(0, "end")
        set_hf_token(None)
        self.status_label.configure(text="Токен удален. Запросы будут выполняться анонимно.", text_color=COLOR_TEXT_MUTED)
        if self.on_saved:
            self.on_saved("")
