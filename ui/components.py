"""Reusable UI components for Whisper Studio.
"""
import os
import sys
from pathlib import Path
from typing import Optional, Callable

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog

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
)
from core.audio_utils import get_media_info, SUPPORTED_EXTENSIONS


class GPUBadge(ctk.CTkFrame):
    """Бейдж статуса видеокарты и готовности CUDA в Header."""

    def __init__(self, master, gpu_info: dict, **kwargs):
        super().__init__(
            master,
            fg_color=COLOR_BG_CARD,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=CORNER_RADIUS_MD,
            **kwargs,
        )
        self.gpu_info = gpu_info

        # Внутренний контейнер
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(padx=10, pady=5)

        # Индикатор (точка)
        dot_color = gpu_info.get("status_color", COLOR_SUCCESS)
        self.dot_label = ctk.CTkLabel(
            inner,
            text="●",
            font=(FONT_TITLE[0], 12),
            text_color=dot_color,
        )
        self.dot_label.pack(side="left", padx=(0, 6))

        # Текст
        text = gpu_info.get("badge_text", "Определение оборудования...")
        self.text_label = ctk.CTkLabel(
            inner,
            text=text,
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_PRIMARY,
        )
        self.text_label.pack(side="left")

    def update_info(self, gpu_info: dict):
        self.gpu_info = gpu_info
        dot_color = gpu_info.get("status_color", COLOR_SUCCESS)
        self.dot_label.configure(text_color=dot_color)
        self.text_label.configure(text=gpu_info.get("badge_text", ""))


class DropZone(ctk.CTkFrame):
    """
    Интерактивная зона Drag & Drop с поддержкой клика для выбора файла.
    """

    def __init__(self, master, on_file_selected: Callable[[str], None], **kwargs):
        super().__init__(
            master,
            fg_color=COLOR_BG_CARD,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=CORNER_RADIUS_LG,
            **kwargs,
        )
        self.on_file_selected = on_file_selected
        self.selected_file_path: Optional[str] = None
        self._is_hovered = False

        self._build_ui()
        self._setup_bindings()
        self._setup_windnd()

    def _build_ui(self):
        # Очистка
        for w in self.winfo_children():
            w.destroy()

        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(expand=True, fill="both", padx=20, pady=20)

        # Иконка
        self.icon_label = ctk.CTkLabel(
            self.container,
            text="📁",
            font=(FONT_TITLE[0], 28),
            text_color=COLOR_TEXT_PRIMARY,
        )
        self.icon_label.pack(pady=(4, 6))

        # Основной текст
        self.title_label = ctk.CTkLabel(
            self.container,
            text="Перетащите аудио- или видеофайл сюда",
            font=FONT_SUBTITLE,
            text_color=COLOR_TEXT_PRIMARY,
        )
        self.title_label.pack(pady=(0, 2))

        # Подсказка
        self.subtitle_label = ctk.CTkLabel(
            self.container,
            text="или нажмите в любое место карточки для выбора\n(MP3, WAV, M4A, FLAC, OGG, MP4, MKV)",
            font=FONT_CAPTION,
            text_color=COLOR_TEXT_MUTED,
            justify="center",
        )
        self.subtitle_label.pack()

    def _is_widget_inside(self, target_widget):
        curr = target_widget
        while curr is not None:
            if curr == self or curr == getattr(self, "_canvas", None):
                return True
            curr = getattr(curr, "master", None)
        return False

    def _on_enter(self, event=None):
        if not self._is_hovered:
            self._is_hovered = True
            self.configure(fg_color=COLOR_BG_CARD_HOVER, border_color=COLOR_BORDER_FOCUS)

    def _on_leave(self, event=None):
        if event is not None:
            try:
                containing = self.winfo_containing(event.x_root, event.y_root)
                if self._is_widget_inside(containing):
                    return
            except Exception:
                pass
        if self._is_hovered:
            self._is_hovered = False
            self.configure(fg_color=COLOR_BG_CARD, border_color=COLOR_BORDER)

    def _setup_bindings(self):
        """Привязывает клик и наведение по всей зоне к открытию диалога."""
        interactive_widgets = [
            self,
            self.container,
            self.icon_label,
            self.title_label,
            self.subtitle_label,
        ]
        for w in interactive_widgets:
            w.bind("<Button-1>", lambda e: self.open_file_dialog())
            w.bind("<Enter>", self._on_enter)
            w.bind("<Leave>", self._on_leave)

    def _setup_windnd(self):
        """Подключает аппаратный Drag & Drop для Windows через windnd."""
        try:
            import windnd

            def on_drop(files):
                if not files:
                    return
                f = files[0]
                if isinstance(f, bytes):
                    # Windows paths in windnd can be cp1251 or utf-8
                    try:
                        path_str = f.decode("utf-8")
                    except UnicodeDecodeError:
                        path_str = f.decode("cp1251", errors="ignore")
                else:
                    path_str = str(f)

                self.set_file(path_str)

            windnd.hook_dropnodes(self, on_drop)
        except Exception:
            pass  # Если windnd недоступен, работает клик

    def open_file_dialog(self):
        filetypes = [
            ("Медиафайлы (Аудио/Видео)", "*.mp3 *.wav *.m4a *.flac *.ogg *.aac *.wma *.mp4 *.mkv *.mov *.webm"),
            ("Все файлы", "*.*"),
        ]
        chosen = filedialog.askopenfilename(
            title="Выберите аудио- или видеозапись",
            filetypes=filetypes,
            initialdir=str(Path(".").resolve()),
        )
        if chosen:
            self.set_file(chosen)

    def set_file(self, file_path: str):
        path = Path(file_path)
        if not path.exists():
            return

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            self.subtitle_label.configure(
                text=f"Неподдерживаемый формат: {path.suffix}\nВыберите поддерживаемое аудио или видео",
                text_color=COLOR_DANGER,
            )
            return

        self.selected_file_path = str(path.resolve())

        try:
            media_info = get_media_info(self.selected_file_path)
            size_mb = media_info["size_mb"]
            dur_str = media_info["duration_str"]
            dur_info = f" • Длительность: {dur_str}" if media_info["duration_sec"] > 0 else ""

            self.icon_label.configure(text="🎵", text_color=COLOR_SUCCESS)
            self.title_label.configure(
                text=path.name,
                font=FONT_SUBTITLE,
                text_color=COLOR_TEXT_PRIMARY,
            )
            self.subtitle_label.configure(
                text=f"Размер: {size_mb} MB{dur_info}\n(Нажмите для выбора другого файла)",
                text_color=COLOR_TEXT_SECONDARY,
            )
        except Exception:
            self.title_label.configure(text=path.name)
            self.subtitle_label.configure(text="(Нажмите для выбора другого файла)")

        if self.on_file_selected:
            self.on_file_selected(self.selected_file_path)

    def reset(self):
        self.selected_file_path = None
        self._build_ui()
        self._setup_bindings()


class CollapsibleSection(ctk.CTkFrame):
    """Сворачиваемая панель аккордеона с деликатной рамкой."""

    def __init__(self, master, title: str, **kwargs):
        super().__init__(
            master,
            fg_color=COLOR_BG_CARD,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=CORNER_RADIUS_MD,
            **kwargs,
        )
        self.title_text = title
        self.is_expanded = False

        # Заголовок с кнопкой-стрелкой
        self.header_btn = ctk.CTkButton(
            self,
            text=f"▶  {self.title_text}",
            anchor="w",
            fg_color="transparent",
            hover_color=COLOR_BG_CARD_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            font=FONT_BODY_BOLD,
            height=34,
            corner_radius=CORNER_RADIUS_SM,
            command=self.toggle,
        )
        self.header_btn.pack(fill="x", padx=6, pady=4)

        # Контентное тело
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")

    def toggle(self):
        if self.is_expanded:
            self.content_frame.pack_forget()
            self.header_btn.configure(text=f"▶  {self.title_text}")
            self.is_expanded = False
        else:
            self.content_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))
            self.header_btn.configure(text=f"▼  {self.title_text}")
            self.is_expanded = True


class ToastMessage(ctk.CTkFrame):
    """Всплывающее уведомление (Toast) с автоскрытием."""

    def __init__(self, master, message: str = "✓ Скопировано в буфер обмена!", **kwargs):
        super().__init__(
            master,
            fg_color=COLOR_ACCENT_WHITE,
            corner_radius=CORNER_RADIUS_MD,
            border_width=0,
            **kwargs,
        )
        label = ctk.CTkLabel(
            self,
            text=message,
            font=FONT_BODY_BOLD,
            text_color=COLOR_ACCENT_TEXT,
        )
        label.pack(padx=16, pady=8)

    def show(self, delay_ms: int = 1800):
        self.place(relx=0.5, rely=0.92, anchor="center")
        self.after(delay_ms, self.hide)

    def hide(self):
        self.place_forget()


def bind_text_shortcuts(widget, allow_edit: bool = True):
    """
    Добавляет полную поддержку горячих клавиш (Ctrl+V, Ctrl+C, Ctrl+A, Ctrl+X)
    с учётом русской раскладки клавиатуры и контекстное меню ПКМ.
    """
    inner = getattr(widget, "_entry", None) or getattr(widget, "_textbox", None)
    if not inner:
        return

    def do_paste(event=None):
        if not allow_edit:
            return "break"
        try:
            text = widget.clipboard_get()
            if text:
                if hasattr(widget, "_entry"):
                    widget.insert("insert", text)
                elif hasattr(widget, "_textbox"):
                    widget.insert("insert", text)
                # Вызываем событие изменения, если привязано
                widget.event_generate("<<Modified>>")
        except Exception:
            pass
        return "break"

    def do_copy(event=None):
        try:
            if hasattr(widget, "_textbox"):
                try:
                    sel = widget.get("sel.first", "sel.last")
                    if sel:
                        widget.clipboard_clear()
                        widget.clipboard_append(sel)
                except Exception:
                    pass
            elif hasattr(widget, "_entry"):
                try:
                    sel = widget.selection_get()
                    if sel:
                        widget.clipboard_clear()
                        widget.clipboard_append(sel)
                except Exception:
                    pass
        except Exception:
            pass
        return "break"

    def do_cut(event=None):
        if not allow_edit:
            return "break"
        do_copy()
        try:
            if hasattr(widget, "_textbox"):
                widget.delete("sel.first", "sel.last")
            elif hasattr(widget, "_entry"):
                widget.delete("sel.first", "sel.last")
            widget.event_generate("<<Modified>>")
        except Exception:
            pass
        return "break"

    def do_select_all(event=None):
        try:
            if hasattr(widget, "_textbox"):
                widget.tag_add("sel", "1.0", "end")
            elif hasattr(widget, "_entry"):
                widget.select_range(0, "end")
        except Exception:
            pass
        return "break"

    def on_key(event):
        is_ctrl = bool(event.state & 4)
        if not is_ctrl:
            return None

        # KeyCode 86 = V, 65 = A, 67 = C, 88 = X
        kc = event.keycode
        ks = str(event.keysym).lower()

        if kc == 86 or ks in ("v", "cyrillic_em", "ntilde"):
            return do_paste()
        elif kc == 65 or ks in ("a", "cyrillic_ef"):
            return do_select_all()
        elif kc == 67 or ks in ("c", "cyrillic_es"):
            return do_copy()
        elif kc == 88 or ks in ("x", "cyrillic_che"):
            return do_cut()

    inner.bind("<Key>", on_key)

    # Контекстное меню ПКМ
    menu = tk.Menu(
        inner,
        tearoff=0,
        bg="#181A1F",
        fg="#F3F4F6",
        activebackground="#272A33",
        activeforeground="#FFFFFF",
        bd=1,
        relief="solid",
    )
    if allow_edit:
        menu.add_command(label="Вставить (Ctrl+V)", command=do_paste)
        menu.add_command(label="Вырезать (Ctrl+X)", command=do_cut)
    menu.add_command(label="Копировать (Ctrl+C)", command=do_copy)
    menu.add_separator()
    menu.add_command(label="Выделить всё (Ctrl+A)", command=do_select_all)
    if allow_edit and hasattr(widget, "_textbox"):
        def do_clear_all():
            widget.delete("1.0", "end")
            widget.event_generate("<<Modified>>")
        menu.add_command(label="Очистить всё", command=do_clear_all)

    def show_popup(event):
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    inner.bind("<Button-3>", show_popup)


def bind_paste_support(widget):
    """Обратная совместимость: привязывает горячие клавиши и контекстное меню."""
    bind_text_shortcuts(widget, allow_edit=True)

