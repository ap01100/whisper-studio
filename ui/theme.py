"""Design tokens and styling for Whisper Studio (Monochrome / Zinc theme).
"""
import customtkinter as ctk

# Цветовая палитра: Deep Dark + Crisp White Accent (Vercel / Linear / Zinc style)
COLOR_BG_PRIMARY = "#0C0D0E"      # Глубокий матовый фон окна
COLOR_BG_SECONDARY = "#141518"    # Вторичный фон панелей
COLOR_BG_CARD = "#181A1F"         # Фон карточек
COLOR_BG_CARD_HOVER = "#21232B"   # Фон карточек при наведении
COLOR_BG_INPUT = "#131417"        # Фон полей ввода

COLOR_BORDER = "#272A33"          # Тонкая рамка (1px)
COLOR_BORDER_FOCUS = "#52525B"    # Рамка в фокусе

# Акцентные цвета: чистый белый для главных кнопок (CTA)
COLOR_ACCENT_WHITE = "#FFFFFF"
COLOR_ACCENT_HOVER = "#E4E4E7"
COLOR_ACCENT_TEXT = "#090A0C"     # Контрастный тёмный текст на белой кнопке

# Вторичные кнопки
COLOR_BTN_SECONDARY = "#22252D"
COLOR_BTN_SECONDARY_HOVER = "#2D313C"

# Индикаторы статуса
COLOR_SUCCESS = "#10B981"         # Изумрудно-зеленый (CUDA Ready)
COLOR_WARNING = "#F59E0B"         # Янтарный (CPU fallback)
COLOR_DANGER = "#EF4444"          # Красный (Остановить / Ошибка)

# Типографика
COLOR_TEXT_PRIMARY = "#F4F4F5"
COLOR_TEXT_SECONDARY = "#A1A1AA"
COLOR_TEXT_MUTED = "#71717A"

FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"

FONT_TITLE = (FONT_FAMILY, 16, "bold")
FONT_SUBTITLE = (FONT_FAMILY, 13, "bold")
FONT_BODY = (FONT_FAMILY, 12)
FONT_BODY_BOLD = (FONT_FAMILY, 12, "bold")
FONT_CAPTION = (FONT_FAMILY, 11)
FONT_CODE = (FONT_MONO, 11)

CORNER_RADIUS_SM = 6
CORNER_RADIUS_MD = 8
CORNER_RADIUS_LG = 10


def apply_global_theme():
    """Применяет глобальные настройки темы CustomTkinter."""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
