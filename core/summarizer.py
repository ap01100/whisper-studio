"""Lecture Summarizer module for Whisper Studio.
Implements academic summarization presets, Map-Reduce chunking for long audio,
Markdown processing and plain-text stripping for multi-format export.
"""
import re
import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Generator

from core.llm_engine import LLMEngine, LLMCancelledException


# ==============================================================================
# Очистка разметки Markdown (Экспорт чистого текста по требованию пользователя)
# ==============================================================================

def strip_markdown(md_text: str) -> str:
    """
    Преобразует Markdown-текст в аккуратный структурированный чистый текст без спецсимволов.
    Удаляет заголовки '#', жирный/курсивный шрифт '**', '*', таблицы, цитаты и ссылки.
    """
    if not md_text:
        return ""

    text = md_text

    # 1. Удаление блоков кода (сохраняя сам код)
    text = re.sub(r"```[a-zA-Z0-9_-]*\n?(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # 2. Преобразование заголовков: '# Заголовок' -> 'ЗАГОЛОВОК'
    def header_repl(match):
        level = len(match.group(1))
        title = match.group(2).strip()
        if level <= 2:
            return f"\n\n{title.upper()}\n" + ("=" * min(60, len(title))) + "\n"
        return f"\n\n{title}\n" + ("-" * min(50, len(title))) + "\n"

    text = re.sub(r"^(#{1,6})\s+(.+)$", header_repl, text, flags=re.MULTILINE)

    # 3. Ссылки: [текст](url) -> текст
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # 4. Выделение: **жирный**, *курсив*, __жирный__, _курсив_
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    text = re.sub(r"~~(.*?)~~", r"\1", text)

    # 5. Цитаты: '> цитата' -> '  цитата'
    text = re.sub(r"^>\s?", "  ", text, flags=re.MULTILINE)

    # 6. Маркированные списки: заменяем '*' и '-' на аккуратную точку '• '
    text = re.sub(r"^\s*[\*\-]\s+", "• ", text, flags=re.MULTILINE)

    # 7. Удаление разделителей '---' или '***'
    text = re.sub(r"^[-\*_]{3,}\s*$", "----------------------------------------", text, flags=re.MULTILINE)

    # 8. Сжатие лишних пустых строк (максимум 2 подряд)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ==============================================================================
# Пресеты промптов для конспектирования
# ==============================================================================

PROMPT_PRESETS: Dict[str, Dict[str, Any]] = {
    "academic": {
        "id": "academic",
        "title": "Академический конспект (Рекомендуется)",
        "description": "Полная структурированная лекция: введение, термины, ключевые концепции, примеры и выводы.",
        "system_prompt": (
            "Ты — ведущий преподаватель университета и эксперт по структурированию академических лекций. "
            "Твоя задача — составить подробный, логически связный и исчерпывающий академический конспект на русском языке "
            "по предоставленной стенограмме лекции.\n\n"
            "Требования к структуре конспекта:\n"
            "# Тема лекции\n"
            "## Краткое введение и цели занятия\n"
            "## Ключевые термины и определения (с чёткими формулировками)\n"
            "## Основное содержание по разделам (с подзаголовками, логикой, формулами и примерами лектора)\n"
            "## Важные выводы и заключение\n\n"
            "Используй форматирование Markdown: списки, жирный шрифт для акцентов, блоки формул при наличии. "
            "Не додумывай факты, которых не было в тексте лекции."
        ),
    },
    "summary": {
        "id": "summary",
        "title": "Краткие тезисы (Executive Summary)",
        "description": "10-15 главных тезисов лекции в виде списка для быстрого повторения.",
        "system_prompt": (
            "Ты — эксперт по лаконичному анализу информации. "
            "Выдели из текста лекции ровно 10–15 главных, самых важных тезисов.\n"
            "Формат:\n"
            "# Главные тезисы лекции\n"
            "• Каждый пункт должен быть законченной, содержательной мыслью.\n"
            "• Выделяй ключевые термины жирным шрифтом.\n"
            "• Пиши строго по существу на грамотном русском языке без 'воды'."
        ),
    },
    "exam": {
        "id": "exam",
        "title": "Подготовка к экзамену / Q&A",
        "description": "Список ключевых экзаменационных вопросов по лекции с развернутыми ответами.",
        "system_prompt": (
            "Ты — строгий, но справедливый экзаменатор. "
            "Составь по материалу лекции список из 7–10 ключевых экзаменационных вопросов, "
            "которые преподаватель может задать студенту, и дай на каждый из них развернутый, глубокий ответ.\n\n"
            "Формат:\n"
            "# Вопросы и ответы для подготовки к экзамену\n\n"
            "### Вопрос 1: [Формулировка вопроса]\n"
            "**Ответ:** [Развернутый ответ на основе лекции]\n\n"
            "Используй форматирование Markdown."
        ),
    },
    "custom": {
        "id": "custom",
        "title": "Пользовательский запрос",
        "description": "Свободная инструкция (например: 'Выдели только то, что лектор говорил про нейросети').",
        "system_prompt": (
            "Ты — персональный AI-ассистент по анализу учебных материалов. "
            "Внимательно выполни указанное задание пользователя на основе стенограммы лекции."
        ),
    },
}


# ==============================================================================
# Класс LectureSummarizer
# ==============================================================================

class LectureSummarizer:
    """Оркестратор конспектирования: управляет Direct и Map-Reduce пайплайнами."""

    def __init__(self, llm_engine: Optional[LLMEngine] = None):
        self.engine = llm_engine if llm_engine is not None else LLMEngine()

    def summarize(
        self,
        transcript_text: str,
        model_id: str = "qwen-2.5-7b",
        preset_key: str = "academic",
        custom_instructions: Optional[str] = None,
        context_window: int = 16384,
        temperature: float = 0.6,
        top_p: float = 0.9,
        progress_callback: Optional[Callable[[str, str, float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """
        Генерирует конспект по тексту лекции.
        progress_callback(token_chunk, full_text_so_far, speed_tps, status_str)
        """
        if not transcript_text or not transcript_text.strip():
            raise ValueError("Текст для конспектирования пуст.")

        # 1. Загрузка модели в память (если еще не загружена)
        if not self.engine.is_loaded() or self.engine.loaded_model_id != model_id:
            if progress_callback:
                progress_callback("", "", 0.0, "Загрузка LLM в видеопамять...")
            self.engine.load_model(model_id=model_id, n_ctx=context_window)

        # 2. Определение пресета и системного промпта
        preset = PROMPT_PRESETS.get(preset_key, PROMPT_PRESETS["academic"])
        system_prompt = preset["system_prompt"]
        if preset_key == "custom" and custom_instructions:
            system_prompt += f"\n\nОсобое указание пользователя: {custom_instructions}"

        # 3. Подсчёт токенов и выбор стратегии (Direct vs Map-Reduce)
        # Оставляем 4096 токенов на ответ и 500 на системный промпт
        max_direct_input_tokens = max(4000, context_window - 4600)
        approx_tokens = self.engine.count_tokens(transcript_text)

        if approx_tokens <= max_direct_input_tokens:
            # Direct Context Mode
            return self._run_direct_mode(
                transcript_text=transcript_text,
                system_prompt=system_prompt,
                max_tokens=4096,
                temperature=temperature,
                top_p=top_p,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
        else:
            # Map-Reduce Chunking Mode для очень длинных лекций (>2-3 часов)
            return self._run_map_reduce_mode(
                transcript_text=transcript_text,
                system_prompt=system_prompt,
                chunk_token_size=max_direct_input_tokens // 2,
                temperature=temperature,
                top_p=top_p,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )

    def _run_direct_mode(
        self,
        transcript_text: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        progress_callback: Optional[Callable[[str, str, float, str], None]],
        cancel_event: Optional[threading.Event],
    ) -> str:
        """Прямой прогон полного текста через LLM."""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Вот полная стенограмма лекции:\n\n"
                    f"\"\"\"\n{transcript_text}\n\"\"\"\n\n"
                    "Составь структурированный конспект строго по инструкции выше."
                ),
            },
        ]

        full_response = []
        token_count = 0
        start_time = time.time()

        if progress_callback:
            progress_callback("", "", 0.0, "Генерация конспекта...")

        for chunk in self.engine.stream_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            cancel_event=cancel_event,
        ):
            full_response.append(chunk)
            token_count += 1
            now = time.time()
            elapsed = now - start_time
            speed_tps = (token_count / elapsed) if elapsed > 0 else 0.0

            if progress_callback:
                current_text = "".join(full_response)
                status = f"Генерация: {token_count} токенов ({speed_tps:.1f} токенов/сек)"
                progress_callback(chunk, current_text, speed_tps, status)

        return "".join(full_response)

    def _run_map_reduce_mode(
        self,
        transcript_text: str,
        system_prompt: str,
        chunk_token_size: int,
        temperature: float,
        top_p: float,
        progress_callback: Optional[Callable[[str, str, float, str], None]],
        cancel_event: Optional[threading.Event],
    ) -> str:
        """
        Иерархический режим (Map-Reduce):
        1. Map: нарезка длинной лекции на блоки по ~2500 слов и извлечение ключевых тезисов каждого блока.
        2. Reduce: сведение всех тезисов в итоговый конспект.
        """
        # Разбиваем текст по абзацам
        paragraphs = transcript_text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
            current_chunk.append(p_clean)
            current_len += len(p_clean)
            # При накоплении ~8000 символов (~2500 слов) формируем чанк
            if current_len >= 8000:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        total_chunks = len(chunks)
        mapped_summaries = []

        # Этап 1: MAP
        for idx, chunk in enumerate(chunks, 1):
            if cancel_event and cancel_event.is_set():
                raise LLMCancelledException("Генерация прервана.")

            if progress_callback:
                status = f"Этап 1/2: Анализ фрагмента {idx} из {total_chunks}..."
                progress_callback("", "", 0.0, status)

            map_messages = [
                {
                    "role": "system",
                    "content": (
                        "Ты — эксперт-аналитик. Выдели из данного фрагмента лекции ключевые факты, "
                        "термины, формулы и смысловые пункты. Будь краток и точен."
                    ),
                },
                {"role": "user", "content": f"Фрагмент {idx}:\n\n{chunk}"},
            ]

            chunk_res = []
            for token in self.engine.stream_chat(
                messages=map_messages,
                max_tokens=1500,
                temperature=0.3,
                top_p=top_p,
                cancel_event=cancel_event,
            ):
                chunk_res.append(token)

            mapped_summaries.append(f"### Фрагмент {idx}:\n" + "".join(chunk_res))

        # Этап 2: REDUCE
        if progress_callback:
            progress_callback("", "", 0.0, "Этап 2/2: Формирование финального конспекта...")

        combined_notes = "\n\n".join(mapped_summaries)
        reduce_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Ниже приведены ключевые материалы всех частей длинной лекции:\n\n"
                    f"{combined_notes}\n\n"
                    "Объедини их в единый, монолитный, логичный и исчерпывающий конспект лекции."
                ),
            },
        ]

        final_response = []
        token_count = 0
        start_time = time.time()

        for chunk in self.engine.stream_chat(
            messages=reduce_messages,
            max_tokens=4096,
            temperature=temperature,
            top_p=top_p,
            cancel_event=cancel_event,
        ):
            final_response.append(chunk)
            token_count += 1
            now = time.time()
            elapsed = now - start_time
            speed_tps = (token_count / elapsed) if elapsed > 0 else 0.0

            if progress_callback:
                current_text = "".join(final_response)
                status = f"Финальная сборка: {token_count} токенов ({speed_tps:.1f} токенов/сек)"
                progress_callback(chunk, current_text, speed_tps, status)

        return "".join(final_response)


# ==============================================================================
# Экспорт конспекта (Markdown / Чистый текст без разметки)
# ==============================================================================

def export_summary_md(content: str, output_path: str) -> None:
    """Сохраняет конспект с полной разметкой Markdown (.md)."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


def export_summary_txt(content: str, output_path: str) -> None:
    """Сохраняет конспект как чистый структурированный текст без Markdown (.txt)."""
    plain_text = strip_markdown(content)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(plain_text + "\n")


def load_text_from_file(file_path: str) -> str:
    """
    Загружает текст из файла с поддержкой различных кодировок (UTF-8, CP1251, Latin-1)
    и форматов (.txt, .md, .srt, .vtt, .json).
    При загрузке субтитров (.srt, .vtt) извлекает связный текст речи без служебных таймкодов.
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    raw_data = p.read_bytes()
    text = None
    for enc in ("utf-8", "utf-8-sig", "cp1251", "cp866", "latin-1"):
        try:
            text = raw_data.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        text = raw_data.decode("utf-8", errors="replace")

    ext = p.suffix.lower()

    if ext == ".json":
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                if "segments" in data and isinstance(data["segments"], list):
                    seg_texts = [
                        s.get("text", "").strip()
                        for s in data["segments"]
                        if isinstance(s, dict) and s.get("text", "").strip()
                    ]
                    return "\n\n".join(seg_texts)
                elif "text" in data:
                    return str(data["text"]).strip()
            elif isinstance(data, list):
                texts = [
                    item.get("text", "").strip()
                    for item in data
                    if isinstance(item, dict) and item.get("text")
                ]
                if texts:
                    return "\n\n".join(texts)
        except Exception:
            pass

    elif ext in (".srt", ".vtt"):
        lines = text.splitlines()
        clean_lines = []
        time_pattern = re.compile(
            r"\d{1,2}:\d{2}(?::\d{2})?[,\.]\d{2,3}\s*-->\s*\d{1,2}:\d{2}(?::\d{2})?[,\.]\d{2,3}"
        )
        tag_pattern = re.compile(r"<[^>]+>")

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.isdigit():
                continue
            if line_str.upper() == "WEBVTT" or line_str.startswith("NOTE"):
                continue
            if time_pattern.search(line_str):
                continue
            clean_str = tag_pattern.sub("", line_str).strip()
            if clean_str:
                clean_lines.append(clean_str)

        if clean_lines:
            paragraphs = []
            cur_para = []
            for cl in clean_lines:
                cur_para.append(cl)
                if cl.endswith((".", "!", "?", "…")):
                    paragraphs.append(" ".join(cur_para))
                    cur_para = []
            if cur_para:
                paragraphs.append(" ".join(cur_para))
            return "\n\n".join(paragraphs)

    return text.strip()
