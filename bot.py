"""
Джарвис — личный ассистент в Telegram для соло-фаундера.

Что умеет:
  • принимает текстовые и голосовые сообщения;
  • голос расшифровывает в текст (OpenAI Whisper);
  • понимает, что это — задача, напоминание или заметка (умная сортировка);
  • всё складывает в базу Notion (колонки: Название, Тип, Когда);
  • если это напоминание — сам пингует тебя в Telegram в нужное время.

Файл рассчитан на запуск «как есть»: заполни .env, установи зависимости, запусти.
Подробности — в README.md.
"""

from __future__ import annotations

import os
import json
import logging
import tempfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx
from openai import OpenAI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ────────────────────────────────────────────────────────────────────────────
# Настройки (берутся из переменных окружения / .env)
# ────────────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# Ключ интеграции Notion и ID базы, куда складываем задачи.
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
NOTION_VERSION = "2022-06-28"
# Часовой пояс, в котором ты говоришь «завтра в 10». По умолчанию — Москва.
TIMEZONE = os.environ.get("TIMEZONE", "Europe/Moscow")
# Модель для сортировки. gpt-4o-mini — дёшево и достаточно умно.
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

TZ = ZoneInfo(TIMEZONE)
client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s", level=logging.INFO
)
log = logging.getLogger("jarvis")

# ────────────────────────────────────────────────────────────────────────────
# Планировщик напоминаний. Хранит задачи в SQLite, поэтому переживает перезапуск.
# ────────────────────────────────────────────────────────────────────────────
scheduler = AsyncIOScheduler(
    jobstores={"default": SQLAlchemyJobStore(url="sqlite:///reminders.db")},
    timezone=TZ,
)

# Глобальная ссылка на приложение — нужна, чтобы напоминание могло отправить
# сообщение из планировщика.
application: Application | None = None


# ────────────────────────────────────────────────────────────────────────────
# Notion — создаём страницу (строку) в базе задач
# ────────────────────────────────────────────────────────────────────────────
NOTION_TYPE_NAME = {"task": "Задача", "reminder": "Напоминание", "note": "Заметка"}


async def save_to_notion(
    title: str, kind: str = "note", when: datetime | None = None
) -> None:
    """Создаёт новую строку в базе Notion.

    Ожидаемые колонки базы:
      • «Название» — тип Title
      • «Тип»      — тип Select  (Задача / Напоминание / Заметка)
      • «Когда»    — тип Date    (заполняется только для напоминаний)
    """
    properties = {
        "Название": {"title": [{"text": {"content": title[:2000]}}]},
        "Тип": {"select": {"name": NOTION_TYPE_NAME.get(kind, "Заметка")}},
    }
    if when is not None:
        properties["Когда"] = {"date": {"start": when.isoformat()}}

    payload = {"parent": {"database_id": NOTION_DATABASE_ID}, "properties": properties}
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=15) as http:
        resp = await http.post(
            "https://api.notion.com/v1/pages", headers=headers, json=payload
        )
        if resp.status_code >= 300:
            log.error("Notion ответил %s: %s", resp.status_code, resp.text)


# ────────────────────────────────────────────────────────────────────────────
# Расшифровка голоса
# ────────────────────────────────────────────────────────────────────────────
async def transcribe_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Скачивает голосовое, прогоняет через Whisper, возвращает текст."""
    voice = update.message.voice or update.message.audio
    tg_file = await context.bot.get_file(voice.file_id)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        await tg_file.download_to_drive(tmp.name)
        path = tmp.name

    try:
        with open(path, "rb") as audio:
            result = client.audio.transcriptions.create(
                model="whisper-1", file=audio
            )
        return result.text.strip()
    finally:
        os.remove(path)


# ────────────────────────────────────────────────────────────────────────────
# Умная сортировка: задача / напоминание / заметка
# ────────────────────────────────────────────────────────────────────────────
CLASSIFY_SYSTEM_PROMPT = """\
Ты — ассистент соло-фаундера. Тебе приходит короткое сообщение (задача, \
напоминание или мысль). Определи тип и верни СТРОГО JSON без пояснений.

Формат ответа:
{
  "type": "task" | "reminder" | "note",
  "clean_text": "аккуратно переформулированный текст, без слов вроде 'напомни'",
  "remind_at": "YYYY-MM-DD HH:MM" или null
}

Правила:
- "reminder" — если есть указание времени ("завтра в 10", "через час", "в пятницу").
  В remind_at запиши конкретное дату-время в 24ч формате.
- "task" — дело, которое надо сделать, но без конкретного времени.
- "note" — идея, мысль, факт «на будущее».
- Если время относительное ("через 2 часа") — посчитай от текущего момента.
- Отвечай только JSON.
"""


async def classify(text: str) -> dict:
    """Спрашивает LLM: это задача, напоминание или заметка + время напоминания."""
    now_str = datetime.now(TZ).strftime("%Y-%m-%d %H:%M (%A)")
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Текущее время: {now_str}.\nСообщение: {text}",
            },
        ],
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, KeyError):
        # На всякий случай: если модель ответила криво — считаем это заметкой.
        return {"type": "note", "clean_text": text, "remind_at": None}


# ────────────────────────────────────────────────────────────────────────────
# Напоминание срабатывает — эта функция вызывается планировщиком
# ────────────────────────────────────────────────────────────────────────────
async def fire_reminder(chat_id: int, text: str) -> None:
    if application is None:
        return
    await application.bot.send_message(
        chat_id=chat_id, text=f"⏰ Напоминание: {text}"
    )


# ────────────────────────────────────────────────────────────────────────────
# Основной обработчик сообщений
# ────────────────────────────────────────────────────────────────────────────
EMOJI = {"task": "✅", "reminder": "⏰", "note": "💡"}
LABEL_RU = {"task": "Задача", "reminder": "Напоминание", "note": "Заметка"}


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id

    # 1. Получаем текст — либо напрямую, либо из голосового.
    if update.message.voice or update.message.audio:
        await context.bot.send_chat_action(chat_id, "typing")
        try:
            text = await transcribe_voice(update, context)
        except Exception as e:  # noqa: BLE001
            log.exception("Ошибка расшифровки")
            await update.message.reply_text("Не смог расшифровать голосовое 😕")
            return
    else:
        text = (update.message.text or "").strip()

    if not text:
        return

    # 2. Сортируем.
    try:
        data = await classify(text)
    except Exception:  # noqa: BLE001
        log.exception("Ошибка классификации")
        data = {"type": "note", "clean_text": text, "remind_at": None}

    kind = data.get("type", "note")
    clean = data.get("clean_text", text)
    emoji = EMOJI.get(kind, "💡")

    # 3. Разбираем время напоминания (если оно есть).
    run_at: datetime | None = None
    if kind == "reminder" and data.get("remind_at"):
        try:
            parsed = datetime.strptime(
                data["remind_at"], "%Y-%m-%d %H:%M"
            ).replace(tzinfo=TZ)
            if parsed > datetime.now(TZ):
                run_at = parsed
        except ValueError:
            log.warning("Не смог разобрать время: %s", data.get("remind_at"))

    # 4. Сохраняем в Notion.
    await save_to_notion(clean, kind, run_at)

    # 5. Если это напоминание с будущим временем — ставим в планировщик.
    reply = f"{emoji} Записала как «{LABEL_RU.get(kind, 'Заметка').lower()}»: {clean}"
    if kind == "reminder":
        if run_at is not None:
            scheduler.add_job(
                fire_reminder,
                "date",
                run_date=run_at,
                args=[chat_id, clean],
                misfire_grace_time=3600,
            )
            reply += f"\n🔔 Напомню {run_at.strftime('%d.%m в %H:%M')}"
        elif data.get("remind_at"):
            reply += "\n(время уже прошло — напоминание не поставила)"

    await update.message.reply_text(reply)


# ────────────────────────────────────────────────────────────────────────────
# Команды
# ────────────────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я твой Джарвис 🤖\n\n"
        "Просто пиши мне или шли голосовые — я сам пойму, что это:\n"
        "✅ задача, ⏰ напоминание или 💡 заметка,\n"
        "и всё сложу в Notion. Напоминания пришлю вовремя сюда же.\n\n"
        "Примеры:\n"
        "• «напомни завтра в 10 позвонить инвестору»\n"
        "• «сделать лендинг к пятнице»\n"
        "• «идея: добавить онбординг по шагам»"
    )


# ────────────────────────────────────────────────────────────────────────────
# Запуск
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    global application
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND
            | filters.VOICE
            | filters.AUDIO,
            handle_message,
        )
    )

    scheduler.start()
    log.info("Джарвис запущен. Часовой пояс: %s", TIMEZONE)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
