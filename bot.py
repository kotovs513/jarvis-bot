"""
Джарвис — личный ассистент в Telegram для соло-фаундера.

Что умеет:
  • принимает текстовые и голосовые сообщения;
  • голос расшифровывает в текст (OpenAI Whisper);
  • понимает, что это — задача, напоминание или заметка (умная сортировка);
  • всё складывает в базу Notion (колонки: Название, Тип, Когда, Статус);
  • под задачей и напоминанием — кнопка «✅ Готово» (ставит статус в Notion);
  • напоминания присылает вовремя, а если есть дата — ещё и за день до события;
  • каждое утро шлёт сводку задач и напоминаний на день (/today — по запросу).

Файл рассчитан на запуск «как есть»: заполни .env, установи зависимости, запусти.
Подробности — в README.md.
"""

from __future__ import annotations

import os
import json
import logging
import tempfile
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
from openai import OpenAI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ────────────────────────────────────────────────────────────────────────────
# Настройки (берутся из переменных окружения / .env)
# ────────────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
NOTION_VERSION = "2022-06-28"
# Часовой пояс, в котором ты говоришь «завтра в 10».
TIMEZONE = os.environ.get("TIMEZONE", "Asia/Bangkok")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
# Во сколько присылать утреннюю сводку (час по местному времени).
MORNING_HOUR = int(os.environ.get("MORNING_HOUR", "9"))
# Необязательно: если задать CHAT_ID, сводка и напоминания работают сразу после
# перезапуска (без него — после первого сообщения боту).
CHAT_ID_ENV = os.environ.get("CHAT_ID")

TZ = ZoneInfo(TIMEZONE)
client = OpenAI(api_key=OPENAI_API_KEY)

# Куда сохраняем ID твоего чата, чтобы слать сводку/напоминания самому.
CHAT_ID_FILE = "chat_id.txt"

NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": NOTION_VERSION,
    "Content-Type": "application/json",
}

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s", level=logging.INFO
)
log = logging.getLogger("jarvis")

scheduler = AsyncIOScheduler(timezone=TZ)
application: Application | None = None
_rebuilt = False  # напоминания восстановлены из Notion в этой сессии?


# ────────────────────────────────────────────────────────────────────────────
# Память о chat_id (чтобы бот мог писать первым)
# ────────────────────────────────────────────────────────────────────────────
def save_chat_id(chat_id: int) -> None:
    try:
        with open(CHAT_ID_FILE, "w") as f:
            f.write(str(chat_id))
    except OSError:
        pass


def load_chat_id() -> int | None:
    if CHAT_ID_ENV:
        return int(CHAT_ID_ENV)
    try:
        with open(CHAT_ID_FILE) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


# ────────────────────────────────────────────────────────────────────────────
# Notion
# ────────────────────────────────────────────────────────────────────────────
NOTION_TYPE_NAME = {"task": "Задача", "reminder": "Напоминание", "note": "Заметка"}


async def notion_create(
    title: str, kind: str = "note", when: datetime | None = None
) -> str | None:
    """Создаёт строку в базе Notion. Возвращает id страницы (или None при ошибке)."""
    properties = {
        "Название": {"title": [{"text": {"content": title[:2000]}}]},
        "Тип": {"select": {"name": NOTION_TYPE_NAME.get(kind, "Заметка")}},
    }
    if when is not None:
        properties["Когда"] = {"date": {"start": when.isoformat()}}

    payload = {"parent": {"database_id": NOTION_DATABASE_ID}, "properties": properties}
    async with httpx.AsyncClient(timeout=15) as http:
        resp = await http.post(
            "https://api.notion.com/v1/pages", headers=NOTION_HEADERS, json=payload
        )
        if resp.status_code >= 300:
            log.error("Notion create %s: %s", resp.status_code, resp.text)
            return None
        return resp.json().get("id")


async def notion_mark_done(page_id: str) -> bool:
    """Ставит Статус = Готово у страницы."""
    payload = {"properties": {"Статус": {"select": {"name": "Готово"}}}}
    async with httpx.AsyncClient(timeout=15) as http:
        resp = await http.patch(
            f"https://api.notion.com/v1/pages/{page_id}",
            headers=NOTION_HEADERS,
            json=payload,
        )
        if resp.status_code >= 300:
            log.error("Notion done %s: %s", resp.status_code, resp.text)
            return False
        return True


async def notion_query_open() -> list[dict]:
    """Возвращает открытые (Статус ≠ Готово) строки базы."""
    payload = {
        "filter": {"property": "Статус", "select": {"does_not_equal": "Готово"}},
        "page_size": 100,
    }
    async with httpx.AsyncClient(timeout=20) as http:
        resp = await http.post(
            f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query",
            headers=NOTION_HEADERS,
            json=payload,
        )
        if resp.status_code >= 300:
            log.error("Notion query %s: %s", resp.status_code, resp.text)
            return []
        return resp.json().get("results", [])


def _page_title(page: dict) -> str:
    try:
        parts = page["properties"]["Название"]["title"]
        return "".join(p.get("plain_text", "") for p in parts) or "(без названия)"
    except (KeyError, TypeError):
        return "(без названия)"


def _page_kind(page: dict) -> str:
    try:
        return page["properties"]["Тип"]["select"]["name"]
    except (KeyError, TypeError):
        return ""


def _page_when(page: dict) -> datetime | None:
    try:
        start = page["properties"]["Когда"]["date"]["start"]
    except (KeyError, TypeError):
        return None
    if not start:
        return None
    try:
        dt = datetime.fromisoformat(start)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt.astimezone(TZ)


# ────────────────────────────────────────────────────────────────────────────
# Расшифровка голоса
# ────────────────────────────────────────────────────────────────────────────
async def transcribe_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    voice = update.message.voice or update.message.audio
    tg_file = await context.bot.get_file(voice.file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        await tg_file.download_to_drive(tmp.name)
        path = tmp.name
    try:
        with open(path, "rb") as audio:
            result = client.audio.transcriptions.create(model="whisper-1", file=audio)
        return result.text.strip()
    finally:
        os.remove(path)


# ────────────────────────────────────────────────────────────────────────────
# Умная сортировка
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
    now_str = datetime.now(TZ).strftime("%Y-%m-%d %H:%M (%A)")
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Текущее время: {now_str}.\nСообщение: {text}"},
        ],
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, KeyError):
        return {"type": "note", "clean_text": text, "remind_at": None}


# ────────────────────────────────────────────────────────────────────────────
# Напоминания
# ────────────────────────────────────────────────────────────────────────────
async def fire_reminder(chat_id: int, text: str) -> None:
    if application is None:
        return
    await application.bot.send_message(chat_id=chat_id, text=f"⏰ Напоминание: {text}")


async def fire_pre_reminder(chat_id: int, text: str, when_str: str) -> None:
    if application is None:
        return
    await application.bot.send_message(
        chat_id=chat_id, text=f"📅 Завтра ({when_str}): {text}"
    )


def schedule_reminder(page_id: str, chat_id: int, when: datetime, text: str) -> None:
    """Ставит напоминание на момент события и (если успеваем) за день до него."""
    now = datetime.now(TZ)
    if when > now:
        scheduler.add_job(
            fire_reminder,
            "date",
            run_date=when,
            args=[chat_id, text],
            id=f"remind:{page_id}",
            replace_existing=True,
            misfire_grace_time=3600,
        )
    pre = when - timedelta(days=1)
    if pre > now:
        scheduler.add_job(
            fire_pre_reminder,
            "date",
            run_date=pre,
            args=[chat_id, text, when.strftime("%d.%m в %H:%M")],
            id=f"pre:{page_id}",
            replace_existing=True,
            misfire_grace_time=3600,
        )


async def rebuild_reminders(chat_id: int) -> None:
    """Восстанавливает будущие напоминания из Notion (переживает перезапуск)."""
    global _rebuilt
    _rebuilt = True
    count = 0
    for page in await notion_query_open():
        if _page_kind(page) != "Напоминание":
            continue
        when = _page_when(page)
        if when and when > datetime.now(TZ):
            schedule_reminder(page["id"], chat_id, when, _page_title(page))
            count += 1
    if count:
        log.info("Восстановлено напоминаний из Notion: %d", count)


# ────────────────────────────────────────────────────────────────────────────
# Утренняя сводка
# ────────────────────────────────────────────────────────────────────────────
def build_summary(pages: list[dict]) -> str:
    now = datetime.now(TZ)
    today = now.date()
    today_reminders, overdue, tasks = [], [], []
    for page in pages:
        kind = _page_kind(page)
        title = _page_title(page)
        when = _page_when(page)
        if kind == "Напоминание" and when:
            if when.date() < today:
                overdue.append(f"• {title} (было {when.strftime('%d.%m %H:%M')})")
            elif when.date() == today:
                today_reminders.append(f"• {when.strftime('%H:%M')} — {title}")
        elif kind == "Задача":
            tasks.append(f"• {title}")

    if not (today_reminders or overdue or tasks):
        return "☀️ Доброе утро! На сегодня всё чисто — задач и напоминаний нет ✨"

    lines = ["☀️ Доброе утро! Вот план на сегодня:"]
    if today_reminders:
        lines.append("\n⏰ Напоминания на сегодня:")
        lines += sorted(today_reminders)
    if overdue:
        lines.append("\n🔴 Просрочено:")
        lines += overdue
    if tasks:
        lines.append("\n✅ Открытые задачи:")
        lines += tasks
    return "\n".join(lines)


async def send_summary(chat_id: int) -> None:
    if application is None:
        return
    pages = await notion_query_open()
    await application.bot.send_message(chat_id=chat_id, text=build_summary(pages))


async def morning_job() -> None:
    chat_id = load_chat_id()
    if chat_id is None:
        log.info("Утренняя сводка пропущена: неизвестен chat_id.")
        return
    await send_summary(chat_id)


# ────────────────────────────────────────────────────────────────────────────
# Обработчик сообщений
# ────────────────────────────────────────────────────────────────────────────
EMOJI = {"task": "✅", "reminder": "⏰", "note": "💡"}
LABEL_RU = {"task": "Задача", "reminder": "Напоминание", "note": "Заметка"}


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    save_chat_id(chat_id)
    if not _rebuilt:
        await rebuild_reminders(chat_id)

    if update.message.voice or update.message.audio:
        await context.bot.send_chat_action(chat_id, "typing")
        try:
            text = await transcribe_voice(update, context)
        except Exception:  # noqa: BLE001
            log.exception("Ошибка расшифровки")
            await update.message.reply_text("Не смог расшифровать голосовое 😕")
            return
    else:
        text = (update.message.text or "").strip()
    if not text:
        return

    try:
        data = await classify(text)
    except Exception:  # noqa: BLE001
        log.exception("Ошибка классификации")
        data = {"type": "note", "clean_text": text, "remind_at": None}

    kind = data.get("type", "note")
    clean = data.get("clean_text", text)
    emoji = EMOJI.get(kind, "💡")

    run_at: datetime | None = None
    if kind == "reminder" and data.get("remind_at"):
        try:
            parsed = datetime.strptime(data["remind_at"], "%Y-%m-%d %H:%M").replace(
                tzinfo=TZ
            )
            if parsed > datetime.now(TZ):
                run_at = parsed
        except ValueError:
            log.warning("Не смог разобрать время: %s", data.get("remind_at"))

    page_id = await notion_create(clean, kind, run_at)

    reply = f"{emoji} Записала как «{LABEL_RU.get(kind, 'Заметка').lower()}»: {clean}"
    if kind == "reminder":
        if run_at is not None and page_id:
            schedule_reminder(page_id, chat_id, run_at, clean)
            reply += f"\n🔔 Напомню {run_at.strftime('%d.%m в %H:%M')}"
            if run_at - timedelta(days=1) > datetime.now(TZ):
                reply += " (и за день до)"
        elif data.get("remind_at"):
            reply += "\n(время уже прошло — напоминание не поставила)"

    # Кнопка «Готово» для задач и напоминаний.
    markup = None
    if kind in ("task", "reminder") and page_id:
        markup = InlineKeyboardMarkup(
            [[InlineKeyboardButton("✅ Готово", callback_data=f"done:{page_id}")]]
        )
    await update.message.reply_text(reply, reply_markup=markup)


async def on_done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    page_id = query.data.split(":", 1)[1]
    ok = await notion_mark_done(page_id)
    if ok:
        await query.answer("Отметила как готово ✅")
        try:
            await query.edit_message_text(f"✅ Готово: {query.message.text}")
        except Exception:  # noqa: BLE001
            await query.edit_message_reply_markup(reply_markup=None)
    else:
        await query.answer("Не получилось обновить Notion 😕", show_alert=True)


# ────────────────────────────────────────────────────────────────────────────
# Команды
# ────────────────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    save_chat_id(update.effective_chat.id)
    await update.message.reply_text(
        "Привет! Я твой Джарвис 🤖\n\n"
        "Пиши мне или шли голосовые — я сам пойму, что это:\n"
        "✅ задача, ⏰ напоминание или 💡 заметка, и всё сложу в Notion.\n\n"
        "• напоминания пришлю вовремя, а если есть дата — ещё и за день до;\n"
        "• под задачей будет кнопка «Готово»;\n"
        "• каждое утро пришлю план на день (или команда /today).\n\n"
        "Примеры:\n"
        "• «напомни завтра в 10 позвонить инвестору»\n"
        "• «сделать лендинг к пятнице»\n"
        "• «идея: добавить онбординг по шагам»"
    )


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    save_chat_id(update.effective_chat.id)
    await send_summary(update.effective_chat.id)


# ────────────────────────────────────────────────────────────────────────────
# Запуск
# ────────────────────────────────────────────────────────────────────────────
async def on_startup(app: Application) -> None:
    chat_id = load_chat_id()
    if chat_id is not None:
        await rebuild_reminders(chat_id)


def main() -> None:
    global application
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(on_startup).build()

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("today", cmd_today))
    application.add_handler(CallbackQueryHandler(on_done, pattern=r"^done:"))
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND | filters.VOICE | filters.AUDIO,
            handle_message,
        )
    )

    scheduler.add_job(
        morning_job,
        "cron",
        hour=MORNING_HOUR,
        minute=0,
        id="morning",
        replace_existing=True,
    )
    scheduler.start()
    log.info("Джарвис запущен. Пояс: %s, сводка в %02d:00.", TIMEZONE, MORNING_HOUR)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
