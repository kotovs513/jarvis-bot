"""
Джарвис — умный ассистент-агент в Telegram для соло-фаундера.

Что умеет:
  • принимает текст и голосовые (голос расшифровывает через Whisper);
  • ПОНИМАЕТ намерение: записать дело / показать список / отметить готовым /
    перенести / удалить / просто ответить на вопрос;
  • складывает задачи, напоминания и заметки в базу Notion;
  • из одного сообщения может создать сразу несколько дел;
  • команды словами: «отметь лендинг готовым», «перенеси на завтра», «удали…»;
  • отвечает на вопросы: «какие задачи на сегодня?», «что просрочено?»;
  • напоминания шлёт вовремя и за день до события;
  • каждое утро — сводка на день (/today — по запросу).

Файл рассчитан на запуск «как есть»: заполни .env, установи зависимости, запусти.
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
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
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
# Настройки
# ────────────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
NOTION_VERSION = "2022-06-28"
TIMEZONE = os.environ.get("TIMEZONE", "Asia/Bangkok")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
MORNING_HOUR = int(os.environ.get("MORNING_HOUR", "9"))
CHAT_ID_ENV = os.environ.get("CHAT_ID")

TZ = ZoneInfo(TIMEZONE)
client = OpenAI(api_key=OPENAI_API_KEY)
CHAT_ID_FILE = "chat_id.txt"

NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": NOTION_VERSION,
    "Content-Type": "application/json",
}

logging.basicConfig(format="%(asctime)s  %(levelname)s  %(message)s", level=logging.INFO)
log = logging.getLogger("jarvis")

scheduler = AsyncIOScheduler(timezone=TZ)
application: Application | None = None
_rebuilt = False


# ────────────────────────────────────────────────────────────────────────────
# Память о chat_id
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


async def notion_create(title, kind="note", when=None):
    props = {
        "Название": {"title": [{"text": {"content": title[:2000]}}]},
        "Тип": {"select": {"name": NOTION_TYPE_NAME.get(kind, "Заметка")}},
    }
    if when is not None:
        props["Когда"] = {"date": {"start": when.isoformat()}}
    payload = {"parent": {"database_id": NOTION_DATABASE_ID}, "properties": props}
    async with httpx.AsyncClient(timeout=15) as http:
        r = await http.post("https://api.notion.com/v1/pages",
                            headers=NOTION_HEADERS, json=payload)
        if r.status_code >= 300:
            log.error("Notion create %s: %s", r.status_code, r.text)
            return None
        return r.json().get("id")


async def notion_mark_done(page_id: str) -> bool:
    payload = {"properties": {"Статус": {"select": {"name": "Готово"}}}}
    async with httpx.AsyncClient(timeout=15) as http:
        r = await http.patch(f"https://api.notion.com/v1/pages/{page_id}",
                            headers=NOTION_HEADERS, json=payload)
        return r.status_code < 300


async def notion_set_when(page_id: str, when: datetime) -> bool:
    payload = {"properties": {"Когда": {"date": {"start": when.isoformat()}}}}
    async with httpx.AsyncClient(timeout=15) as http:
        r = await http.patch(f"https://api.notion.com/v1/pages/{page_id}",
                            headers=NOTION_HEADERS, json=payload)
        return r.status_code < 300


async def notion_delete(page_id: str) -> bool:
    payload = {"archived": True}
    async with httpx.AsyncClient(timeout=15) as http:
        r = await http.patch(f"https://api.notion.com/v1/pages/{page_id}",
                            headers=NOTION_HEADERS, json=payload)
        return r.status_code < 300


async def notion_query_open():
    payload = {
        "filter": {"property": "Статус", "select": {"does_not_equal": "Готово"}},
        "page_size": 100,
    }
    async with httpx.AsyncClient(timeout=20) as http:
        r = await http.post(
            f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query",
            headers=NOTION_HEADERS, json=payload)
        if r.status_code >= 300:
            log.error("Notion query %s: %s", r.status_code, r.text)
            return []
        return r.json().get("results", [])


def _title(page):
    try:
        return "".join(p.get("plain_text", "")
                    for p in page["properties"]["Название"]["title"]) or "(без названия)"
    except (KeyError, TypeError):
        return "(без названия)"


def _kind(page):
    try:
        return page["properties"]["Тип"]["select"]["name"]
    except (KeyError, TypeError):
        return ""


def _when(page):
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
# Голос
# ────────────────────────────────────────────────────────────────────────────
async def transcribe_voice(update, context):
    voice = update.message.voice or update.message.audio
    tg_file = await context.bot.get_file(voice.file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        await tg_file.download_to_drive(tmp.name)
        path = tmp.name
    try:
        with open(path, "rb") as audio:
            return client.audio.transcriptions.create(
                model="whisper-1", file=audio).text.strip()
    finally:
        os.remove(path)


# ────────────────────────────────────────────────────────────────────────────
# Слой понимания (роутер намерений)
# ────────────────────────────────────────────────────────────────────────────
ROUTER_PROMPT = """\
Ты — Джарвис, умный ассистент соло-фаундера в Telegram. Тебе приходит сообщение.
Определи, ЧТО хочет человек, и верни СТРОГО JSON без пояснений.

Возможные намерения (intent):
1. "capture" — человек фиксирует одно или несколько дел (задачи, напоминания, идеи).
   Верни "items": массив, каждый элемент:
     {"type":"task"|"reminder"|"note", "text":"аккуратная формулировка", "remind_at":"YYYY-MM-DD HH:MM"|null}
   - reminder — если есть время («завтра в 10», «через час»); посчитай абсолютное время.
   - task — дело без конкретного времени.
   - note — идея/мысль/факт.
   - Если в сообщении НЕСКОЛЬКО дел — сделай несколько элементов.
2. "list" — человек спрашивает, что у него есть/на сегодня/просрочено.
   Верни "scope": "today" | "overdue" | "all".
3. "complete" — просит отметить дело выполненным. Верни "match":"о каком деле речь (ключевые слова)".
4. "reschedule" — просит перенести дело. Верни "match" и "remind_at":"YYYY-MM-DD HH:MM".
5. "delete" — просит удалить дело. Верни "match".
6. "chat" — обычный вопрос, просьба совета, приветствие — то, что не про управление списком.
   Верни "answer":"короткий полезный ответ ассистента".

Правила:
- Вопросы вроде «какие задачи на сегодня?», «что просрочено?», «покажи список» — это "list", НЕ capture.
- «отметь X готовым», «сделал X», «X выполнено» — "complete".
- «перенеси X на завтра», «сдвинь X на 15:00» — "reschedule".
- «удали X», «убери X» — "delete".
- Отвечай только JSON.
"""


# Примеры (few-shot) — резко повышают точность распознавания намерения.
ROUTER_EXAMPLES = [
    ("какие сегодня задачи?", '{"intent":"list","scope":"today"}'),
    ("на сегодня есть напоминания?", '{"intent":"list","scope":"today"}'),
    ("что у меня просрочено?", '{"intent":"list","scope":"overdue"}'),
    ("покажи все дела", '{"intent":"list","scope":"all"}'),
    ("что по задачам", '{"intent":"list","scope":"today"}'),
    ("напомни завтра в 10 позвонить инвестору",
     '{"intent":"capture","items":[{"type":"reminder","text":"Позвонить инвестору","remind_at":"2025-01-02 10:00"}]}'),
    ("сделать лендинг и написать пост в канал",
     '{"intent":"capture","items":[{"type":"task","text":"Сделать лендинг","remind_at":null},'
     '{"type":"task","text":"Написать пост в канал","remind_at":null}]}'),
    ("идея: добавить онбординг по шагам",
     '{"intent":"capture","items":[{"type":"note","text":"Добавить онбординг по шагам","remind_at":null}]}'),
    ("отметь лендинг готовым", '{"intent":"complete","match":"лендинг"}'),
    ("сделал пост", '{"intent":"complete","match":"пост"}'),
    ("перенеси звонок с инвестором на завтра в 15:00",
     '{"intent":"reschedule","match":"звонок с инвестором","remind_at":"2025-01-02 15:00"}'),
    ("удали заметку про онбординг", '{"intent":"delete","match":"онбординг"}'),
    ("спасибо, ты супер", '{"intent":"chat","answer":"Всегда рада помочь! 💪"}'),
    ("как лучше приоритизировать задачи?",
     '{"intent":"chat","answer":"Начни с того, что двигает выручку и имеет дедлайн. '
     'Остальное — во вторую очередь."}'),
]


async def route(text: str) -> dict:
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M (%A)")
    messages = [{"role": "system", "content": ROUTER_PROMPT}]
    for user_ex, assistant_ex in ROUTER_EXAMPLES:
        messages.append({"role": "user", "content": f"Сообщение: {user_ex}"})
        messages.append({"role": "assistant", "content": assistant_ex})
    messages.append({"role": "user", "content": f"Текущее время: {now}.\nСообщение: {text}"})
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:  # noqa: BLE001
        log.exception("Ошибка роутера")
        return {"intent": "capture",
                "items": [{"type": "note", "text": text, "remind_at": None}]}


async def pick_target(match: str, pages: list[dict]) -> dict | None:
    """Выбирает наиболее подходящее дело из списка по описанию пользователя."""
    if not pages:
        return None
    listing = "\n".join(f"{i}. {_title(p)}" for i, p in enumerate(pages))
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content":
                "Верни JSON {\"index\": N} — номер самого подходящего дела из списка "
                "под запрос пользователя, или {\"index\": null} если ничего не подходит."},
                {"role": "user", "content": f"Запрос: {match}\n\nСписок:\n{listing}"},
            ],
        )
        idx = json.loads(resp.choices[0].message.content).get("index")
        if isinstance(idx, int) and 0 <= idx < len(pages):
            return pages[idx]
    except Exception:  # noqa: BLE001
        log.exception("Ошибка выбора дела")
    return None


# ────────────────────────────────────────────────────────────────────────────
# Напоминания
# ────────────────────────────────────────────────────────────────────────────
async def fire_reminder(chat_id, text):
    if application:
        await application.bot.send_message(chat_id=chat_id, text=f"⏰ Напоминание: {text}")


async def fire_pre_reminder(chat_id, text, when_str):
    if application:
        await application.bot.send_message(
            chat_id=chat_id, text=f"📅 Завтра ({when_str}): {text}")


def schedule_reminder(page_id, chat_id, when, text):
    now = datetime.now(TZ)
    if when > now:
        scheduler.add_job(fire_reminder, "date", run_date=when, args=[chat_id, text],
                        id=f"remind:{page_id}", replace_existing=True,
                        misfire_grace_time=3600)
    pre = when - timedelta(days=1)
    if pre > now:
        scheduler.add_job(fire_pre_reminder, "date", run_date=pre,
                        args=[chat_id, text, when.strftime("%d.%m в %H:%M")],
                        id=f"pre:{page_id}", replace_existing=True, misfire_grace_time=3600)


def unschedule_reminder(page_id):
    for jid in (f"remind:{page_id}", f"pre:{page_id}"):
        try:
            scheduler.remove_job(jid)
        except Exception:  # noqa: BLE001
            pass


async def rebuild_reminders(chat_id):
    global _rebuilt
    _rebuilt = True
    n = 0
    for page in await notion_query_open():
        if _kind(page) != "Напоминание":
            continue
        w = _when(page)
        if w and w > datetime.now(TZ):
            schedule_reminder(page["id"], chat_id, w, _title(page))
            n += 1
    if n:
        log.info("Восстановлено напоминаний: %d", n)


# ────────────────────────────────────────────────────────────────────────────
# Списки / сводка
# ────────────────────────────────────────────────────────────────────────────
def build_list(pages, scope, greeting=False):
    now = datetime.now(TZ)
    today = now.date()
    today_rem, overdue, future_rem, tasks = [], [], [], []
    for p in pages:
        k, t, w = _kind(p), _title(p), _when(p)
        if k == "Напоминание" and w:
            if w.date() < today:
                overdue.append((w, f"• {t} (было {w.strftime('%d.%m %H:%M')})"))
            elif w.date() == today:
                today_rem.append((w, f"• {w.strftime('%H:%M')} — {t}"))
            else:
                future_rem.append((w, f"• {w.strftime('%d.%m %H:%M')} — {t}"))
        elif k == "Задача":
            tasks.append(f"• {t}")

    def srt(lst):
        return [x for _, x in sorted(lst)]

    blocks = []
    if scope == "overdue":
        if overdue:
            blocks.append("🔴 Просрочено:\n" + "\n".join(srt(overdue)))
    elif scope == "all":
        if today_rem:
            blocks.append("⏰ Сегодня:\n" + "\n".join(srt(today_rem)))
        if future_rem:
            blocks.append("📅 Дальше:\n" + "\n".join(srt(future_rem)))
        if overdue:
            blocks.append("🔴 Просрочено:\n" + "\n".join(srt(overdue)))
        if tasks:
            blocks.append("✅ Задачи:\n" + "\n".join(tasks))
    else:  # today
        if today_rem:
            blocks.append("⏰ Напоминания на сегодня:\n" + "\n".join(srt(today_rem)))
        if overdue:
            blocks.append("🔴 Просрочено:\n" + "\n".join(srt(overdue)))
        if tasks:
            blocks.append("✅ Открытые задачи:\n" + "\n".join(tasks))

    if not blocks:
        return ("☀️ Доброе утро! На сегодня всё чисто ✨" if greeting
                else "Пусто — ни задач, ни напоминаний ✨")
    head = "☀️ Доброе утро! Вот план на сегодня:\n\n" if greeting else ""
    return head + "\n\n".join(blocks)


async def send_list(chat_id, scope="today", greeting=False):
    if application:
        pages = await notion_query_open()
        await application.bot.send_message(chat_id=chat_id,
                                        text=build_list(pages, scope, greeting))


async def morning_job():
    chat_id = load_chat_id()
    if chat_id is None:
        log.info("Утренняя сводка пропущена: нет chat_id.")
        return
    await send_list(chat_id, "today", greeting=True)


# ────────────────────────────────────────────────────────────────────────────
# Обработка сообщений
# ────────────────────────────────────────────────────────────────────────────
EMOJI = {"task": "✅", "reminder": "⏰", "note": "💡"}
LABEL = {"task": "задача", "reminder": "напоминание", "note": "заметка"}


def done_markup(page_id):
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("✅ Готово", callback_data=f"done:{page_id}")]])


async def do_capture(update, chat_id, items):
    if not items:
        await update.message.reply_text("Не поняла, что записать 🤔")
        return
    for it in items:
        kind = it.get("type", "note")
        text = (it.get("text") or "").strip()
        if not text:
            continue
        when = None
        if kind == "reminder" and it.get("remind_at"):
            try:
                p = datetime.strptime(it["remind_at"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
                if p > datetime.now(TZ):
                    when = p
            except ValueError:
                pass
        page_id = await notion_create(text, kind, when)
        reply = f"{EMOJI.get(kind, '💡')} Записала как «{LABEL.get(kind, 'заметка')}»: {text}"
        if kind == "reminder" and when and page_id:
            schedule_reminder(page_id, chat_id, when, text)
            reply += f"\n🔔 Напомню {when.strftime('%d.%m в %H:%M')}"
            if when - timedelta(days=1) > datetime.now(TZ):
                reply += " (и за день до)"
        markup = done_markup(page_id) if kind in ("task", "reminder") and page_id else None
        await update.message.reply_text(reply, reply_markup=markup)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

    await context.bot.send_chat_action(chat_id, "typing")
    plan = await route(text)
    intent = plan.get("intent", "capture")

    if intent == "list":
        await send_list(chat_id, plan.get("scope", "today"))
        return

    if intent == "chat":
        answer = plan.get("answer") or "Чем помочь?"
        await update.message.reply_text(answer)
        return

    if intent in ("complete", "reschedule", "delete"):
        pages = await notion_query_open()
        target = await pick_target(plan.get("match", text), pages)
        if not target:
            await update.message.reply_text(
                "Не нашла подходящее дело 🤔 Уточни, что именно.")
            return
        title = _title(target)
        pid = target["id"]

        if intent == "complete":
            ok = await notion_mark_done(pid)
            unschedule_reminder(pid)
            await update.message.reply_text(
                f"✅ Отметила готовым: {title}" if ok else "Не вышло обновить Notion 😕")
        elif intent == "delete":
            ok = await notion_delete(pid)
            unschedule_reminder(pid)
            await update.message.reply_text(
                f"🗑 Удалила: {title}" if ok else "Не вышло удалить 😕")
        else:  # reschedule
            when = None
            if plan.get("remind_at"):
                try:
                    when = datetime.strptime(
                        plan["remind_at"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
                except ValueError:
                    pass
            if not when:
                await update.message.reply_text("Не поняла, на когда перенести 🤔")
                return
            ok = await notion_set_when(pid, when)
            if ok:
                schedule_reminder(pid, chat_id, when, title)
                await update.message.reply_text(
                    f"🔄 Перенесла «{title}» на {when.strftime('%d.%m в %H:%M')}")
            else:
                await update.message.reply_text("Не вышло перенести 😕")
        return

    # intent == capture (по умолчанию)
    items = plan.get("items")
    if not items:
        # Подстраховка: похоже на вопрос — покажем список, а не «не поняла».
        low = text.lower()
        if "?" in text or any(low.startswith(w) for w in (
                "какие", "что ", "сколько", "покажи", "есть ли", "какая", "какое")):
            await send_list(chat_id, "today")
            return
    await do_capture(update, chat_id, items)


async def on_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    pid = query.data.split(":", 1)[1]
    if await notion_mark_done(pid):
        unschedule_reminder(pid)
        await query.answer("Готово ✅")
        try:
            await query.edit_message_text(f"✅ Готово: {query.message.text}")
        except Exception:  # noqa: BLE001
            await query.edit_message_reply_markup(reply_markup=None)
    else:
        await query.answer("Не получилось обновить Notion 😕", show_alert=True)


# ────────────────────────────────────────────────────────────────────────────
# Команды
# ────────────────────────────────────────────────────────────────────────────
async def cmd_start(update, context):
    save_chat_id(update.effective_chat.id)
    await update.message.reply_text(
        "Привет! Я твой Джарвис 🤖 — теперь понимаю тебя по-настоящему.\n\n"
        "Можешь:\n"
        "• записывать дела — «напомни завтра в 10 позвонить», «сделать лендинг»;\n"
        "• спрашивать — «какие задачи на сегодня?», «что просрочено?»;\n"
        "• командовать — «отметь лендинг готовым», «перенеси на завтра», «удали…»;\n"
        "• кидать несколько дел одним сообщением;\n"
        "• просто спросить совет.\n\n"
        "Всё складываю в Notion, напоминания пришлю вовремя. Команда /today — план на день.")


async def cmd_today(update, context):
    save_chat_id(update.effective_chat.id)
    await send_list(update.effective_chat.id, "today")


# ────────────────────────────────────────────────────────────────────────────
# Запуск
# ────────────────────────────────────────────────────────────────────────────
async def on_startup(app):
    chat_id = load_chat_id()
    if chat_id is not None:
        await rebuild_reminders(chat_id)


def main():
    global application
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(on_startup).build()
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("today", cmd_today))
    application.add_handler(CallbackQueryHandler(on_done, pattern=r"^done:"))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND | filters.VOICE | filters.AUDIO, handle_message))

    scheduler.add_job(morning_job, "cron", hour=MORNING_HOUR, minute=0,
                    id="morning", replace_existing=True)
    scheduler.start()
    log.info("Джарвис-агент запущен. Пояс: %s, сводка в %02d:00.", TIMEZONE, MORNING_HOUR)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
