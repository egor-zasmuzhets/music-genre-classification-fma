"""
Telegram-бот для классификации жанра музыки.

Запуск:
    python bot.py
"""

import io
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

import aiohttp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, MessageHandler, CommandHandler,
    CallbackQueryHandler, filters, ContextTypes,
)

from ensemble import GenreClassifier, ClassifierResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BOT_TOKEN = ""

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4"}
SUPPORTED_STR = ", ".join(sorted(SUPPORTED_EXTENSIONS))
DEFAULT_ACTIVE_MODELS = {"cnn", "xgb", "stacker"}

WELCOME_TEXT = (
    "Привет! Я определяю музыкальный жанр с помощью ML.\n\n"
    "Отправь аудиофайл — получишь предсказание от трёх моделей:\n"
    "🧠 CNN, 🌲 XGBoost и 🔗 Ensemble.\n\n"
    f"Поддерживаемые форматы: {SUPPORTED_STR}\n"
    "Максимальный размер: 20 МБ.\n\n""Поиск трека по названию:\n""  /search Nirvana Smells Like Teen Spirit\n""  /find The Beatles Hey Jude\n""  !Radiohead Creep  или просто: поиск"
)

MODEL_LABELS = {
    "cnn":     "🧠 CNN",
    "xgb":     "🌲 XGBoost",
    "stacker": "🔗 Ensemble",
}

classifier: GenreClassifier = None

last_result: Dict[int, Tuple[ClassifierResult, dict]] = {}
active_models: Dict[int, set] = {}
search_results: Dict[int, list] = {}
awaiting_search: Dict[int, bool] = {}



def _get_active(user_id: int) -> set:
    return active_models.get(user_id, DEFAULT_ACTIVE_MODELS.copy())


def _get_pred(result: ClassifierResult, key: str):
    return {"cnn": result.cnn, "xgb": result.xgb, "stacker": result.stacker}[key]



def _format_main(result: ClassifierResult, user_id: int) -> str:
    """Только топ-1 по каждой активной модели."""
    active = _get_active(user_id)
    lines = [f"🎵 {result.filename}\n"]
    for key in ("cnn", "xgb", "stacker"):
        if key not in active:
            continue
        pred = _get_pred(result, key)
        genre, conf = pred.top3[0]
        lines.append(f"{MODEL_LABELS[key]}  →  {genre}  {conf:.0%}")
    body = "\n".join(lines)
    return f"<pre>{body}</pre>"


def _format_details(result: ClassifierResult, timings: dict, user_id: int) -> str:
    """Топ-3 по всем моделям + время инференса. Всегда показывает все три."""
    lines = [f"Подробный отчёт", f"Трек: {result.filename}", ""]

    for key in ("cnn", "xgb", "stacker"):
        pred = _get_pred(result, key)
        t = timings.get(key, 0)
        label = {"cnn": "CNN", "xgb": "XGBoost", "stacker": "Ensemble"}[key]
        lines.append(f"[ {label} ]  {t:.2f}s")
        lines.append(f"  {'-'*20}  {'-'*10}  ----")
        for rank, (genre, conf) in enumerate(pred.top3, 1):
            bar = "█" * int(conf * 10) + "·" * (10 - int(conf * 10))
            lines.append(f"  {genre:<20s}  {bar}  {conf:>4.0%}")
        lines.append("")

    total = sum(timings.get(k, 0) for k in ("cnn", "xgb", "stacker"))
    lines.append(f"Суммарно: {total:.2f}s")
    body = "\n".join(lines)
    return f"<pre>{body}</pre>"



def _build_chart(result: ClassifierResult, user_id: int) -> io.BytesIO:
    """Барчарт вероятностей по всем жанрам для каждой активной модели."""
    active = [k for k in ("cnn", "xgb", "stacker") if k in _get_active(user_id)]
    n = len(active)
    genres = result.genre_names

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    colors_default = "#4e8df5"
    color_top      = "#f5a623"

    for ax, key in zip(axes, active):
        pred  = _get_pred(result, key)
        proba = np.array(pred.all_proba)
        order = np.argsort(proba)
        top_i = int(np.argmax(proba))

        colors = [color_top if i == top_i else colors_default for i in order]
        ax.barh(range(len(genres)), proba[order], color=colors, height=0.65)
        ax.set_yticks(range(len(genres)))
        ax.set_yticklabels([genres[i] for i in order], fontsize=8)
        ax.set_title(MODEL_LABELS[key], fontsize=10, pad=6)
        ax.set_xlim(0, max(proba) * 1.2)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

        for j, idx in enumerate(order):
            ax.text(proba[idx] + 0.004, j, f"{proba[idx]:.1%}", va="center", fontsize=7)

    fig.suptitle(result.filename, fontsize=9, y=1.01)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf



def _main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📊 График",  callback_data="chart"),
            InlineKeyboardButton("🔍 Детали",  callback_data="details"),
        ],
        [
            InlineKeyboardButton("⚙️ Настроить модели", callback_data="settings"),
        ],
    ])


def _settings_keyboard(user_id: int) -> InlineKeyboardMarkup:
    active = _get_active(user_id)
    def btn(key, name):
        mark = "✅" if key in active else "☐"
        return InlineKeyboardButton(f"{mark} {name}", callback_data=f"toggle_{key}")

    return InlineKeyboardMarkup([
        [btn("cnn", "CNN"), btn("xgb", "XGBoost"), btn("stacker", "Ensemble")],
        [InlineKeyboardButton("← Назад", callback_data="back")],
    ])



async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME_TEXT)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME_TEXT)




async def deezer_search(query: str, limit: int = 5) -> list:
    """Ищет треки на Deezer, возвращает список dict с полями title/artist/preview/id."""
    url = "https://api.deezer.com/search"
    params = {"q": query, "limit": limit}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
    tracks = []
    for item in data.get("data", []):
        if item.get("preview"):  # только треки с превью
            tracks.append({
                "id":      item["id"],
                "title":   item["title"],
                "artist":  item["artist"]["name"],
                "preview": item["preview"],  # 30-сек MP3 URL
            })
    return tracks


def _search_keyboard(tracks: list) -> InlineKeyboardMarkup:
    """Кнопка на каждый найденный трек."""
    buttons = [
        [InlineKeyboardButton(
            f"{t['artist']} — {t['title']}"[:50],
            callback_data=f"pick_{i}"
        )]
        for i, t in enumerate(tracks)
    ]
    return InlineKeyboardMarkup(buttons)


async def _do_search(message, user_id: int, query: str) -> None:
    """Выполняет поиск и отправляет результаты."""
    awaiting_search.pop(user_id, None)
    await message.reply_text(f"🔎 Ищу: {query}...")
    try:
        tracks = await deezer_search(query)
    except Exception as e:
        logger.exception("Deezer search failed: %s", e)
        await message.reply_text("❌ Не удалось выполнить поиск. Попробуй позже.")
        return
    if not tracks:
        await message.reply_text("Ничего не нашёл. Попробуй другой запрос.")
        return
    search_results[user_id] = tracks
    await message.reply_text("Выбери трек для классификации:", reply_markup=_search_keyboard(tracks))


async def handle_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команды /search /find — если запрос есть сразу ищем, иначе просим ввести."""
    user_id = update.message.from_user.id
    query   = " ".join(context.args).strip() if context.args else ""
    if query:
        await _do_search(update.message, user_id, query)
    else:
        awaiting_search[user_id] = True
        await update.message.reply_text("Введи название трека или артиста:")

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    user_id = message.from_user.id

    MAX_BYTES = 20 * 1024 * 1024

    if message.audio:
        if message.audio.file_size and message.audio.file_size > MAX_BYTES:
            await message.reply_text("❌ Файл слишком большой. Максимум — 20 МБ.")
            return
        tg_file  = await message.audio.get_file()
        filename = message.audio.file_name or "track.mp3"
    elif message.document:
        filename = message.document.file_name or ""
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            await message.reply_text(
                f"Формат не поддерживается.\nОтправь файл в одном из форматов:\n{SUPPORTED_STR}"
            )
            return
        if message.document.file_size and message.document.file_size > MAX_BYTES:
            await message.reply_text("❌ Файл слишком большой. Максимум — 20 МБ.")
            return
        tg_file = await message.document.get_file()
    elif message.voice:
        if message.voice.file_size and message.voice.file_size > MAX_BYTES:
            await message.reply_text("❌ Файл слишком большой. Максимум — 20 МБ.")
            return
        tg_file  = await message.voice.get_file()
        filename = "voice.ogg"
    else:
        return

    await message.reply_text("⏳ Анализирую...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = Path(tmp_dir) / filename
        await tg_file.download_to_drive(audio_path)

        try:
            audio_path = classifier._to_wav_if_needed(audio_path)

            t0 = time.perf_counter()
            cnn_proba = classifier._predict_cnn(audio_path)
            t_cnn = time.perf_counter() - t0

            t0 = time.perf_counter()
            xgb_proba = classifier._predict_xgb(audio_path)
            t_xgb = time.perf_counter() - t0

            t0 = time.perf_counter()
            import numpy as np
            stack_input   = np.concatenate([cnn_proba, xgb_proba]).reshape(1, -1)
            stacker_proba = classifier._stacker.predict_proba(stack_input)[0]
            t_stacker = time.perf_counter() - t0

            from ensemble import ClassifierResult, ModelPrediction
            top_k = 3
            result = ClassifierResult(
                filename    = Path(filename).name,
                cnn         = classifier._make_prediction(cnn_proba,     top_k),
                xgb         = classifier._make_prediction(xgb_proba,     top_k),
                stacker     = classifier._make_prediction(stacker_proba, top_k),
                genre_names = classifier.genre_names,
            )
            timings = {"cnn": t_cnn, "xgb": t_xgb, "stacker": t_stacker}

            last_result[user_id] = (result, timings)
            await message.reply_text(_format_main(result, user_id), reply_markup=_main_keyboard(), parse_mode="HTML")

        except Exception as e:
            logger.exception("Ошибка классификации: %s", e)
            await message.reply_text("❌ Не удалось обработать файл. Убедись что он не повреждён.")


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query   = update.callback_query
    user_id = query.from_user.id
    await query.answer()

    entry = last_result.get(user_id)

    if query.data == "chart":
        if not entry:
            await query.message.reply_text("Сначала отправь аудиофайл.")
            return
        result, _ = entry
        buf = _build_chart(result, user_id)
        await query.message.reply_photo(photo=buf, caption="📊 Вероятности по жанрам")

    elif query.data == "details":
        if not entry:
            await query.message.reply_text("Сначала отправь аудиофайл.")
            return
        result, timings = entry
        await query.message.reply_text(_format_details(result, timings, user_id), parse_mode="HTML")

    elif query.data == "settings":
        await query.message.reply_text(
            "Выбери модели для основного ответа:",
            reply_markup=_settings_keyboard(user_id),
        )

    elif query.data.startswith("toggle_"):
        key    = query.data.replace("toggle_", "")
        active = _get_active(user_id).copy()
        if key in active:
            if len(active) > 1:
                active.discard(key)
        else:
            active.add(key)
        active_models[user_id] = active
        await query.edit_message_reply_markup(reply_markup=_settings_keyboard(user_id))

    elif query.data.startswith("pick_"):
        idx = int(query.data.replace("pick_", ""))
        tracks = search_results.get(user_id, [])
        if not tracks or idx >= len(tracks):
            await query.message.reply_text("Результаты поиска устарели. Повтори запрос.")
            return
        track = tracks[idx]
        label = f"{track['artist']} — {track['title']}"
        await query.message.reply_text(f"⏳ Скачиваю превью и анализирую...\n🎵 {label}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(track["preview"], timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    preview_bytes = await resp.read()
            import tempfile, numpy as np
            from ensemble import ClassifierResult
            with tempfile.TemporaryDirectory() as tmp_dir:
                mp3_path = Path(tmp_dir) / f"{track['id']}.mp3"
                mp3_path.write_bytes(preview_bytes)
                audio_path = classifier._to_wav_if_needed(mp3_path)
                t0 = time.perf_counter()
                cnn_proba = classifier._predict_cnn(audio_path)
                t_cnn = time.perf_counter() - t0
                t0 = time.perf_counter()
                xgb_proba = classifier._predict_xgb(audio_path)
                t_xgb = time.perf_counter() - t0
                t0 = time.perf_counter()
                stack_input = np.concatenate([cnn_proba, xgb_proba]).reshape(1, -1)
                stacker_proba = classifier._stacker.predict_proba(stack_input)[0]
                t_stacker = time.perf_counter() - t0
                result = ClassifierResult(
                    filename    = label,
                    cnn         = classifier._make_prediction(cnn_proba,     3),
                    xgb         = classifier._make_prediction(xgb_proba,     3),
                    stacker     = classifier._make_prediction(stacker_proba, 3),
                    genre_names = classifier.genre_names,
                )
                timings = {"cnn": t_cnn, "xgb": t_xgb, "stacker": t_stacker}
                last_result[user_id] = (result, timings)
                q = f"{track['artist']} {track['title']}".replace(" ", "+")
                links = (
                    f"🔗 Найти трек:\n"
                    f"  Spotify: https://open.spotify.com/search/{q}\n"
                    f"  Deezer:  https://www.deezer.com/search/{q}"
                )
                await query.message.reply_audio(
                    audio=preview_bytes,
                    title=track["title"],
                    performer=track["artist"],
                    duration=30,
                    caption="🎵 30-сек превью от Deezer",
                )
                await query.message.reply_text(
                    _format_main(result, user_id) + "\n" + links,
                    reply_markup=_main_keyboard(),
                    parse_mode="HTML",
                )
        except Exception as e:
            logger.exception("Preview classify failed: %s", e)
            await query.message.reply_text("❌ Не удалось обработать превью.")

    elif query.data == "back":
        if not entry:
            await query.message.reply_text("Сначала отправь аудиофайл.")
            return
        result, _ = entry
        await query.message.reply_text(
            _format_main(result, user_id),
            reply_markup=_main_keyboard(),
            parse_mode="HTML",
        )


SEARCH_TRIGGERS = {"поиск", "search", "find", "искать", "найти"}
HELP_TEXT = (
    "Вот что я умею:\n\n"
    "🎵 Отправь аудиофайл — определю жанр\n"
    f"   Форматы: {SUPPORTED_STR}\n"
    "   Максимум: 20 МБ\n\n"
    "🔎 Поиск трека:\n"
    "   /search Nirvana Smells Like Teen Spirit\n"
    "   /find The Beatles Hey Jude\n"
    "   !Radiohead Creep\n"
    "   или просто напиши: поиск\n\n"
    "❓ /help — показать это сообщение"
)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Текстовые сообщения: ! префикс, триггер-слова, ожидание поиска."""
    text    = update.message.text.strip()
    user_id = update.message.from_user.id

    if text.startswith("!"):
        query = text[1:].strip()
        if query:
            await _do_search(update.message, user_id, query)
        else:
            awaiting_search[user_id] = True
            await update.message.reply_text("Введи название трека или артиста:")
        return

    if text.lower() in SEARCH_TRIGGERS:
        awaiting_search[user_id] = True
        await update.message.reply_text("Введи название трека или артиста:")
        return

    if awaiting_search.get(user_id):
        await _do_search(update.message, user_id, text)
        return

    await update.message.reply_text(HELP_TEXT)


async def handle_wrong_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT)



def main() -> None:
    global classifier
    logger.info("Загружаю модели...")
    classifier = GenreClassifier.load()
    logger.info("Модели загружены. Запускаю бота...")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(
        filters.AUDIO | filters.Document.ALL | filters.VOICE,
        handle_audio,
    ))
    app.add_handler(CommandHandler("search", handle_search))
    app.add_handler(CommandHandler("find",   handle_search))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(
        filters.ALL & ~filters.COMMAND,
        handle_wrong_message,
    ))

    logger.info("Бот запущен.")
    app.run_polling()


if __name__ == "__main__":
    main()