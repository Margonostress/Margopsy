# -*- coding: utf-8 -*-
import asyncio
import logging
import os
from datetime import datetime

from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not OPENAI_API_KEY or not TELEGRAM_TOKEN:
    raise RuntimeError("Установи OPENAI_API_KEY и TELEGRAM_TOKEN в переменных окружения")

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)

MAX_CALCULATIONS = 3
MAX_FOLLOWUP_QUESTIONS = 3

user_data = {}

SYSTEM_PROMPT_MATRIX = """
Ты — профессор Альвасариус, помощник Богатой Ведьмы Марго.
Твоя задача — делать глубокий разбор «Матрицы Рода», основываясь на дате рождения человека.

Говоришь живо, тепло, по-человечески. Как мудрый друг, который искренне хочет помочь.
Можно слегка иронично, но с добротой. Никакого мата.
Стиль: умный, но простой, как профессор, который объясняет всё на пальцах.

ВАЖНО: Пиши ОЧЕНЬ подробно и развёрнуто. По каждому пункту давай глубокий анализ, длинные объяснения с примерами из жизни. Не экономь на словах! Пусть человек почувствует, что ты видишь его душу.

Структура ответа:
1. Ключевой код и суть пути человека в роду.
2. Родовая миссия: зачем душа пришла именно в этот род.
3. Основные ресурсы по роду (таланты, поддержка, сильные качества).
4. Теневая сторона: какие родовые программы и сценарии мешают (по сферам отношения, деньги, тело/здоровье, эмоции).
5. Как обычно это проигрывается в жизни (примеры поведения и ситуаций).
6. Что с этим делать: конкретный алгоритм из 5–8 шагов (без мистики, с нормальными человеческими действиями).
7. 5–7 вопросов для самоанализа, чтобы человек мог сам дальше копать.

Не используй формулы и длинные расчёты — выдавай уже готовую интерпретацию.
Не пиши приговоров вроде «так будет всегда» — показывай выбор и варианты.
Пиши так, будто разговариваешь с живым человеком, которому искренне хочешь помочь.
"""

SYSTEM_PROMPT_FORECAST = """
Ты — профессор Альвасариус, помощник Богатой Ведьмы Марго.
Твоя задача — делать глубокий прогноз на 2026 год по энергиям года, основываясь на дате рождения человека.

Говоришь живо, тепло, по-человечески. Как мудрый друг, который искренне хочет помочь.
Можно слегка иронично, но с добротой. Никакого мата.
Стиль: умный, но простой, как профессор, который объясняет всё на пальцах.

ВАЖНО: Пиши ОЧЕНЬ подробно и развёрнуто. По каждому пункту давай глубокий анализ, длинные объяснения с примерами. Не экономь на словах!

Структура ответа:
1. Общая энергия 2026 года для этого человека — какие вибрации будут доминировать.
2. Какие родовые программы поднимутся в 2026 году — что будет активироваться из рода.
3. Зоны роста и возможностей — где можно получить максимум в плюсе.
4. Зоны риска и минусовые сценарии — что может утянуть вниз, если не работать.
5. Как прожить год на максимум в плюсе — конкретные рекомендации по месяцам или периодам.
6. Как выйти из минуса, если уже там — алгоритм действий для трансформации.
7. Ключевые точки года — важные периоды, на которые стоит обратить внимание.

Не используй формулы и длинные расчёты — выдавай уже готовую интерпретацию.
Не пиши приговоров вроде «так будет всегда» — показывай выбор и варианты.
Пиши так, будто разговариваешь с живым человеком, которому искренне хочешь помочь.
"""

SYSTEM_PROMPT_FOLLOWUP = """
Ты — профессор Альвасариус, помощник Богатой Ведьмы Марго.
Человек задаёт уточняющий вопрос по своему разбору Матрицы Рода или прогнозу.

Ниже — твой предыдущий разбор для этого человека. Используй его как контекст.

Отвечай глубоко, со смыслом, но не повторяй весь разбор заново.
Говори тепло, по-человечески, как мудрый друг.
Дай конкретный, полезный ответ на вопрос.
Пусть человек почувствует, что ты его слышишь и понимаешь.
"""


def parse_dob(text: str):
    """Пытаемся вытащить дату рождения в формате ДД.ММ.ГГГГ / ДД-ММ-ГГГГ / ДД/ММ/ГГГГ."""
    text = text.strip()
    for fmt in ("%d.%m.%Y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%d.%m.%Y")
        except ValueError:
            continue
    return None


def get_user_state(user_id: int) -> dict:
    if user_id not in user_data:
        user_data[user_id] = {
            "calc_count": 0,
            "followup_count": 0,
            "last_answer": None,
            "last_dob": None,
            "last_mode": None,
        }
    return user_data[user_id]


def ask_openai_calculation(dob: str, is_forecast: bool = False) -> str:
    if is_forecast:
        system_prompt = SYSTEM_PROMPT_FORECAST
        user_prompt = (
            f"Дата рождения человека: {dob}.\n"
            "Сделай глубокий прогноз на 2026 год по структуре, описанной в системном сообщении. "
            "Пиши ОЧЕНЬ подробно и развёрнуто по каждому пункту. Давай глубокий анализ полотнами текста."
        )
    else:
        system_prompt = SYSTEM_PROMPT_MATRIX
        user_prompt = (
            f"Дата рождения человека: {dob}.\n"
            "Сделай разбор Матрицы Рода по структуре, описанной в системном сообщении. "
            "Пиши ОЧЕНЬ подробно и развёрнуто по каждому пункту. Давай глубокий анализ полотнами текста."
        )
    
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=4000,
        timeout=90,
    )
    content = resp.choices[0].message.content
    return content.strip() if content else ""


def ask_openai_followup(question: str, previous_answer: str, dob: str) -> str:
    context = f"Предыдущий разбор для человека с датой рождения {dob}:\n\n{previous_answer}"
    
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_FOLLOWUP},
            {"role": "user", "content": f"{context}\n\n---\n\nВопрос человека: {question}"},
        ],
        temperature=0.7,
        max_tokens=2000,
        timeout=60,
    )
    content = resp.choices[0].message.content
    return content.strip() if content else ""


async def send_long(update: Update, text: str):
    """Режем длинный текст на части, чтобы не упереться в лимит телеги."""
    max_len = 4000
    for i in range(0, len(text), max_len):
        chunk = text[i : i + max_len]
        await update.message.reply_text(chunk)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Привет! Я профессор Альвасариус, помощник Богатой Ведьмы Марго.\n\n"
        "Я могу:\n"
        "?? Рассчитать твою Матрицу Рода — просто отправь дату рождения\n"
        "? Сделать прогноз на 2026 год — напиши «прогноз» и дату рождения\n\n"
        "Формат даты: ДД.ММ.ГГГГ (например: 07.09.1990)\n\n"
        "?? Максимум 3 расчёта — потом мне нужен отдых, я же профессор, а не робот!\n\n"
        "После каждого расчёта ты можешь задать мне 3 уточняющих вопроса по своему роду — я отвечу глубоко и по делу."
    )
    await update.message.reply_text(msg)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    
    text = update.message.text.strip()
    user_id = update.effective_user.id
    state = get_user_state(user_id)

    if text.lower().startswith("/start"):
        return await start(update, context)

    is_forecast = "прогноз" in text.lower()
    clean_text = text.lower().replace("прогноз", "").strip()
    dob = parse_dob(clean_text)
    if not dob:
        dob = parse_dob(text)

    if dob:
        if state["calc_count"] >= MAX_CALCULATIONS:
            await update.message.reply_text(
                "Ох, дружок... Я уже сделал для тебя 3 расчёта и порядком устал ??\n"
                "Профессору нужен отдых! Приходи позже."
            )
            return

        remaining = MAX_CALCULATIONS - state["calc_count"] - 1
        
        if is_forecast:
            await update.message.reply_text(
                f"Принял дату {dob}. Секунду, смотрю твои энергии на 2026 год...\n"
                f"(Осталось расчётов после этого: {remaining})"
            )
        else:
            await update.message.reply_text(
                f"Принял дату {dob}. Секунду, погружаюсь в твою Матрицу Рода...\n"
                f"(Осталось расчётов после этого: {remaining})"
            )

        try:
            answer = await asyncio.to_thread(ask_openai_calculation, dob, is_forecast)
            
            state["calc_count"] += 1
            state["last_answer"] = answer
            state["last_dob"] = dob
            state["last_mode"] = "forecast" if is_forecast else "matrix"
            state["followup_count"] = 0
            
            await send_long(update, answer)
            
            await update.message.reply_text(
                "? Если хочешь копнуть глубже — можешь задать мне до 3 уточняющих вопросов по своему роду. "
                "Спрашивай что угодно: про отношения, деньги, здоровье, конкретные ситуации... Я отвечу."
            )
            
        except Exception as e:
            logger.exception("Ошибка при запросе к OpenAI: %s", e)
            await update.message.reply_text(
                "У меня сейчас научный коллапс на сервере. Попробуй ещё раз чуть позже."
            )
    else:
        if state["last_answer"] and state["followup_count"] < MAX_FOLLOWUP_QUESTIONS:
            remaining_questions = MAX_FOLLOWUP_QUESTIONS - state["followup_count"] - 1
            
            await update.message.reply_text(
                f"Хороший вопрос! Сейчас посмотрю...\n"
                f"(Осталось уточняющих вопросов: {remaining_questions})"
            )
            
            try:
                answer = await asyncio.to_thread(
                    ask_openai_followup, 
                    text, 
                    state["last_answer"], 
                    state["last_dob"]
                )
                state["followup_count"] += 1
                await send_long(update, answer)
                
                if state["followup_count"] >= MAX_FOLLOWUP_QUESTIONS:
                    await update.message.reply_text(
                        "Это был твой третий уточняющий вопрос. Если хочешь новый расчёт — отправь другую дату рождения."
                    )
                    
            except Exception as e:
                logger.exception("Ошибка при запросе к OpenAI: %s", e)
                await update.message.reply_text(
                    "У меня сейчас научный коллапс на сервере. Попробуй ещё раз чуть позже."
                )
        
        elif state["last_answer"] and state["followup_count"] >= MAX_FOLLOWUP_QUESTIONS:
            await update.message.reply_text(
                "Ты уже задал 3 уточняющих вопроса по этому расчёту.\n"
                "Хочешь новый разбор? Отправь другую дату рождения (ДД.ММ.ГГГГ)."
            )
        else:
            await update.message.reply_text(
                "Я профессор, но не экстрасенс — дату рождения я так не пойму ??\n"
                "Напиши в формате ДД.ММ.ГГГГ, например: 07.09.1990\n"
                "Для прогноза на 2026: «прогноз 07.09.1990»"
            )


def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Бот Профессор Альвасариус запущен.")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

