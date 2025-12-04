# -*- coding: utf-8 -*-
import asyncio
import logging
import os
from datetime import datetime

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart
from openai import OpenAI

# -------- КЛЮЧИ --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not OPENAI_API_KEY or not TELEGRAM_TOKEN:
    raise RuntimeError("Нужно указать OPENAI_API_KEY и TELEGRAM_TOKEN в переменных окружения")

client = OpenAI(api_key=OPENAI_API_KEY)
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# -------- ЛОГИ --------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# -------- ХРАНИЛКА --------
user_data = {}

def get_state(uid):
    if uid not in user_data:
        user_data[uid] = {
            "name": None,
            "gender": None,
            "dob": None,
            "calc_count": 0,
            "followups": 0,
            "last_answer": None,
        }
    return user_data[uid]

# -------- ПАРСИНГ ДАТЫ --------
def parse_dob(text):
    text = text.strip()
    for fmt in ("%d.%m.%Y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%d.%m.%Y")
        except:
            pass
    return None

# -------- СИСТЕМНЫЙ ПРОМПТ --------
SYSTEM_PROMPT = """
Ты — профессор Альвасариус, помощник Богатой Ведьмы Марго.
Говоришь мягко, глубоко, как психолог с 30-летней практикой
(Фрейд — бессознательное, Юнг — тени и архетипы, Ялом — смысл и экзистенция)
и включаешь родовые динамики Хеллингера.

Ты разбираешь только МАТРИЦУ РОДА:
- родовые сценарии
- лояльности
- переплетения
- тени
- ресурсы
- миссия рода
- алгоритм выхода

Избегай арканов, слов "матрица судьбы", эзотерической воды и приговоров.
Пиши длинно, развёрнуто, глубоко, до 4 частей текста.

Структура разбора:
1. Суть родового кода человека
2. Родовая миссия и что он завершает
3. Сильные стороны и поддержка рода
4. Теневые программы (по Юнгу, Фрейду)
5. Родовые динамики Хеллингера
6. Проявления в реальной жизни
7. Алгоритм выхода
8. Работа с тенью (практика)
9. Вопросы для самоанализа

Всегда в конце добавляй:
"Для связи с моим руководством Марго @margo_nostress"
"""

FOLLOWUP_PROMPT = """
Отвечай как психолог Фрейд + Юнг + Ялом, с родовыми динамиками.
Кратко, глубоко, по сути.
Не повторяй разбор, не воды.
В конце добавляй:
"Для связи с руководством бота — Марго @margo_nostress"
"""

# -------- OPENAI --------
def ask_openai_full(name, gender, dob):
    user_prompt = f"""
Имя: {name}
Пол: {gender}
Дата рождения: {dob}

Сделай глубокий разбор Матрицы Рода по структуре в системном сообщении.
Пиши длинно, дели на блоки естественным образом.
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=6000,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def ask_openai_followup(question, prev, name, gender, dob):
    user_prompt = f"""
Имя: {name}
Пол: {gender}
Дата рождения: {dob}

Предыдущий текст:
{prev}

Вопрос:
{question}
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": FOLLOWUP_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=3000,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

# -------- ОТПРАВКА ДЛИННЫХ СООБЩЕНИЙ --------
async def send_long(message: Message, text: str):
    max_len = 3900
    for i in range(0, len(text), max_len):
        await message.answer(text[i:i+max_len])

# -------- ХЕНДЛЕРЫ --------
@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "Привет! Я профессор Альвасариус.\n\n"
        "Чтобы рассчитать твою Матрицу Рода — напиши три строки:\n"
        "1. Имя\n"
        "2. Пол (женский или мужской)\n"
        "3. Дата рождения (ДД.ММ.ГГГГ)"
    )

@dp.message(F.text)
async def handle_text(message: Message):
    text = message.text.strip()
    uid = message.from_user.id
    st = get_state(uid)

    # --- если ещё нет имени
    if st["name"] is None:
        st["name"] = text
        await message.answer("Записал имя. Теперь напиши пол (женский или мужской).")
        return

    # --- если нет пола
    if st["gender"] is None:
        if text.lower() not in ("женский", "мужской"):
            await message.answer("Пол должен быть: женский или мужской.")
            return
        st["gender"] = text.lower()
        await message.answer("Отлично. Теперь напиши дату рождения: ДД.ММ.ГГГГ")
        return

    # --- если нет даты рождения
    if st["dob"] is None:
        dob = parse_dob(text)
        if not dob:
            await message.answer("Формат даты не распознан. Напиши ДД.ММ.ГГГГ")
            return
        st["dob"] = dob
        await message.answer("Принял. Погружаюсь в Матрицу Рода…")

        answer = await asyncio.to_thread(ask_openai_full, st["name"], st["gender"], st["dob"])
        st["last_answer"] = answer

        await send_long(message, answer)

        await message.answer(
            "Можешь задать до 3 уточняющих вопросов по своему роду. Я отвечу глубоко и по делу."
        )
        return

    # --- если это уточняющий вопрос
    if st["followups"] < 3:
        st["followups"] += 1

        answer = await asyncio.to_thread(
            ask_openai_followup,
            text,
            st["last_answer"],
            st["name"],
            st["gender"],
            st["dob"],
        )
        await send_long(message, answer)

        if st["followups"] >= 3:
            await message.answer(
                "Это был последний уточняющий вопрос. "
                "Если хочешь новый разбор — напиши /start"
            )
        return

    # --- лимит уточнений
    await message.answer("Лимит уточняющих вопросов исчерпан. Напиши /start для нового разбора.")

# -------- ЗАПУСК --------
async def main():
    logger.info("Бот Альвасариус запущен.")
    await dp.start_polling(bot, allowed_updates=["message"])

if __name__ == "__main__":
    asyncio.run(main())


