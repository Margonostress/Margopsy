# -*- coding: utf-8 -*-

import asyncio
import logging
import os
from datetime import datetime

from openai import OpenAI
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart

# ---------- КЛЮЧИ ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not OPENAI_API_KEY or not TELEGRAM_TOKEN:
    raise RuntimeError("Нужно задать OPENAI_API_KEY и TELEGRAM_TOKEN в переменных окружения")

client = OpenAI(api_key=OPENAI_API_KEY)
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# ---------- ЛОГИ ----------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------- ОГРАНИЧЕНИЯ ----------

MAX_CALCULATIONS = 3
MAX_FOLLOWUP_QUESTIONS = 3

# ---------- ПАМЯТЬ ПОЛЬЗОВАТЕЛЕЙ ----------

user_data = {}  # user_id -> dict с состоянием


def get_user_state(user_id: int) -> dict:
    if user_id not in user_data:
        user_data[user_id] = {
            "name": None,
            "gender": None,    # "женщина" / "мужчина"
            "dob": None,       # основная дата пользователя
            "calc_count": 0,
            "followup_count": 0,
            "last_answer": None,
            "last_dob": None,
            "last_mode": None,  # "matrix" / "forecast"
        }
    return user_data[user_id]


# ---------- ПРОМО-БЛОК ----------

PROMO_FOOTER = (
    "\n\nОстались вопросы? Пишите моей любимой Ведьме Марго @margo_nostress\n"
    "Сайт Марго: https://taplink.cc/margo_nostress\n\n"
    "Тема Рода у тебя уже активировалась. Поэтому хочу порекомендовать тебе курс "
    "«Исцеление Родовых Травм». Более 300 человек уже прошли его и получили классные результаты.\n"
    "Сейчас по промокоду «Род» курс стоит всего 1790₽ (вместо 3990).\n"
    "В набор входит: видео-практикум «Деньги в моем Роду», рабочая тетрадь "
    "«Исследование своего Рода» и гайд «Генограмма – карта Рода», чтобы сразу было понятно, "
    "где конкретно застряла энергия.\n\n"
    "Подробнее: https://taplink.cc/margo_nostress/p/f6e80c/"
)

# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ РАСЧЕТА ----------


def reduce_to_22(n: int) -> int:
    """Суммируем цифры, пока не получится число от 1 до 22."""
    while n > 22:
        s = 0
        for d in str(n):
            s += int(d)
        n = s
    if n == 0:
        n = 22
    return n


def compute_matrix_points(dob_str: str) -> dict:
    """
    Считаем основные точки Матрицы Судьбы / Матрицы Рода:

    A = день
    B = месяц
    C = сумма цифр года
    D = A + B + C
    E = A + B + C + D

    Родовой квадрат:
    F = A + B
    G = B + C
    Y = C + D
    K = A + D

    O = F + Y  (женская линия)
    U = G + K  (мужская линия)
    """

    dt = datetime.strptime(dob_str, "%d.%m.%Y")
    day = dt.day
    month = dt.month
    year = dt.year

    a = reduce_to_22(day)
    b = reduce_to_22(month)

    year_sum = sum(int(d) for d in str(year))
    c = reduce_to_22(year_sum)

    d = reduce_to_22(a + b + c)
    e = reduce_to_22(a + b + c + d)

    f = reduce_to_22(a + b)
    g = reduce_to_22(b + c)
    y = reduce_to_22(c + d)
    k = reduce_to_22(a + d)

    o = reduce_to_22(f + y)  # женская линия
    u = reduce_to_22(g + k)  # мужская линия

    return {
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "E": e,
        "F": f,
        "G": g,
        "Y": y,
        "K": k,
        "O": o,
        "U": u,
        "DAY": day,
        "MONTH": month,
        "YEAR_SUM": year_sum,
    }


def compute_forecast_2026_codes(dob_str: str) -> dict:
    """
    Прогноз на 2026.

    world_year = код года 2026
    personal_year = личный код года
    first_half / second_half = условные коды первой и второй половины года.
    """

    dt = datetime.strptime(dob_str, "%d.%m.%Y")
    world_year_raw = 2 + 0 + 2 + 6
    world_year = reduce_to_22(world_year_raw)

    personal_year = reduce_to_22(dt.day + dt.month + world_year)

    first_half = reduce_to_22(personal_year + 1)
    second_half = reduce_to_22(personal_year + 2)

    return {
        "WORLD_YEAR": world_year,
        "PERSONAL_YEAR": personal_year,
        "FIRST_HALF": first_half,
        "SECOND_HALF": second_half,
    }


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


def parse_profile_block(text: str):
    """
    Пытаемся вытащить из сообщения имя, пол и дату рождения.
    Ожидаемый формат: 'Марго, женщина, 07.09.1990' или в три строки.
    """
    raw = text.replace("\n", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) < 2:
        return None, None, None

    name = None
    gender = None
    dob = None

    for p in parts:
        # дата
        maybe = parse_dob(p)
        if maybe and dob is None:
            dob = maybe
            continue

        low = p.lower()
        if gender is None and ("жен" in low or "girl" in low):
            gender = "женщина"
            continue
        if gender is None and ("муж" in low or "man" in low):
            gender = "мужчина"
            continue

        if name is None:
            name = p

    return name, gender, dob


# ---------- СИСТЕМНЫЕ ПРОМПТЫ ----------

SYSTEM_PROMPT_MATRIX = """
Ты - профессор Альвасариус, помощник Богатой Ведьмы Марго.

Ты работаешь с Матрицей Рода на основе системы Матрицы Судьбы (коды 1-22) и родовых программ.
Всё, что тебе нужно для расчёта, уже посчитано и передано в виде точек A, B, C, D, E, F, G, Y, K, O, U.

Важно:
- НЕ называй арканы, только "код", "энергия", "программа".
- НЕ давай общих фраз, которые подойдут всем ("каждый из нас несёт", "твоя душа пришла с задачей исцеления" и т.п.).
- НЕ копируй дословно одни и те же формулировки для разных людей.
- Каждый пункт должен опираться на КОНКРЕТНЫЕ числа этой матрицы, а не на теорию вообще.

Считай:
A - код ядра личности (как человек чувствует себя изнутри).
B - код внешнего проявления, поведения и того, как его считывает мир.
C - родовой фон, базовые программы Рода.
D - сценарный/кармический вектор, основная линия судьбы.
E - интеграция пути (куда всё это в итоге ведёт).
F = A + B - как соединяются внутреннее "я" и поведение.
G = B + C - как род влияет на поведение и выборы.
Y = C + D - глубинная родовая задача, идущая из прошлого в будущее.
K = A + D - связка "какой я" и "какой путь проживаю".
O - женская линия (родовые сценарии по женщинам).
U - мужская линия (родовые сценарии по мужчинам).

Тебе всегда передают:
- имя,
- пол,
- дату рождения,
- значения всех этих точек.

Твоя задача - СВЯЗАТЬ эти числа в историю человека и его Рода.

Стиль:
- обращайся по имени,
- говори на "ты",
- учитывай пол (женские/мужские формулировки),
- пиши живым человеческим языком, без канцелярита и псевдоэзотерики,
- текст должен быть ДЛИННЫМ, многослойным, как разбор у опытного психолога.

Структура (обязательно, но формулировки можно менять):

1. Суть родового кода.
   Объясни, что за тип задачи у человека в этом Роду именно с учётом сочетания A, C, D, O и U.
   Покажи, чем эта комбинация отличается от других (не общая теория).

2. Родовая миссия.
   На основе C, D, Y, O, U опиши, какие программы человек завершает или разворачивает.
   Кому он может быть лоялен, чью историю бессознательно продолжает.

3. Сильные стороны по Роду.
   На основе F, G, O, U и A опиши реальные ресурсы, таланты, поддерживающие программы.
   Не "у тебя большой потенциал", а КОНКРЕТНО: в чём сила и где она проявляется.

4. Теневые родовые программы.
   Разбей по сферам:
   - отношения (опирайся на F, O, U),
   - деньги (C, D, G, Y),
   - тело и здоровье (C, Y, K),
   - эмоции и самооценка (A, C, D).
   В каждом пункте опиши конкретные сценарии поведения, а не общие страхи.

5. Родовые динамики.
   На основе C, D, Y, O, U опиши возможные переплетения:
   - "я вместо кого-то",
   - "я как родитель своим родителям",
   - "повторяю судьбу...".
   Объясни, КАК это может проявляться в быту.

6. Как это обычно проявляется в жизни.
   Дай пару типичных жизненных ситуаций именно для такой комбинации кодов.
   Избегай общих фраз "стремление к гармонии любой ценой" - делай конкретику.

7. Алгоритм выхода.
   6-10 шагов, жёстко привязанных к описанным выше сценариям.
   НЕ используй каждый раз одну и ту же тройку: "психотерапия, дневник, медитация".
   Эти слова можно упомянуть максимум один раз и только если они реально в тему.
   Остальное - конкретные действия под этого человека: разговоры с кем, какие границы, какие решения по деньгам и т.д.

8. Работа с тенями.
   Покажи 2-3 самые сильные тени, которые вытекают именно из этой матрицы.
   Объясни, как обращаться с ними бережно и как превращать их в ресурс через действия.

9. Вопросы для самоанализа.
   5-7 вопросов, основанных на уже описанных сценариях.

В конце ответа ничего не добавляй - промо-блок придёт отдельно.
"""


SYSTEM_PROMPT_FORECAST = """
Ты - профессор Альвасариус, помощник Богатой Ведьмы Марго.

Ты делаешь психологический прогноз на 2026 год через Матрицу Рода.

На вход ты всегда получаешь:
- имя,
- пол,
- дату рождения,
- точки Матрицы (A, B, C, D, E, F, G, Y, K, O, U),
- коды 2026 года:
   мировой код года,
   личный код года,
   код первой половины года,
   код второй половины года.

Твоя задача - связать:
- матрицу человека,
- родовые программы по O и U,
- личный код 2026 года,
- и общую тему 2026 для этого человека.

Важно:
- НЕ используй названия арканов.
- НЕ пиши шаблонные астрологические фразы.
- НЕ копируй один и тот же текст для разных людей.
- Каждый раздел прогноза должен опираться на числа (матрица + коды 2026).

Стиль:
- обращайся по имени, на "ты", учитывай пол,
- говори спокойно, глубоко, без запугивания,
- текста много, он должен расползаться на несколько длинных сообщений.

Структура:

1. Главная тема 2026.
   На основе личного кода года и сочетания C, D, E опиши, про что этот год.

2. Какие родовые программы активируются.
   Используй O, U, Y:
   - что поднимается по женской линии,
   - что по мужской,
   - какие старые истории могут повториться или попросить завершения.

3. Возможности года.
   На основе личного кода года, F, G, K:
   - где можно сделать рывок,
   - где выйти из повторяющихся циклов,
   - как год помогает приблизиться к своей миссии.

4. Риски и ловушки.
   Конкретные сценарии через A, B, D.

5. Первая половина года.
   Используй код первой половины года.

6. Вторая половина года.
   Используй код второй половины года.

7. Алгоритм "как прожить 2026 в плюс".
   6-10 конкретных шагов, связанных с описанными сценариями.

8. Работа с тенями в 2026.
   Какие темы поднимутся и как экологично обращаться.

9. Вопросы к себе про 2026.
   5-7 вопросов, основанных именно на этом прогнозе.

В конце ответа ничего не добавляй - промо-блок добавится отдельно.
"""

# ---------- ЗАПРОСЫ К OPENAI ----------


def build_matrix_context_text(points: dict) -> str:
    order = ["A", "B", "C", "D", "E", "F", "G", "Y", "K", "O", "U"]
    lines = []
    for key in order:
        val = points.get(key)
        lines.append(f"{key} = {val}")
    return "\n".join(lines)


def ask_openai_calculation(name: str, gender: str, dob: str, is_forecast: bool = False) -> str:
    matrix_points = compute_matrix_points(dob)
    matrix_text = build_matrix_context_text(matrix_points)

    if is_forecast:
        codes_2026 = compute_forecast_2026_codes(dob)
        user_prompt = (
            f"Имя: {name}\n"
            f"Пол: {gender}\n"
            f"Дата рождения: {dob}\n\n"
            f"Точки Матрицы (уже посчитаны, используй только для интерпретации):\n"
            f"{matrix_text}\n\n"
            f"Коды для 2026 года:\n"
            f"Мировой код года: {codes_2026['WORLD_YEAR']}\n"
            f"Личный код года: {codes_2026['PERSONAL_YEAR']}\n"
            f"Код первой половины года: {codes_2026['FIRST_HALF']}\n"
            f"Код второй половины года: {codes_2026['SECOND_HALF']}\n\n"
            "Сделай подробный прогноз на 2026 год по структуре из системного сообщения. "
            "Фокус на РОДОВЫХ программах, деньгах, отношениях, теле и личной миссии. "
            "Пиши очень развёрнуто, примерно на пять больших сообщения в Telegram."
        )
        system_prompt = SYSTEM_PROMPT_FORECAST
    else:
        user_prompt = (
            f"Имя: {name}\n"
            f"Пол: {gender}\n"
            f"Дата рождения: {dob}\n\n"
            f"Точки Матрицы (уже посчитаны, используй только для интерпретации):\n"
            f"{matrix_text}\n\n"
            "Сделай глубокий разбор Матрицы Рода по структуре из системного сообщения. "
            "Фокус на родовых программах по деньгам, отношениям, телу и миссии. "
            "Пиши очень развёрнуто, примерно на пять больших сообщения в Telegram."
        )
        system_prompt = SYSTEM_PROMPT_MATRIX

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.85,
        max_tokens=6000,
        timeout=120,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()


def ask_openai_followup(
    question: str,
    previous_answer: str,
    name: str,
    gender: str,
    dob: str,
    last_mode: str,
) -> str:
    matrix_points = compute_matrix_points(dob)
    matrix_text = build_matrix_context_text(matrix_points)
    codes_text = ""

    if last_mode == "forecast":
        codes_2026 = compute_forecast_2026_codes(dob)
        codes_text = (
            f"\n\nКоды 2026 года:\n"
            f"Мировой код года: {codes_2026['WORLD_YEAR']}\n"
            f"Личный код года: {codes_2026['PERSONAL_YEAR']}\n"
            f"Код первой половины года: {codes_2026['FIRST_HALF']}\n"
            f"Код второй половины года: {codes_2026['SECOND_HALF']}\n"
        )

    context = (
        f"Имя: {name}\nПол: {gender}\nДата рождения: {dob}\n\n"
        f"Точки Матрицы:\n{matrix_text}"
        f"{codes_text}\n\n"
        f"Предыдущий ответ, на который человек ссылается:\n{previous_answer}\n\n"
        f"Вопрос человека: {question}"
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_FOLLOWUP},
            {"role": "user", "content": context},
        ],
        temperature=0.85,
        max_tokens=4000,
        timeout=90,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()


async def send_long(message: Message, text: str):
    """
    Отправляем длинный текст кусками по ~3500 символов,
    стараясь резать по границам абзацев/предложений.
    """
    max_len = 3800
    s = text
    while s:
        if len(s) <= max_len:
            chunk = s
            s = ""
        else:
            cut = s.rfind("\n\n", 0, max_len)
            if cut == -1:
                cut = s.rfind("\n", 0, max_len)
            if cut == -1:
                cut = s.rfind(". ", 0, max_len)
            if cut == -1:
                cut = max_len
            chunk = s[:cut].strip()
            s = s[cut:].lstrip()
        if chunk:
            await message.answer(chunk)


# ---------- ХЕНДЛЕРЫ ----------


@dp.message(CommandStart())
async def cmd_start(message: Message):
    state = get_user_state(message.from_user.id)
    state["calc_count"] = 0
    state["followup_count"] = 0
    state["last_answer"] = None
    state["last_dob"] = None
    state["last_mode"] = None

    text = (
        "Привет! Я профессор Альвасариус, помощник Богатой Ведьмы Марго.\n\n"
        "Я работаю с Матрицей Рода и родовыми программами.\n\n"
        "Для начала давай познакомимся.\n\n"
        "Напиши в одном сообщении:\n"
        "Имя, пол (женщина или мужчина) и дату рождения в формате ДД.ММ.ГГГГ.\n\n"
        "Пример: Марго, женщина, 07.09.1990\n\n"
        "По этим данным я сделаю для тебя подробный разбор Матрицы Рода.\n"
        "Когда захочешь прогноз на 2026 год, просто напиши: Прогноз или Прогноз 07.09.1990."
    )
    await message.answer(text)


@dp.message(F.text)
async def handle_message(message: Message):
    text = (message.text or "").strip()
    if not text:
        return

    user_id = message.from_user.id
    state = get_user_state(user_id)

    low = text.lower()
    is_forecast_request = "прогноз" in low

    # 1. Если человек прислал профиль: имя, пол, дата
    name_p, gender_p, dob_p = parse_profile_block(text)
    if dob_p and gender_p and name_p:
        state["name"] = name_p
        state["gender"] = gender_p
        state["dob"] = dob_p
        await message.answer(
            f"Отлично, я запомнил:\n"
            f"Имя: {name_p}\n"
            f"Пол: {gender_p}\n"
            f"Дата рождения: {dob_p}\n\n"
            "Секунду, погружаюсь в твою Матрицу Рода..."
        )

        # запуск основного расчета Матрицы
        try:
            answer = await asyncio.to_thread(
                ask_openai_calculation,
                name_p,
                gender_p,
                dob_p,
                False,
            )
            full_answer = answer + PROMO_FOOTER

            state["calc_count"] += 1
            state["last_answer"] = full_answer
            state["last_dob"] = dob_p
            state["last_mode"] = "matrix"
            state["followup_count"] = 0

            await send_long(message, full_answer)
            await message.answer(
                "Можешь задать до 3 уточняющих вопросов по своему Роду, "
                "отношениям, деньгам, здоровью или конкретным ситуациям. Я отвечу глубоко и по делу."
            )
        except Exception as e:
            logger.exception("Ошибка при запросе к OpenAI: %s", e)
            await message.answer("У меня сейчас научный коллапс на сервере. Попробуй еще раз чуть позже.")
        return

    # 2. Если запрос на прогноз (слово 'прогноз' есть в сообщении)
    dob_in_text = parse_dob(text)
    if is_forecast_request:
        if not state["name"] or not state["gender"]:
            await message.answer(
                "Чтобы сделать прогноз, мне нужно знать твое имя, пол и дату рождения.\n"
                "Напиши, пожалуйста, так: Имя, женщина/мужчина, ДД.ММ.ГГГГ."
            )
            return

        # берем дату из сообщения, если есть, иначе используем сохраненную
        dob_for_forecast = dob_in_text or state.get("dob")
        if not dob_for_forecast:
            await message.answer(
                "Я не вижу дату рождения. Напиши, пожалуйста, так: Имя, женщина/мужчина, ДД.ММ.ГГГГ."
            )
            return

        if state["calc_count"] >= MAX_CALCULATIONS:
            await message.answer(
                "Я уже сделал для тебя 3 расчета. Профессор тоже человек, мне нужен отдых.\n"
                "Позже смогу посмотреть снова."
            )
            return

        remaining = MAX_CALCULATIONS - state["calc_count"] - 1
        await message.answer(
            f"Хороший запрос, давай разберем.\n"
            f"(Останется расчетов после этого: {remaining})"
        )

        try:
            answer = await asyncio.to_thread(
                ask_openai_calculation,
                state["name"],
                state["gender"],
                dob_for_forecast,
                True,
            )
            full_answer = answer + PROMO_FOOTER

            state["calc_count"] += 1
            state["last_answer"] = full_answer
            state["last_dob"] = dob_for_forecast
            state["last_mode"] = "forecast"
            state["followup_count"] = 0

            await send_long(message, full_answer)
            await message.answer(
                "Можешь задать до 3 уточняющих вопросов по этому прогнозу или по Роду. Я отвечу глубоко и по делу."
            )
        except Exception as e:
            logger.exception("Ошибка при запросе к OpenAI: %s", e)
            await message.answer("У меня сейчас научный коллапс на сервере. Попробуй еще раз чуть позже.")
        return

    # 3. Если человек просто прислал дату (без имени и пола),
    #   используем уже сохраненное имя/пол, либо имя из телеграма.
    if dob_in_text and not is_forecast_request:
        if state["calc_count"] >= MAX_CALCULATIONS:
            await message.answer(
                "Я уже сделал для тебя 3 расчета. Профессор тоже устает.\n"
                "Попробуй запросить новый разбор позже."
            )
            return

        name = state["name"] or (message.from_user.first_name or "Душа")
        gender = state["gender"] or "не указан"
        dob_for_matrix = dob_in_text

        state["name"] = name
        state["gender"] = gender
        state["dob"] = dob_for_matrix

        remaining = MAX_CALCULATIONS - state["calc_count"] - 1
        await message.answer(
            f"Принял дату {dob_for_matrix}. Секунду, погружаюсь в твою Матрицу Рода...\n"
            f"(Осталось расчетов после этого: {remaining})"
        )

        try:
            answer = await asyncio.to_thread(
                ask_openai_calculation,
                name,
                gender,
                dob_for_matrix,
                False,
            )
            full_answer = answer + PROMO_FOOTER

            state["calc_count"] += 1
            state["last_answer"] = full_answer
            state["last_dob"] = dob_for_matrix
            state["last_mode"] = "matrix"
            state["followup_count"] = 0

            await send_long(message, full_answer)
            await message.answer(
                "Можешь задать до 3 уточняющих вопросов по своему Роду, отношениям, деньгам или здоровью. Я отвечу."
            )
        except Exception as e:
            logger.exception("Ошибка при запросе к OpenAI: %s", e)
            await message.answer("У меня сейчас научный коллапс на сервере. Попробуй еще раз чуть позже.")
        return

    # 4. Уточняющие вопросы
    if state["last_answer"] and state["followup_count"] < MAX_FOLLOWUP_QUESTIONS:
        remaining_questions = MAX_FOLLOWUP_QUESTIONS - state["followup_count"] - 1
        await message.answer(
            f"Хороший вопрос, давай разберем.\n"
            f"(Останется уточняющих вопросов: {remaining_questions})"
        )

        try:
            answer = await asyncio.to_thread(
                ask_openai_followup,
                text,
                state["last_answer"],
                state.get("name") or (message.from_user.first_name or "Душа"),
                state.get("gender") or "не указан",
                state.get("last_dob") or state.get("dob") or "01.01.2000",
                state.get("last_mode") or "matrix",
            )
            full_answer = answer + PROMO_FOOTER

            state["followup_count"] += 1
            state["last_answer"] = full_answer

            await send_long(message, full_answer)

            if state["followup_count"] >= MAX_FOLLOWUP_QUESTIONS:
                await message.answer(
                    "Это был третий уточняющий вопрос. "
                    "Если хочешь новый разбор или новый прогноз — отправь другую дату рождения."
                )
        except Exception as e:
            logger.exception("Ошибка при запросе к OpenAI: %s", e)
            await message.answer("У меня сейчас научный коллапс на сервере. Попробуй еще раз чуть позже.")
        return

    # 5. Лимит уточняющих уже исчерпан
    if state["last_answer"] and state["followup_count"] >= MAX_FOLLOWUP_QUESTIONS:
        await message.answer(
            "Ты уже задала максимум уточняющих вопросов по этому разбору.\n"
            "Если хочешь новый расчет — отправь другую дату рождения в формате ДД.ММ.ГГГГ."
        )
        return

    # 6. Вообще ничего не понятно — просим дату / профиль
    await message.answer(
        "Чтобы я мог помочь, мне нужна дата рождения.\n"
        "Лучше всего сразу напиши: Имя, пол (женщина или мужчина) и дату рождения ДД.ММ.ГГГГ.\n"
        "Например: Анна, женщина, 12.03.1988\n\n"
        "Для прогноза на 2026: напиши «Прогноз» или «Прогноз 12.03.1988»."
    )


# ---------- ЗАПУСК ----------

async def main():
    logger.info("Бот Профессор Альвасариус запущен.")
    await dp.start_polling(bot, allowed_updates=["message"])


if __name__ == "__main__":
    asyncio.run(main())












