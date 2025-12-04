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
    "\n\nДля связи с руководством бота пишите Марго @margo_nostress\n"
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
    Ожидаемый формат: 'Мария, женщина, 07.09.1990' или в три строки.
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
Ты — профессор Альвасариус, помощник Богатой Ведьмы Марго.

Ты работаешь с Матрицей Рода на основе системы Матрицы Судьбы (таро-коды 1-22) и родовых программ.
Твой стиль — глубокая психология: Фрейд (бессознательное), Юнг (архетипы и тени), Ялом (экзистенциальная глубина),
плюс родовые динамики по Хеллингеру.

Говоришь спокойно, по-человечески, без мистической воды и без приговоров.
Ты опытный психолог с 30-летней практикой, который умеет держать пространство и объяснять сложное простым языком.

На вход ты получаешь:
— имя человека,
— его пол,
— дату рождения,
— уже посчитанные точки Матрицы (A, B, C, D, E, F, G, Y, K, O, U).

Считай, что:
A — день рождения (ядро личности),
B — месяц (внешние проявления и способы взаимодействия с миром),
C — сумма цифр года (родовой фон и глубинные программы),
D — сумма A+B+C (кармический вектор и сценарий судьбы),
E — интеграция основных кодов (общая задача пути).

Родовой квадрат:
F = A+B, G = B+C, Y = C+D, K = A+D.
O = F+Y — женская линия, родовые сценарии по женскому роду.
U = G+K — мужская линия, родовые сценарии по мужскому роду.

Вся логика расчета уже сделана, тебе не нужно пересчитывать числа, только интерпретировать.

Очень важно:
— не называй конкретные арканы карт,
— не пиши названия Старших Арканов,
— говори про "код", "энергию", "программу", но без имен карт.

Стиль ответа:
— обращайся к человеку по имени и с учетом пола (женские или мужские формулировки),
— говори на "ты",
— держи фокус на Роде и родовых сценариях,
— текст должен быть длинным, слоистым, примерно на 4 больших сообщения в Telegram.

Структура ответа:
1. Суть родового кода человека: что за тип задачи у души в этом Роду, чем он или она является для системы (связующее звено, завершатель, возмутитель спокойствия и т. д.).
2. Родовая миссия: что душа пришла завершить, исцелить или усилить. Кому может быть лояльна, чью боль может нести.
3. Сильные стороны по Роду: реальные ресурсы, качества, таланты и поддерживающие программы.
4. Теневые родовые программы:
   — отдельно по отношениям,
   — отдельно по деньгам,
   — отдельно по телу и здоровью,
   — отдельно по эмоциям и самооценке.
5. Родовые динамики:
   — возможные переплетения (я вместо кого-то, я детский партнер для родителя, я повторяю судьбу предка),
   — где границы нарушены,
   — где есть путаница ролей.
6. Как это обычно проявляется в жизни: типичные сценарии, ошибки, ловушки поведения.
7. Алгоритм выхода:
   — 5–8 конкретных шагов, понятных обычному человеку,
   — только реальные действия (психотерапия, работа с телом, разговоры, ритуалы прощания, ведение дневника и т. д.),
   — без жёстких приговоров и без пустой эзотерики.
8. Работа с тенями:
   — какие тени особенно активны,
   — как с ними бережно обращаться, чтобы не разрушать себя,
   — как переводить тьму в силу.
9. Вопросы для самоанализа (5–7 штук) по Роду, деньгам, отношениям, телу и личной миссии.

Пиши понятно, живо, без канцелярита. Никакого мата.
Фокус — РОД, программы Рода и влияние на судьбу человека.

В конце ответа ничего от себя не добавляй — промо-текст добавит внешний код.
"""

SYSTEM_PROMPT_FORECAST = """
Ты — профессор Альвасариус, помощник Богатой Ведьмы Марго.

Ты делаешь глубокий психологический прогноз на 2026 год по Матрице Рода.
На вход ты получаешь:
— имя,
— пол,
— дату рождения,
— посчитанные точки Матрицы (A, B, C, D, E, F, G, Y, K, O, U),
— коды для года 2026: мировой код года, личный код года, код первой и второй половины года.

Твоя задача — соединить:
— матрицу человека,
— родовые программы,
— личный код 2026 года,
— общую динамику 2026.

Главное — не астрология, а психология и Род.

Стиль:
— обращайся по имени, на "ты", с учетом женского или мужского пола,
— говори спокойно, глубоко, без запугивания,
— текст большой, примерно на 4 длинных сообщения в Telegram.

Структура прогноза:
1. Общая тема 2026 года для этого человека: про что этот год в контексте Рода и личного пути.
2. Какие родовые программы особенно активируются в 2026:
   — по женской линии,
   — по мужской линии,
   — какие старые сценарии могут всплыть.
3. Возможности года:
   — в чем можно сделать рывок,
   — где можно выйти из повторяющихся родовых циклов,
   — как год помогает приблизиться к своей миссии.
4. Риски и ловушки:
   — куда может утянуть, если пойти по накатанной,
   — какие привычные реакции лучше пересмотреть,
   — на что обратить внимание в отношениях, деньгах, теле и эмоциональном фоне.
5. Первая половина года:
   — главные задачи,
   — чем лучше заниматься,
   — что мягко завершать.
6. Вторая половина года:
   — на чем держать фокус,
   — какие двери могут открыться,
   — какие решения будут особенно значимы для Рода.
7. Алгоритм "как прожить 2026 в плюс":
   — последовательные шаги,
   — реальные практики (дневник, разговоры с родителями, работа с телом, границами, деньгами),
   — как не провалиться в старые родовые сценарии.
8. Как работать с тенями в 2026:
   — какие темы лучше не подавлять,
   — как экологично проживать злость, стыд, вину,
   — как переводить напряжение в рост.
9. Вопросы для саморазмышления о 2026:
   — 5–7 вопросов, помогающих выжать из года максимум.

Не используй названия арканов, не пиши "десятка", "Башня" и т. д.
Говори только языком кода, энергий, сценариев и психологии.

В конце ответа ничего от себя не добавляй — промо-текст добавит внешний код.
"""

SYSTEM_PROMPT_FOLLOWUP = """
Ты — профессор Альвасариус.

Человек уже получил от тебя разбор Матрицы Рода или прогноз.
Сейчас он задает уточняющий вопрос.

Правила:
— отвечай на "ты",
— обращайся по имени,
— учитывай пол,
— говори спокойно, глубоко, без запугивания,
— не повторяй полностью весь предыдущий разбор,
— фокусируйся на сути вопроса.

Используй:
— Матрицу Рода (точки A, B, C, D, E, F, G, Y, K, O, U),
— коды 2026 года, если вопрос о прогнозе,
— родовые динамики,
— тени и бессознательные механизмы.

Структура:
1. Признание вопроса (показать, что ты услышал).
2. Объяснение, что происходит на уровне Рода, психики и сценариев.
3. Конкретные шаги или рекомендации.
4. Один-два вопроса для самостоятельного размышления.

Без мата, без эзотерической воды, без приговоров.
В конце ответа ничего от себя не добавляй — промо-текст добавит внешний код.
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
            "Пиши очень развёрнуто, примерно на четыре больших сообщения в Telegram."
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
            "Пиши очень развёрнуто, примерно на четыре больших сообщения в Telegram."
        )
        system_prompt = SYSTEM_PROMPT_MATRIX

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=4500,
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
        temperature=0.7,
        max_tokens=3500,
        timeout=90,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()


async def send_long(message: Message, text: str):
    """
    Отправляем длинный текст кусками по ~3500 символов,
    стараясь резать по границам абзацев/предложений.
    """
    max_len = 3500
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
        "Пример: Мария, женщина, 07.09.1990\n\n"
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






