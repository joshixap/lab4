import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- все группы и функции по цифровизации (оставлены без изменений) ---
SPECIALTY_GROUPS = [
    (1, [
        "Терапевт", "Педиатр", "Стоматолог", "Оториноларинголог (ЛОР)",
        "Офтальмолог", "Дерматолог"
    ]),
    (2, [
        "Хирург", "Гинеколог", "Кардиолог", "Уролог", "Травматолог-ортопед", "Гастроэнтеролог",
        "Невролог", "Психотерапевт", "Пульмонолог", "Эндокринолог", "Маммолог",
        "Сосудистый хирург (флеболог)", "Ревматолог", "Диетолог", "Физиотерапевт",
        "Аллерголог-иммунолог", "Проктолог"
    ]),
    (3, [
        "Психиатр", "Нефролог", "Онколог", "Инфекционист", "Стоматолог-хирург", "Эндоскопист",
        "Гематолог", "Токсиколог", "Фтизиатр", "Диабетолог", "Мануальный терапевт",
        "Колопроктолог", "Врач функциональной диагностики", "Врач УЗИ-диагностики",
        "Сексолог", "Косметолог", "Трихолог", "Подолог", "Реабилитолог",
        "Врач ЛФК", "Неонатолог"
    ]),
    (4, [
        "Детский невролог", "Детский кардиолог", "Детский эндокринолог"
    ]),
    (5, [
        "Акушер-гинеколог", "Генетик", "Кардиохирург", "Неврореаниматолог", "Эпилептолог",
        "Сомнолог", "Андролог", "Сурдолог"
    ])
]
SYMPTOM_GROUPS = [
    (1, [
        "повышение температуры", "лихорадка", "кашель", "головная боль", "слабость",
        "боль", "отёки", "рвота", "тошнота", "снижение аппетита", "потеря аппетита",
        "бессонница", "жажда"
    ]),
    (2, [
        "одышка", "насморк", "снижение слуха", "хрипы", "изжога", "диарея", "запор",
        "заложенность носа", "головокружение", "потливость", "припухлость",
        "чувствительность зубов", "утренняя скованность", "кровотечение",
        "боль в животе", "боль в груди", "боль в пояснице", "ограничение движений",
        "нарушение движений"
    ]),
    (3, [
        "онемение конечностей", "панические атаки", "тревога", "депрессия", "перепады настроения",
        "выделения", "зуд", "психологические проблемы", "отек", "невозможность удаления зуба",
        "слабость после перенесенных заболеваний", "повышенное давление", "высокое давление",
        "аритмия", "шум в ушах", "снижение либидо", "нарушения сна", "потеря веса", "набор веса",
        "снижение веса", "хроническая усталость", "хроническая лихорадка", "ночная потливость"
    ]),
    (4, [
        "кровь в моче", "боль при мочеиспускании", "частые позывы", "изменение цвета мочи",
        "отставание в росте", "опережение в росте", "избыток веса у детей", "проблемы с половым созреванием",
        "выделения из сосков", "уплотнения", "акне", "рубцы", "пигментация", "новообразования на коже",
        "родинки", "кровохарканье", "трудности с кормлением", "желтуха", "судороги", "приступы потери сознания",
        "подбор терапии при эпилепсии", "гиперактивность", "тики", "задержка речевого развития",
        "недержание", "сложные случаи геморроя", "схватки", "варикозные узлы", "сосудистые звездочки",
        "шумы в сердце у ребенка", "синеватый оттенок кожи", "последствия инсультов", "последствия черепно-мозговых травм",
        "проблемы с алкоголем", "проблемы с курением", "зависимость от препаратов", "абстиненция",
        "нарушения эрекции", "бесплодие", "боль в яичках", "низкое либидо", "необходимость проведения ФГДС",
        "необходимость проведения колоноскопии", "необходимость операции"
    ])
]
TEST_GROUPS = [
    (1, [
        "общий анализ крови", "общий анализ мочи", "биохимический анализ крови", "глюкоза крови"
    ]),
    (2, [
        "ЭКГ", "УЗИ", "рентген грудной клетки", "УЗИ органов брюшной полости", "рентген зубов",
        "мазок из носа/горла", "мазок на флору", "мазок на инфекции", "соскоб кожи",
        "соскоб кожи на грибки", "УЗИ органов малого таза", "УЗИ почек и мочевого пузыря",
        "ЭЭГ", "спирометрия", "анализ на гормоны щитовидной железы", "ТТГ", "ревматоидный фактор",
        "коагулограмма", "аудиометрия", "анализ кала на скрытую кровь", "ФГДС", "аллергопробы",
        "Эхокардиография", "маммография", "онкомаркеры", "серологические анализы"
    ]),
    (3, [
        "МРТ головного мозга", "МРТ сустава", "МРТ позвоночника", "УЗИ беременности", "генетический анализ",
        "колоноскопия", "анализ на гликированный гемоглобин", "бактериологический посев мочи",
        "УЗДГ вен нижних конечностей", "спермограмма", "томография глаза", "офтальмоскопия",
        "психодиагностика", "индивидуальное занятие ЛФК (30 минут)",
        "консультация врача-реабилитолога", "механотерапия на аппарате с электроприводом",
        "ультразвуковая терапия (1 поле)", "дарсонвализация кожи (1 поле)",
        "магнитотерапия (1 поле)", "полисомнография"
    ])
]
def load_group_mapping(groups):
    mapping = {}
    for num, items in groups:
        for item in items:
            mapping[item.strip().lower()] = num
    return mapping
SPECIALTY_MAP = load_group_mapping(SPECIALTY_GROUPS)
SYMPTOM_MAP = load_group_mapping(SYMPTOM_GROUPS)
TEST_MAP = load_group_mapping(TEST_GROUPS)

def get_gender(full_name):
    fname = full_name.split()[1] if len(full_name.split()) > 1 else full_name.split()[0]
    return 1 if fname.lower().endswith('а') else 0

def passport_year(passport):
    digits = ''.join(filter(str.isdigit, passport))
    if len(digits) >= 4:
        year = int(digits[2:4])
        return 2000 + year if year < 30 else 1900 + year
    return None

def snils_code(snils):
    snils_digits = ''.join(filter(str.isdigit, snils))
    if len(snils_digits) < 11:
        return None
    base9 = [int(d) for d in snils_digits[:9]]
    total = sum([a * b for a, b in zip(base9, reversed(range(1, 10)))])
    if total < 100:
        control = total
    elif total in (100, 101):
        control = 0
    else:
        control = total % 101
        if control in (100, 101):
            control = 0
    last2 = snils_digits[-2:]
    return f"{str(control).zfill(3)}{last2}"

def specialty_group(specialty):
    spec = specialty.strip().lower()
    return SPECIALTY_MAP.get(spec, "Other")

def symptoms_group(symp):
    items = [x.strip().lower() for x in symp.replace(" и ", ",").replace(";", ",").split(",")]
    res = []
    for i in items:
        if i:
            res.append(SYMPTOM_MAP.get(i, "Other"))
    return res[0] if res else "Other"

def hours_since_2022(date_str):
    try:
        d = datetime.strptime(date_str.strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            d = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        except:
            return None
    base = datetime(2022, 1, 1)
    diff = d - base
    return diff.total_seconds() // 3600

def test_group(test):
    items = [x.strip().lower() for x in test.replace(";", ",").split(",")]
    res = []
    for i in items:
        if i:
            res.append(TEST_MAP.get(i, "Other"))
    return res[0] if res else "Other"

def card_figures(card_str):
    try:
        if "|" in card_str:
            pay_system, bank = card_str.split("|")
        else:
            pay_system, bank = card_str.split()
    except Exception:
        return "Unknown"
    pay_system = pay_system.upper()
    bank = bank.upper().strip()
    if pay_system == 'MIR':
        if bank == 'SBERBANK OF RUSSIA':
            figures = '2202 20'
        elif bank == 'TINKOFF BANK':
            figures = '2200 70'
        elif bank == 'VTB BANK':
            figures = '2200 40'
        else:
            figures = '2200 56'
    elif pay_system == 'MASTERCARD':
        if bank == 'SBERBANK OF RUSSIA':
            figures = '5228 60'
        elif bank == 'TINKOFF BANK':
            figures = '5389 94'
        elif bank == 'VTB BANK':
            figures = '5211 94'
        else:
            figures = '5112 23'
    else:
        if bank == 'SBERBANK OF RUSSIA':
            figures = '4039 33'
        elif bank == 'TINKOFF BANK':
            figures = '4377 73'
        elif bank == 'VTB BANK':
            figures = '4986 29'
        else:
            figures = '4306 43'
    return figures

def main():
    # Проверьте кодировку! (encoding='utf-8' или 'cp1251')
    df = pd.read_csv('dataset.csv', sep=';', dtype=str, encoding='utf-8')
    print("Столбцы:", df.columns.tolist()) # debug

    # 1. Цифровизация ФИО
    df['Gender'] = df['ФИО'].apply(get_gender)
    plt.figure()
    df['Gender'].value_counts().plot(kind="bar", color=['blue', 'red'])
    plt.title("Распределение по полу (0 – муж., 1 – жен.)")
    plt.xlabel("Пол")
    plt.ylabel("Число записей")
    plt.xticks([0,1], ['Male', 'Female'])
    plt.savefig("dist1.png"); plt.close()

    # 2. Год выдачи паспорта
    df['PassportYear'] = df['Паспорт'].apply(passport_year)
    plt.figure()
    df['PassportYear'].dropna().astype(int).hist(bins=16, color="green")
    plt.title("Год выдачи паспорта")
    plt.xlabel("Год")
    plt.ylabel("Частота")
    plt.savefig("dist2.png"); plt.close()

    # 3. СНИЛС-код (последние две цифры и контрольная)
    df['SNILSCode'] = df['СНИЛС'].apply(snils_code)
    plt.figure()
    snils_values = df['SNILSCode'].dropna()
    snils_groups = snils_values.value_counts()
    snils_groups.plot(kind='bar')
    plt.title("Группы СНИЛС (контрольная+2 последние цифры)")
    plt.xlabel("Код")
    plt.ylabel("Частота")
    plt.savefig("dist3.png"); plt.close()

    # 4. Группа врача
    df['DoctorGroup'] = df['Врач'].apply(specialty_group)
    plt.figure()
    doc_counts = df['DoctorGroup'].value_counts().sort_index()
    doc_counts.plot(kind='bar')
    plt.title("Распределение по группам врачей")
    plt.xlabel("№ группы")
    plt.ylabel("Частота")
    plt.savefig("dist4.png"); plt.close()

    # 5. Группа симптома
    df['SymptomGroup'] = df['Симптомы'].apply(symptoms_group)
    plt.figure()
    df['SymptomGroup'].value_counts().sort_index().plot(kind='bar', color='orange')
    plt.title("Распределение по группам симптомов")
    plt.xlabel("№ группы")
    plt.ylabel("Частота")
    plt.savefig("dist5.png"); plt.close()

    # 6. Время с 2022 до визита (часы)
    df['VisitHours'] = df['Дата_посещения'].apply(hours_since_2022)
    plt.figure()
    vi_hours = df['VisitHours'].dropna().astype(int)
    vi_hours.hist(bins=30, color='purple')
    plt.title("Часы с 2022 года до визита")
    plt.xlabel("Часы")
    plt.ylabel("Частота")
    plt.savefig("dist6.png"); plt.close()

    # 7. Время до анализов
    df['AnalysisHours'] = df['Дата_анализов'].apply(hours_since_2022)
    plt.figure()
    an_hours = df['AnalysisHours'].dropna().astype(int)
    an_hours.hist(bins=30, color='cyan')
    plt.title("Часы с 2022 года до анализов")
    plt.xlabel("Часы")
    plt.ylabel("Частота")
    plt.savefig("dist7.png"); plt.close()

    # 8. Группа анализа
    df['TestGroup'] = df['Анализы'].apply(test_group)
    plt.figure()
    df['TestGroup'].value_counts().sort_index().plot(kind='bar', color='teal')
    plt.title("Распределение по группам анализов")
    plt.xlabel("№ группы")
    plt.ylabel("Частота")
    plt.savefig("dist8.png"); plt.close()

    # 9. Стоимость
    df['Cost'] = pd.to_numeric(df['Стоимость'], errors='coerce')
    plt.figure()
    df['Cost'].dropna().hist(bins=40, color='darkred')
    plt.title("Распределение стоимости")
    plt.xlabel("Стоимость")
    plt.ylabel("Частота")
    plt.savefig("dist9.png"); plt.close()

    # 10. Банковские фигурные коды
    df['BankFigures'] = df['Карта_оплаты'].apply(card_figures)
    plt.figure()
    df['BankFigures'].value_counts().plot(kind='bar', color='magenta')
    plt.title("Распределение карт по банковским системам")
    plt.xlabel("Код")
    plt.ylabel("Частота")
    plt.savefig("dist10.png"); plt.close()

if __name__ == "__main__":
    main()