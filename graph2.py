import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re

# ----------- Справочники -------------------
SPECIALTIES = {
    "Group1": [
        "Терапевт", "Педиатр", "Стоматолог", "Оториноларинголог (ЛОР)", "Офтальмолог", "Дерматолог"
    ],
    "Group2": [
        "Хирург", "Гинеколог", "Кардиолог", "Уролог", "Травматолог-ортопед", "Гастроэнтеролог",
        "Невролог", "Психотерапевт", "Пульмонолог", "Эндокринолог", "Маммолог",
        "Сосудистый хирург (флеболог)", "Ревматолог", "Диетолог", "Физиотерапевт",
        "Аллерголог-иммунолог", "Проктолог"
    ],
    "Group3": [
        "Психиатр", "Нефролог", "Онколог", "Инфекционист", "Стоматолог-хирург", "Эндоскопист",
        "Гематолог", "Токсиколог", "Фтизиатр", "Диабетолог", "Мануальный терапевт",
        "Колопроктолог", "Врач функциональной диагностики", "Врач УЗИ-диагностики",
        "Сексолог", "Косметолог", "Трихолог", "Подолог", "Реабилитолог",
        "Врач ЛФК", "Неонатолог"
    ],
    "Group4": [
        "Детский невролог", "Детский кардиолог", "Детский эндокринолог"
    ],
    "Group5": [
        "Акушер-гинеколог", "Генетик", "Кардиохирург", "Неврореаниматолог", "Эпилептолог",
        "Сомнолог", "Андролог", "Сурдолог"
    ]
}

SYMPTOMS = {
    "Group1": [
        "повышение температуры", "лихорадка", "кашель", "головная боль", "слабость",
        "боль", "отёки", "рвота", "тошнота", "снижение аппетита", "потеря аппетита",
        "бессонница", "жажда"
    ],
    "Group2": [
        "одышка", "насморк", "снижение слуха", "хрипы", "изжога", "диарея", "запор",
        "заложенность носа", "головокружение", "потливость", "припухлость",
        "чувствительность зубов", "утренняя скованность", "кровотечение",
        "боль в животе", "боль в груди", "боль в пояснице", "ограничение движений",
        "нарушение движений"
    ],
    "Group3": [
        "онемение конечностей", "панические атаки", "тревога", "депрессия", "перепады настроения",
        "выделения", "зуд", "психологические проблемы", "отек", "невозможность удаления зуба",
        "слабость после перенесенных заболеваний", "повышенное давление", "высокое давление",
        "аритмия", "шум в ушах", "снижение либидо", "нарушения сна", "потеря веса", "набор веса",
        "снижение веса", "хроническая усталость", "хроническая лихорадка", "ночная потливость"
    ],
    "Group4": [
        "кровь в моче", "боль при мочеиспускании", "частые позывы", "изменение цвета мочи",
        "отставание в росте", "опережение в росте", "избыток веса у детей", "проблемы с половым созреванием",
        "выделения из сосков", "уплотнения", "акне", "рубцы", "пигментация", "новообразования на коже",
        "родинки", "кровохарканье", "трудности с кормлением", "желтуха", "судороги", "приступы потери сознания",
        "снижение слуха", "подбор терапии при эпилепсии", "гиперактивность", "тики", "задержка речевого развития",
        "недержание", "сложные случаи геморроя", "схватки", "варикозные узлы", "сосудистые звездочки",
        "шумы в сердце у ребенка", "синеватый оттенок кожи", "последствия инсультов", "последствия черепно-мозговых травм",
        "проблемы с алкоголем", "проблемы с курением", "зависимость от препаратов", "абстиненция",
        "нарушения эрекции", "бесплодие", "боль в яичках", "низкое либидо", "необходимость проведения ФГДС",
        "необходимость проведения колоноскопии", "необходимость операции"
    ]
}

TEST_GROUPS = {
    "Group1": [
        "общий анализ крови", "общий анализ мочи", "биохимический анализ крови", "глюкоза крови"
    ],
    "Group2": [
        "экг", "узи", "рентген грудной клетки", "узи органов брюшной полости", "рентген зубов",
        "мазок из носа/горла", "мазок на флору", "мазок на инфекции", "соскоб кожи",
        "соскоб кожи на грибки", "узи органов малого таза", "узи почек и мочевого пузыря",
        "ээг", "спирометрия", "анализ на гормоны щитовидной железы", "ттг", "ревматоидный фактор",
        "коагулограмма", "аудиометрия", "анализ кала на скрытую кровь", "фгдс", "аллергопробы",
        "эхокардиография", "маммография", "онкомаркеры", "серологические анализы"
    ],
    "Group3": [
        "мрт головного мозга", "мрт сустава", "мрт позвоночника", "узи беременности", "генетический анализ",
        "колоноскопия", "анализ на гликированный гемоглобин", "бактериологический посев мочи",
        "уздг вен нижних конечностей", "спермограмма", "томография глаза", "офтальмоскопия",
        "психодиагностика", "индивидуальное занятие лфк (30 минут)",
        "консультация врача-реабилитолога", "механотерапия на аппарате с электроприводом",
        "ультразвуковая терапия (1 поле)", "дарсонвализация кожи (1 поле)",
        "магнитотерапия (1 поле)", "полисомнография"
    ]
}
# ---------------------------------------------------------

def normalize_val(val):
    return val.lower().replace('ё','е').strip()

def map_value_to_group(val, group_dict):
    val_norm = normalize_val(val)
    for group_name, value_list in group_dict.items():
        for ref in value_list:
            ref_norm = normalize_val(ref)
            if val_norm == ref_norm:
                return group_name
    return "Other"

def split_row(row):
    # Разбивает по запятой и удаляет пробелы с краев
    return [x.strip() for x in re.split(r',', row) if x.strip()]

def parse_doctors(df):
    group_hist = {g:0 for g in SPECIALTIES}
    group_hist["Other"] = 0
    for v in df['Врач'].dropna():
        grp = map_value_to_group(v, SPECIALTIES)
        group_hist[grp] += 1
    return group_hist

def parse_symptoms(df):
    group_hist = {g:0 for g in SYMPTOMS}
    group_hist["Other"] = 0
    for row in df['Симптомы'].dropna():
        for val in split_row(row):
            grp = map_value_to_group(val, SYMPTOMS)
            group_hist[grp] += 1
    return group_hist

def parse_tests(df):
    group_hist = {g:0 for g in TEST_GROUPS}
    group_hist["Other"] = 0
    for row in df['Анализы'].dropna():
        for val in split_row(row):
            grp = map_value_to_group(val, TEST_GROUPS)
            group_hist[grp] += 1
    return group_hist

def parse_dates(df, col_name):
    def days_since_2022(date_str):
        if not date_str or pd.isna(date_str):
            return None
        date_str = date_str.replace('T', ' ').strip()
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        except Exception:
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d")
            except Exception:
                return None
        base = datetime(2022, 1, 1)
        return (d - base).days

    days_list = [days_since_2022(val) for val in df[col_name].dropna()]
    return [d for d in days_list if d is not None and d >= 0]

def plot_bar(data, xlabel, ylabel, title, filename, color=None):
    plt.figure()
    plt.bar(data.keys(), data.values(), color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def plot_hist(data, xlabel, ylabel, title, filename, color=None):
    plt.figure()
    plt.hist(data, bins=50, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def main():
    df = pd.read_csv('dataset.csv', sep=';', dtype=str, encoding='utf-8')

    doctor_stats = parse_doctors(df)
    plot_bar(doctor_stats, "Doctor Group", "Count", "Doctor Group Distribution", "doctor_group_dist.png")

    symptom_stats = parse_symptoms(df)
    plot_bar(symptom_stats, "Symptom Group", "Count", "Symptom Group Distribution", "symptom_group_dist.png", color='orange')

    test_stats = parse_tests(df)
    plot_bar(test_stats, "Test Group", "Count", "Test Group Distribution", "test_group_dist.png", color='teal')

    visit_days = parse_dates(df, 'Дата_посещения')
    plot_hist(visit_days, "Дни с 2022 года до визита", "Частота", "Распределение посещений по дням", "visit_days_dist.png", color='purple')

    analysis_days = parse_dates(df, 'Дата_анализов')
    plot_hist(analysis_days, "Дни с 2022 года до анализов", "Частота", "Распределение анализов по дням", "analysis_days_dist.png", color='cyan')

if __name__ == "__main__":
    main()