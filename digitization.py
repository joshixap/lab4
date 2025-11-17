import pandas as pd
import numpy as np
import re
import math
from datetime import datetime

def digitize_fio(fio):
    parts = fio.upper().split()
    if any(p.endswith("А") or p.endswith("Я") or p.endswith("ВНА") or p.endswith("ИНА") for p in parts):
        return 1
    return 0

def digitize_passport(passport):
    digits = re.sub(r"\D", "", passport if pd.notnull(passport) else "")
    if len(digits) >= 4:
        year = int(digits[2:4])
        return 2000 + year
    return np.nan

def digitize_snils(snils):
    snils_digits = re.sub(r"\D", "", snils if pd.notnull(snils) else "")
    if len(snils_digits) >= 11:
        num = snils_digits[:9]
        ctrl = snils_digits[-2:]
        s = sum(int(num[i]) * (9 - i) for i in range(9))
        if s < 100:
            chk = f"{s:02d}"
        elif s in [100, 101]:
            chk = "00"
        elif s > 101:
            rem = s % 101
            chk = f"{rem:02d}" if rem < 100 else "00"
        else:
            chk = "00"
        return int(chk + ctrl)
    return np.nan

def digitize_symptoms(symptoms, symptoms_csv):
    group_weights = [0.42, 0.28, 0.14, 0.14]
    symptoms_map = {}
    for idx, line in enumerate(symptoms_csv, 1):
        items = line.split(";")
        if len(items) >= 3:
            name, _, group = items
            symptoms_map[name.strip().lower()] = (idx, int(group))
    input_symptoms = [s.strip().lower() for s in (symptoms if pd.notnull(symptoms) else "").split(",")]
    digitsum = 0
    last_group = 1
    for s in input_symptoms:
        if s in symptoms_map:
            idx, group = symptoms_map[s]
            weight = group_weights[group - 1]
            digitsum += idx * weight
            last_group = group
    return int(math.floor(digitsum)) * 10 + last_group

def digitize_doctor(doctor, specialties_csv):
    group_weights = [0.4, 0.37, 0.15, 0.03, 0.05]
    specialties_map = {}
    for idx, line in enumerate(specialties_csv, 1):
        items = line.split(";")
        if len(items) >= 2:
            name, group = items
            specialties_map[name.strip().lower()] = (idx, int(group))
    doctor_clean = (doctor if pd.notnull(doctor) else "").strip().lower()
    if doctor_clean in specialties_map:
        idx, group = specialties_map[doctor_clean]
        weight = group_weights[group - 1]
        value = int(math.floor(idx * weight)) * 10 + group
        return value
    return np.nan

def digitize_visit_date(dt_str):
    try:
        if pd.isnull(dt_str):
            return np.nan
        visit_date = datetime.strptime(str(dt_str)[:16], "%Y-%m-%dT%H:%M")
        start_date = datetime(2022, 1, 1, 0, 0)
        diff = visit_date - start_date
        hrs = int(diff.total_seconds() // 3600)
        return hrs
    except Exception:
        return np.nan

def digitize_analyses(analyses, analyses_csv):
    group_weights = [0.51, 0.34, 0.15]
    analyses_map = {}
    for idx, line in enumerate(analyses_csv, 1):
        items = line.split(";")
        if len(items) >= 2:
            name, group = items
            analyses_map[name.strip().lower()] = (idx, int(group))
    input_analyses = [a.strip().lower() for a in (analyses if pd.notnull(analyses) else "").split(",")]
    digitsum = 0
    last_group = 1
    for a in input_analyses:
        if a in analyses_map:
            idx, group = analyses_map[a]
            weight = group_weights[group - 1]
            digitsum += idx * weight
            last_group = group
    return int(math.floor(digitsum)) * 10 + last_group

def digitize_test_date(dt_str):
    try:
        if pd.isnull(dt_str):
            return np.nan
        test_date = datetime.strptime(str(dt_str)[:16], "%Y-%m-%dT%H:%M")
        start_date = datetime(2022, 1, 1, 0, 0)
        diff = test_date - start_date
        hrs = int(diff.total_seconds() // 3600)
        return hrs
    except Exception:
        return np.nan

def digitize_cost(cost):
    try:
        return float(cost)
    except Exception:
        return np.nan

def digitize_bank(card_num):
    digits = (card_num if pd.notnull(card_num) else "").replace(" ", "")[:6]
    if digits.startswith("2202") or digits.startswith("2200"):
        return 1
    elif digits.startswith("5228") or digits.startswith("5389") or digits.startswith("5211") or digits.startswith("5112"):
        return 2
    elif digits.startswith("4039") or digits.startswith("4377") or digits.startswith("4986") or digits.startswith("4306"):
        return 3
    else:
        return 4

def digitize_row(row, symptoms_csv, specialties_csv, analyses_csv):
    return [
        digitize_fio(row[0]),
        digitize_passport(row[1]),
        digitize_snils(row[2]),
        digitize_symptoms(row[3], symptoms_csv),
        digitize_doctor(row[4], specialties_csv),
        digitize_visit_date(row[5]),
        digitize_analyses(row[6], analyses_csv),
        digitize_test_date(row[7]),
        digitize_cost(row[8]),
        digitize_bank(row[9]),
    ]

def digitize_dataset(df, symptoms_csv, specialties_csv, analyses_csv):
    """
    Преобразует исходный DataFrame df к цифровой форме, используя справочники
    symptoms_csv, specialties_csv, analyses_csv.
    """
    cols = [
        "Пол", "Год_выдачи_паспорта", "СНИЛС_цифр", "Симптомы_код",
        "Врач_код", "Часы_до_визита", "Анализы_код", "Часы_до_анализа",
        "Стоимость", "Банк_код"
    ]
    digitized = []
    for _, row in df.iterrows():
        vals = digitize_row(row.tolist(), symptoms_csv, specialties_csv, analyses_csv)
        digitized.append(vals)
    return pd.DataFrame(digitized, columns=cols)

def main():
    # Чтение справочников
    try:
        with open('symptoms.csv', encoding='utf8') as f:
            symptoms_csv = [line.strip() for line in f if line.strip()]
        with open('specialties.csv', encoding='utf8') as f:
            specialties_csv = [line.strip() for line in f if line.strip()]
        with open('analyses.csv', encoding='utf8') as f:
            analyses_csv = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Один из файлов symptoms.csv, specialties.csv, analyses.csv не найден.")
        symptoms_csv, specialties_csv, analyses_csv = [], [], []

    # Чтение исходного датасета
    colnames = [
        "ФИО", "Паспорт", "СНИЛС", "Симптомы", "Врач",
        "Дата_посещения", "Анализы", "Дата_анализов", "Стоимость", "Карта_оплаты"
    ]
    df = pd.read_csv("dataset.csv", sep=';', header=None, dtype=str, names=colnames)
    
    # Цифровизация
    df_num = digitize_dataset(df, symptoms_csv, specialties_csv, analyses_csv)
    
    # Сохранение результата
    df_num.to_csv("result.csv", index=False)
    print("Цифровизация завершена. Результат сохранён в result.csv.")

if __name__ == "__main__":
    main()