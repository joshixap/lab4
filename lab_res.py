import pandas as pd
from lab4 import digitize_dataset

# Чтение справочников для цифровизации
with open('symptoms.csv', encoding='utf8') as f:
    symptoms_csv = [line.strip() for line in f if line.strip()]

with open('specialties.csv', encoding='utf8') as f:
    specialties_csv = [line.strip() for line in f if line.strip()]

with open('analyses.csv', encoding='utf8') as f:
    analyses_csv = [line.strip() for line in f if line.strip()]

# Чтение исходного датасета
df = pd.read_csv('dataset.csv', sep=';')

# Цифровизация
digitized_df = digitize_dataset(df, symptoms_csv, specialties_csv, analyses_csv)

# Сохранение результата
digitized_df.to_csv('result.csv', index=False)
print('Готово! Файл result.csv создан.')