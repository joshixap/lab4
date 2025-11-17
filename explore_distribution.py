import pandas as pd
import matplotlib.pyplot as plt

# Загрузи твой CSV (или замени на нужный путь)
df = pd.read_csv("result.csv")

# Какие признаки визуализировать?
numeric_cols = [
    "Год_выдачи_паспорта", "СНИЛС_цифр", "Симптомы_код", "Врач_код",
    "Часы_до_визита", "Анализы_код", "Часы_до_анализа", "Стоимость"
]
categorical_cols = ["Пол", "Банк_код"]

# ------- Гистограммы по числовым признакам -------
plt.figure(figsize=(16, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    df[col].dropna().hist(bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()

# ------- Бар-графики по категориальным признакам -------
plt.figure(figsize=(10, 4))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, 2, i)
    df[col].value_counts().plot(kind="bar")
    plt.title(f"Распределение {col}")
plt.tight_layout()
plt.show()