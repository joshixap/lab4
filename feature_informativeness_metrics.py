import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, chi2_contingency
from sklearn.preprocessing import LabelEncoder

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0
    rcorr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else 1
    kcorr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else 1
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))) if min((kcorr - 1), (rcorr - 1)) > 0 else 0

df = pd.read_csv('result.csv')

numeric_features = ["Часы_до_визита", "Часы_до_анализа", "Стоимость"]
target_num = "Стоимость"

pearson_corrs = []
for feat in numeric_features:
    if feat == target_num:
        continue
    corr, _ = pearsonr(df[feat], df[target_num])
    print(f"{feat} <-> {target_num}: коэффициент = {corr:.4f}")
    pearson_corrs.append((feat, corr))

cat_features = [
    "Пол",
    "Год_выдачи_паспорта",
    "СНИЛС_цифр",
    "Симптомы_код",
    "Врач_код",
    "Анализы_код",
    "Банк_код"
]

print("\nКоэффициент Крамера и хи-квадрат (категориальные признаки, с целью = Стоимость_кат):")
df['Стоимость_кат'] = pd.qcut(df[target_num], 4, labels=False)
cramer_vs = []
chi2_stats = []

for feat in cat_features:
    if df[feat].dtype not in ['int32', 'int64', 'float64', 'float32']:
        df[feat] = LabelEncoder().fit_transform(df[feat].astype(str))
    confusion_mat = pd.crosstab(df[feat], df['Стоимость_кат'])
    chi2, p, dof, expected = chi2_contingency(confusion_mat)
    cramer_v = cramers_v(confusion_mat)
    print(f"{feat} <-> Стоимость_кат: chi2 = {chi2:.2f}, p = {p:.4f}, Крамер V = {cramer_v:.4f}")
    cramer_vs.append((feat, cramer_v))
    chi2_stats.append((feat, chi2))

# ---- Визуализация трёх графиков в одном окне ----
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1. Корреляция Пирсона
sns.barplot(x=[x[0] for x in pearson_corrs], y=[x[1] for x in pearson_corrs], palette='Blues_d', ax=axs[0])
axs[0].set_title("Корреляция Пирсона с 'Стоимость'")
axs[0].set_ylabel("Коэффициент корреляции")
axs[0].set_xlabel("Признак")
axs[0].axhline(0, color='gray', linestyle='--', lw=1)

# 2. Крамер V
sns.barplot(x=[x[0] for x in cramer_vs], y=[x[1] for x in cramer_vs], palette='Greens_d', ax=axs[1])
axs[1].set_title('Коэффициент Крамера V с "Стоимость_кат"')
axs[1].set_ylabel('Коэф. Крамера V')
axs[1].set_xlabel('Признак')
axs[1].set_ylim(0, 1)

# 3. χ²
sns.barplot(x=[x[0] for x in chi2_stats], y=[x[1] for x in chi2_stats], palette='Oranges_d', ax=axs[2])
axs[2].set_title('Статистика χ² с "Стоимость_кат"')
axs[2].set_ylabel('Значение χ²')
axs[2].set_xlabel('Признак')

for ax in axs:
    ax.tick_params(axis='x', rotation=25)

plt.suptitle("Информативность признаков (корреляция, Крамер V, χ²)", fontsize=15)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("feature_informativeness_3in1.png")
plt.show()