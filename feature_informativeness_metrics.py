import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, chi2_contingency
from sklearn.preprocessing import LabelEncoder

def compute_pearson(df, numeric_features, target):
    pearson_corrs = []
    for feat in numeric_features:
        if feat == target:
            continue
        try:
            corr, _ = pearsonr(df[feat], df[target])
        except Exception:
            corr = np.nan
        pearson_corrs.append((feat, corr))
    return pearson_corrs

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0
    rcorr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else 1
    kcorr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else 1
    denom = min((kcorr - 1), (rcorr - 1)) if min((kcorr - 1), (rcorr - 1)) > 0 else 1
    return np.sqrt(phi2corr / denom) if denom > 0 else 0

def compute_cramer_chi2(df, cat_features, target_num, target_cat_col='Стоимость_кат', n_quantiles=4):
    cramer_vs = []
    chi2_stats = []
    # Создать кат. переменную для стоимости, если нет
    if target_cat_col not in df.columns:
        df[target_cat_col] = pd.qcut(df[target_num], n_quantiles, labels=False)
    for feat in cat_features:
        # Преобразовать в числа, если не числовой
        if df[feat].dtype not in ['int32', 'int64', 'float64', 'float32']:
            df[feat] = LabelEncoder().fit_transform(df[feat].astype(str))
        confusion_mat = pd.crosstab(df[feat], df[target_cat_col])
        chi2, p, dof, expected = chi2_contingency(confusion_mat)
        cramer_v = cramers_v(confusion_mat)
        cramer_vs.append((feat, cramer_v))
        chi2_stats.append((feat, chi2))
    return cramer_vs, chi2_stats

def plot_informativeness(pearson_corrs, cramer_vs, chi2_stats, out_path="feature_informativeness_3in1.png"):
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
    plt.savefig(out_path)
    plt.show()

def compute_feature_informativeness(df, target="Стоимость"):
    # Универсальный ранжированный вывод признаков (и для GUI)
    features = [c for c in df.columns if c != target]
    informativeness = []
    for feat in features:
        try:
            if pd.api.types.is_numeric_dtype(df[feat]) and pd.api.types.is_numeric_dtype(df[target]):
                corr, _ = pearsonr(df[feat].dropna(), df[target].dropna())
                metric = abs(corr)
                metric_name = 'pearson'
            else:
                v1 = LabelEncoder().fit_transform(df[feat].astype(str))
                v2 = LabelEncoder().fit_transform(df[target].astype(str))
                mat = pd.crosstab(v1, v2)
                chi2, p, dof, exp = chi2_contingency(mat)
                phi2 = chi2 / df.shape[0]
                k = min(mat.shape)
                if k > 1 and phi2 > 0:
                    cramer_v = np.sqrt(phi2 / (k - 1))
                else:
                    cramer_v = 0.0
                metric = cramer_v
                metric_name = 'cramer_v'
            informativeness.append({
                "feature": feat,
                "metric": metric,
                "type": metric_name
            })
        except Exception:
            informativeness.append({
                "feature": feat,
                "metric": 0.0,
                "type": "error"
            })
    df_inf = pd.DataFrame(informativeness)
    return df_inf.sort_values(by="metric", ascending=False).reset_index(drop=True)

def analyze_feature_informativeness(df, numeric_features, cat_features, target_num, target_cat_col='Стоимость_кат'):
    pearson_corrs = compute_pearson(df, numeric_features, target_num)
    cramer_vs, chi2_stats = compute_cramer_chi2(df, cat_features, target_num, target_cat_col)
    plot_informativeness(pearson_corrs, cramer_vs, chi2_stats)
    # Для интеграции с GUI — можете вернуть df с метриками для всех признаков
    metrics_df = compute_feature_informativeness(df, target=target_num)
    return pearson_corrs, cramer_vs, chi2_stats, metrics_df

########## Точка входа ##########
if __name__ == "__main__":
    df = pd.read_csv('result.csv')
    numeric_features = ["Часы_до_визита", "Часы_до_анализа", "Стоимость"]
    target_num = "Стоимость"
    cat_features = [
        "Пол",
        "Год_выдачи_паспорта",
        "СНИЛС_цифр",
        "Симптомы_код",
        "Врач_код",
        "Анализы_код",
        "Банк_код"
    ]
    analyze_feature_informativeness(
        df,
        numeric_features=numeric_features,
        cat_features=cat_features,
        target_num=target_num,
        target_cat_col="Стоимость_кат"
    )