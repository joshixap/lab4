import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def fill_linear_multi_iter(df, target_col, feature_cols, max_iter=20):
    """
    Итеративно заполняет пропуски в target_col по нескольким признакам feature_cols.
    Останавливается, если новые значения не появляются или достигнут лимит итераций.
    """
    df = df.copy()
    for _ in range(max_iter):
        na_before = df[target_col].isna().sum()
        mask_train = df[target_col].notnull() & df[feature_cols].notnull().all(axis=1)
        if mask_train.sum() < 2:
            break
        X_train = df.loc[mask_train, feature_cols]
        y_train = df.loc[mask_train, target_col]
        model = LinearRegression()
        model.fit(X_train, y_train)
        # Ищем, где можем восстановить target_col (есть все признаки, а target_col пустой)
        mask_missing = df[target_col].isnull() & df[feature_cols].notnull().all(axis=1)
        if mask_missing.sum() == 0:
            break
        X_missing = df.loc[mask_missing, feature_cols]
        df.loc[mask_missing.index, target_col] = model.predict(X_missing)
        na_after = df[target_col].isna().sum()
        if na_after == 0 or na_after == na_before:
            break
    return df

def zet_fill_simple(df, col):
    """
    Заполняет пропуски в 'col' только по значениям верхнего и нижнего соседа (i-1, i+1).
    Если оба соседа заполнены:
      - Для числовых признаков — берёт среднее.
      - Для категориальных — берёт моду (или первого соседа при проблеме).
    Все остальные пропуски должны быть заполнены по другим правилам отдельно.
    """
    df = df.copy()
    arr = df[col].values.copy()
    is_numtype = pd.api.types.is_numeric_dtype(df[col])
    n = len(arr)
    for i in range(n):
        if pd.isnull(arr[i]):
            neighbors = []
            # Верхний сосед
            if i > 0 and not pd.isnull(arr[i-1]):
                neighbors.append(arr[i-1])
            # Нижний сосед
            if i < n-1 and not pd.isnull(arr[i+1]):
                neighbors.append(arr[i+1])
            if len(neighbors) == 2:
                if is_numtype:
                    try:
                        arr[i] = np.mean([float(x) for x in neighbors])
                    except Exception:
                        pass
                else:
                    try:
                        arr[i] = pd.Series(neighbors).mode()[0]
                    except Exception:
                        arr[i] = neighbors[0]
    df[col] = arr
    return df

def fill_remaining_gaps(df, fill_strategy="mean", cols=None):
    """
    Заполняет любые оставшиеся пропуски в указанных столбцах (или во всех, если не указаны)
    с помощью среднего/моды по уже заполненным значениям.
    """
    df = df.copy()
    if cols is None:
        cols = df.columns
    for col in cols:
        if df[col].isna().sum() == 0:
            continue
        is_numtype = pd.api.types.is_numeric_dtype(df[col])
        if is_numtype and fill_strategy == "mean":
            fill_value = df[col].dropna().mean()
        elif is_numtype and fill_strategy == "median":
            fill_value = df[col].dropna().median()
        else:
            fill_value = df[col].dropna().mode()[0] if not df[col].dropna().mode().empty else None
        df[col] = df[col].fillna(fill_value)
    return df

if __name__ == "__main__":
    df = pd.read_csv("isodata.csv", sep=",")
    features = [
        "Пол",
        "Год_выдачи_паспорта",
        "СНИЛС_цифр",
        "Симптомы_код",
        "Врач_код",
        "Часы_до_визита",
        "Анализы_код",
        "Часы_до_анализа",
        "Стоимость",
        "Банк_код"
    ]

    # 1. Zet-алгоритм для каждого признака
    for col in features:
        df = zet_fill_simple(df, col)

    # 2. Восстанавливаем сначала строки с одним пробелом для каждого признака
    for target in features:
        df_one_empty = df[df[target].isna() & df[[f for f in features if f != target]].notnull().all(axis=1)]
        if not df_one_empty.empty:
            df.loc[df_one_empty.index, :] = fill_linear_multi_iter(
                df_one_empty, target, [f for f in features if f != target]
            )

    # 3. Для строк с несколькими пробелами (например, 2 и более)
    for target in features:
        df_multi_empty = df[
            df[target].isna() & (df[[f for f in features if f != target]].notnull().sum(axis=1) >= 1)
        ]
        if not df_multi_empty.empty:
            df.loc[df_multi_empty.index, :] = fill_linear_multi_iter(
                df_multi_empty, target, [f for f in features if f != target]
            )

    # 4. В конце все оставшиеся пустые ячейки восполняются средними/модой
    df = fill_remaining_gaps(df, fill_strategy="mean", cols=features)