import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def neighbors_window(arr, i, window=2):
    n = len(arr)
    neighbors = []
    for w in range(1, window+1):
        if i-w >= 0 and not pd.isnull(arr[i-w]):
            neighbors.append(arr[i-w])
    for w in range(1, window+1):
        if i+w < n and not pd.isnull(arr[i+w]):
            neighbors.append(arr[i+w])
    return neighbors

def zet_fill_window(df, col, window=2, min_neighbors=1):
    df = df.copy()
    arr = df[col].values.copy()
    is_numtype = pd.api.types.is_numeric_dtype(df[col])
    n = len(arr)
    changed = False
    for i in range(n):
        if pd.isnull(arr[i]):
            neighbors = neighbors_window(arr, i, window=window)
            if len(neighbors) >= min_neighbors:
                if is_numtype:
                    arr[i] = np.mean([float(x) for x in neighbors])
                else:
                    mode = pd.Series(neighbors).mode()
                    arr[i] = mode.iloc[0] if not mode.empty else neighbors[0]
                changed = True
            # Самый жесткий but logic-based: если нет достаточных соседей, подставляем ближайший ненулевой (сверху или снизу)
            elif len(neighbors) == 0:
                # поищи ближайший заполненный сверху/снизу (forwards/backwards fill)
                prev_value = None
                next_value = None
                for k in range(i-1, -1, -1):
                    if not pd.isnull(arr[k]):
                        prev_value = arr[k]
                        break
                for k in range(i+1, n):
                    if not pd.isnull(arr[k]):
                        next_value = arr[k]
                        break
                # если есть оба, бери ближайший по индексу
                candidates = [(abs(k - i), val) for k, val in ((i-1, prev_value), (i+1, next_value)) if val is not None]
                if candidates:
                    arr[i] = sorted(candidates)[0][1]
                    changed = True
                # если ни одного — жестко, оставь NaN
    df[col] = arr
    return df, changed

def add_window_features(df, col, window=2):
    feats = pd.DataFrame(index=df.index)
    arr = df[col].values
    for w in range(1, window+1):
        feats[f"{col}_prev{w}"] = pd.Series(arr).shift(w)
        feats[f"{col}_next{w}"] = pd.Series(arr).shift(-w)
    return feats

def fill_linear_window_multi_iter(df, target_col, feature_cols, window=2, min_filled=5, max_iter=20):
    df = df.copy()
    window_feats = add_window_features(df, target_col, window)
    full_features = pd.concat([df[feature_cols], window_feats], axis=1)

    for _ in range(max_iter):
        na_before = df[target_col].isna().sum()
        mask_train = (
            df[target_col].notnull() & (full_features.notnull().sum(axis=1) >= min_filled)
        )
        if mask_train.sum() < 2:
            break
        X_train = full_features.loc[mask_train]
        y_train = df.loc[mask_train, target_col]
        X_train = X_train.fillna(X_train.mean())
        model = LinearRegression()
        model.fit(X_train, y_train)

        mask_missing = (
            df[target_col].isnull() & (full_features.notnull().sum(axis=1) >= min_filled)
        )
        # Жесткий импьюинг: если нет min_filled, все равно попытаться заполнить, используя наиболее похожий (по расстоянию до заполненных)
        mask_missing_loose = df[target_col].isnull() & ~(full_features.notnull().sum(axis=1) >= min_filled)
        if mask_missing.sum() == 0 and mask_missing_loose.sum() == 0:
            break

        if mask_missing.sum() > 0:
            X_missing = full_features.loc[mask_missing]
            X_missing = X_missing.fillna(X_train.mean())
            new_values = model.predict(X_missing)
            for idx, value in zip(mask_missing.index, new_values):
                if pd.isnull(df.at[idx, target_col]):
                    df.at[idx, target_col] = value

        # Импьюинг на "тупых" строках: возьми ближайшее заполненное по индексу (forward/backward fill по строке)
        if mask_missing_loose.sum() > 0:
            arr = df[target_col].values
            n = len(arr)
            for idx in mask_missing_loose.index:
                # поищи ближайший заполненный сверху/снизу (либо в этом же столбце, либо в feature_cols)
                # здесь по этому столбцу
                prev_value = None
                next_value = None
                for k in range(idx-1, -1, -1):
                    if not pd.isnull(arr[k]):
                        prev_value = arr[k]
                        break
                for k in range(idx+1, n):
                    if not pd.isnull(arr[k]):
                        next_value = arr[k]
                        break
                candidates = [(abs(k - idx), val) for k, val in ((idx-1, prev_value), (idx+1, next_value)) if val is not None]
                if candidates:
                    df.at[idx, target_col] = sorted(candidates)[0][1]
    return df

def fill_remaining_gaps(df, fill_strategy="ffill", cols=None):
    # Самый логичный: ffill+bfill по колонке
    df = df.copy()
    if cols is None:
        cols = df.columns
    df[cols] = df[cols].ffill().bfill()
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
    window = 2
    min_neighbors = 1
    min_filled = 1
    n_rounds = 20

    # Zet-алгоритм для всех признаков
    for _ in range(n_rounds):
        changed_any = False
        for col in features:
            df, changed = zet_fill_window(df, col, window=window, min_neighbors=min_neighbors)
            changed_any = changed_any or changed
        if not changed_any:
            break

    # Регрессия для всех признаков по всему df
    for _ in range(n_rounds):
        na_sum_before = df[features].isna().sum().sum()
        for target in features:
            other_feats = [f for f in features if f != target]
            df = fill_linear_window_multi_iter(df, target, other_feats, window=window, min_filled=min_filled)
        na_sum_after = df[features].isna().sum().sum()
        if na_sum_after == na_sum_before:
            break

    # Финальное заполнение ffill+bfill — самый жесткий но логичный импьюинг!
    df = fill_remaining_gaps(df, fill_strategy="ffill", cols=features)

    df.to_csv("isodata_filled.csv", index=False)